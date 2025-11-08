"""Small helper to download and run a Hugging Face causal LM locally using
transformers' AutoTokenizer and AutoModelForCausalLM.

This provides a thin, dependency-light wrapper that:
- downloads model/tokenizer when first used (via from_pretrained)
- exposes a simple `generate` method that returns the generated continuation

Usage example:
    from gepa.utils.hf_local import HFLocalModel

    m = HFLocalModel("gpt2")
    # this will download if not already present
    m.download_if_needed()
    out = m.generate("Write a short poem about AI.")
    print(out)

Notes:
- Requires the `transformers` package (and optionally `torch`) installed in the
  environment. On macOS/Apple Silicon you may want a wheel from conda/miniforge
  or to install with extra care for the torch backend.
"""
from __future__ import annotations

import os
from typing import Optional

import transformers

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed in some envs
    torch = None

from transformers import AutoModelForCausalLM, AutoTokenizer


class HFLocalModel:
    """Manage a local HF causal LM and provide a simple generate() method.

    The class lazily downloads model files when `download_if_needed()` or
    `load()` is called. `generate()` will call `load()` automatically.

    Args:
        model_name: model identifier on the Hub (e.g. "gpt2" or a path).
        cache_dir: optional transformers cache dir.
        device: torch device string (e.g. "cpu", "cuda:0"). If None, the
            helper will try to use CUDA if available and fall back to CPU.
        trust_remote_code: pass through to `from_pretrained` when model needs
            remote code execution.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_fast_tokenizer = use_fast_tokenizer
        self.trust_remote_code = trust_remote_code

        # decide device
        if device is not None:
            self.device = device
        else:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def _from_pretrained_kwargs(self) -> dict:
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    def is_downloaded(self) -> bool:
        """Return True if tokenizer and model files appear to be present locally.

        This uses `local_files_only=True` which will raise if files are missing.
        """
        try:
            AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True, use_fast=self.use_fast_tokenizer
            )
            AutoModelForCausalLM.from_pretrained(
                self.model_name, local_files_only=True
            )
            return True
        except Exception:
            return False

    def download_if_needed(self) -> None:
        """Download tokenizer and model files if they are not present locally.

        This is idempotent: if files already exist this is a no-op.
        """
        if self.is_downloaded():
            return
        # call from_pretrained normally which will download the files
        self.load()

    def load(self) -> None:
        """Load (and if needed download) tokenizer and model into memory.

        After this returns `self.tokenizer` and `self.model` are set.
        """
        if self.tokenizer is not None and self.model is not None:
            return

        kwargs = self._from_pretrained_kwargs()

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=self.use_fast_tokenizer, **kwargs
        )

        # model
        # Use low_cpu_mem_usage when available to reduce memory spikes on load.
        model_kwargs = dict(kwargs)
        try:
            # some transformers versions support low_cpu_mem_usage
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
                **model_kwargs,
            )
        except TypeError:
            # fallback if `low_cpu_mem_usage` isn't supported
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code, **model_kwargs
            )

        # move to device
        if torch is not None and self.device.startswith("cuda"):
            try:
                self.model.to(self.device)
            except Exception:
                # if moving fails, keep model on CPU
                self.device = "cpu"
        else:
            # ensure model is on CPU
            self.model.to("cpu")

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_tokens: Optional[list[str]] = None,
    ) -> str:
        """Generate text from `prompt` and return only the newly generated continuation.

        This method will call `load()` automatically if the model isn't loaded.
        """
        if self.tokenizer is None or self.model is None:
            self.load()

        assert self.tokenizer is not None and self.model is not None

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # move inputs to model device
        if torch is not None and self.device.startswith("cuda"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
        )

        with torch.no_grad() if torch is not None else _nullcontext():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # outputs: batch x seq_len. We only handle single-input prompts here.
        output_ids = outputs[0]
        # skip the input tokens to return only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        if output_ids.shape[0] == 0:
            return ""

        generated_ids = output_ids[input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # apply simple stop token truncation if provided
        if stop_tokens:
            for t in stop_tokens:
                idx = text.find(t)
                if idx != -1:
                    text = text[:idx]
                    break

        return text.strip()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _nullcontext():
    return _NullContext()


def get_local_hf_model(model_name: str, cache_dir: Optional[str] = None, **kwargs) -> HFLocalModel:
    """Convenience factory that returns a ready HFLocalModel instance.

    The returned object will still lazily download on `load()`/`generate()`, so
    callers can call `download_if_needed()` explicitly when desired.
    """
    return HFLocalModel(model_name=model_name, cache_dir=cache_dir, **kwargs)


__all__ = ["HFLocalModel", "get_local_hf_model"]
