"""Local copy of HF helper moved into the HoVer example folder.

This is a copy of `gepa.utils.hf_local` placed next to the example so the
example can run in environments where the package layout or PYTHONPATH varies
 (e.g., HPC). Keep this file small and behaviour-compatible with the original.
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
        if self.is_downloaded():
            return
        self.load()

    def load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        kwargs = self._from_pretrained_kwargs()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=self.use_fast_tokenizer, **kwargs
        )

        model_kwargs = dict(kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
                **model_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code, **model_kwargs
            )

        if torch is not None and self.device.startswith("cuda"):
            try:
                self.model.to(self.device)
            except Exception:
                self.device = "cpu"
        else:
            self.model.to("cpu")

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7,
                 top_p: float = 0.9, do_sample: bool = True, stop_tokens: Optional[list[str]] = None) -> str:
        if self.tokenizer is None or self.model is None:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt")
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

        output_ids = outputs[0]
        input_len = inputs["input_ids"].shape[1]
        if output_ids.shape[0] == 0:
            return ""

        generated_ids = output_ids[input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
    return HFLocalModel(model_name=model_name, cache_dir=cache_dir, **kwargs)


__all__ = ["HFLocalModel", "get_local_hf_model"]
