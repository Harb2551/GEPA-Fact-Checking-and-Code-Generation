"""
Custom GEPA Adapter for HoVer Dataset
======================================

This adapter provides custom evaluation and reflection for the HoVer fact verification task.
"""

from typing import Any, Callable, TypedDict
from typing import Optional
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

import sys
from pathlib import Path as _PathForSys


# Try to ensure the project's local `src/` directory is on sys.path so
# `from gepa.utils.hf_local import ...` can find the helper when this
# module is executed directly (e.g., on HPC or after cloning into arbitrary paths).
def _ensure_local_src_on_path(start: _PathForSys, max_up: int = 8) -> None:
    cur = start
    for _ in range(max_up):
        candidate = cur / "src"
        if candidate.exists() and (candidate / "gepa").exists():
            p = str(candidate)
            if p not in sys.path:
                sys.path.insert(0, p)
            # Debug-friendly message when running on remote machines
            print(f"[gepa] added to sys.path: {p}")
            return
        cur = cur.parent


# Inlined HFLocalModel: a lazy-loading, minimal helper to run a HF causal LM
# locally using `transformers`. Imports heavy ML libraries only inside
# methods to avoid import-time failures on systems without torch/cuda.
class HFLocalModel:
    """Lightweight local HF causal LM helper (lazy imports).

    This loads `transformers` and `torch` only when needed (in `load()`).
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

        self._transformers = None
        self._torch = None
        self.AutoTokenizer = None
        self.AutoModelForCausalLM = None

        self.device = device or "cpu"

        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None

    def _from_pretrained_kwargs(self) -> dict:
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    def _ensure_transformers_loaded(self):
        if self._transformers is not None:
            return
        try:
            import transformers as _transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise ImportError("HFLocalModel requires the 'transformers' package: %s" % e)
        try:
            import torch as _torch
        except Exception:
            _torch = None

        self._transformers = _transformers
        self._torch = _torch
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM

        if self.device == "cpu" and self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self.device = "cuda"
            except Exception:
                self.device = "cpu"

    def is_downloaded(self) -> bool:
        self._ensure_transformers_loaded()
        try:
            self.AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True, use_fast=self.use_fast_tokenizer
            )
            self.AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=True)
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

        self._ensure_transformers_loaded()
        kwargs = self._from_pretrained_kwargs()

        self.tokenizer = self.AutoTokenizer.from_pretrained(
            self.model_name, use_fast=self.use_fast_tokenizer, **kwargs
        )

        model_kwargs = dict(kwargs)
        try:
            self.model = self.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
                **model_kwargs,
            )
        except TypeError:
            self.model = self.AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code, **model_kwargs
            )

        if self._torch is not None and str(self.device).startswith("cuda"):
            try:
                self.model.to(self.device)
            except Exception:
                self.device = "cpu"
        else:
            try:
                self.model.to("cpu")
            except Exception:
                pass

        try:
            self.model.eval()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_tokens: Optional[list[str]] = None,
    ) -> str:
        if self.tokenizer is None or self.model is None:
            self.load()

        _torch = self._torch

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if _torch is not None and str(self.device).startswith("cuda"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=(self.tokenizer.eos_token_id or getattr(self.tokenizer, "pad_token_id", None)),
        )

        ctx = _torch.no_grad() if _torch is not None else _nullcontext()
        with ctx:
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



# Define data types
class HoVerDataInst(TypedDict):
    input: str  # Claim + Context
    answer: str  # SUPPORTED or NOT_SUPPORTED
    additional_context: dict[str, str]


class HoVerTrajectory(TypedDict):
    data: HoVerDataInst
    full_response: str
    predicted_label: str


class HoVerRolloutOutput(TypedDict):
    full_response: str
    predicted_label: str


class HoVerAdapter(GEPAAdapter[HoVerDataInst, HoVerTrajectory, HoVerRolloutOutput]):
    """
    Custom adapter for HoVer fact verification task.
    
    This adapter:
    1. Evaluates candidates by checking if predicted label matches ground truth
    2. Provides detailed feedback for reflection including reasoning about failures
    3. Extracts predicted labels from LLM responses
    """
    
    def __init__(
        self,
        model: str | Callable,
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] = {},
    ):
        if isinstance(model, str):
            import litellm
            self.litellm = litellm
        self.model = model
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs
    
    def _extract_label(self, response: str) -> str:
        """Extract SUPPORTED or NOT_SUPPORTED from LLM response.

        Normalize whitespace/hyphens to avoid substring collisions like
        'NOT SUPPORTED' being misread as 'SUPPORTED'. Prioritize negative
        cues before positive ones.
        """
        response_upper = response.upper()
        norm = response_upper.replace(" ", "_").replace("-", "_")

        # Negative cues first
        if (
            "NOT_SUPPORTED" in norm
            or "NOTSUPPORT" in norm
            or "REFUTE" in norm
            or "REFUTED" in norm
            or "CONTRADICT" in norm
        ):
            return "NOT_SUPPORTED"

        # Positive cues
        if "SUPPORTED" in norm or "SUPPORTS" in norm or "ENTAIL" in norm:
            return "SUPPORTED"

        # Default conservative choice
        return "NOT_SUPPORTED"
    
    def evaluate(
        self,
        batch: list[HoVerDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[HoVerTrajectory, HoVerRolloutOutput]:
        """
        Evaluate candidate on a batch of HoVer examples.
        
        Args:
            batch: List of HoVer examples (claim + context)
            candidate: Dict with 'system_prompt' key
            capture_traces: Whether to capture detailed trajectories
        
        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories
        """
        outputs: list[HoVerRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[HoVerTrajectory] | None = [] if capture_traces else None
        
        system_content = candidate['system_prompt']
        
        # Prepare batch requests for LLM
        litellm_requests = []
        for data in batch:
            # If the example contains few-shot examples (attached by train_hover),
            # format them and prepend to the user content so the task model sees
            # few-shot examples inline.
            user_content = data['input']
            if isinstance(data, dict) and data.get("few_shot"):
                few = data.get("few_shot")
                # few can be a list of example dicts or a raw string wrapper
                try:
                    if isinstance(few, list):
                        examples_text = []
                        for i_ex, ex in enumerate(few, start=1):
                            inp = ex.get("input") if isinstance(ex, dict) else str(ex)
                            lbl = ex.get("label") if isinstance(ex, dict) else ""
                            examples_text.append(f"Example {i_ex}: {inp}\nLabel: {lbl}")
                        few_text = "\n\n".join(examples_text)
                    elif isinstance(few, dict) and "raw" in few:
                        few_text = few["raw"]
                    else:
                        few_text = str(few)
                except Exception:
                    few_text = str(few)

                user_content = f"Few-shot examples:\n{few_text}\n\nOriginal input:\n{user_content}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            litellm_requests.append(messages)
        
        # Get LLM responses
        try:
            if isinstance(self.model, str):
                # Support routing to a local Hugging Face model using the
                # `hf/<model_id>` prefix. Example: TASK_LM="hf/gpt2" or
                # "hf/your-org/your-model". This will use the HFLocalModel
                # helper to download (if needed) and run generation locally.
                if self.model.startswith("hf/"):
                    # Directly use the in-file HFLocalModel class to avoid any
                    # external-import indirection. The HFLocalModel performs
                    # lazy imports of heavy ML libs so importing this module
                    # still works on systems without torch/transformers.
                    model_id = self.model.split("/", 1)[1]
                    local_model = HFLocalModel(model_id)

                    responses = []
                    # convert the message list to a single prompt string per example
                    for messages in litellm_requests:
                        # messages expected to be [{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}]
                        parts = []
                        for m in messages:
                            parts.append(f"[{m.get('role', '')}] {m.get('content','')}")
                        prompt = "\n\n".join(parts)
                        try:
                            out = local_model.generate(prompt)
                        except Exception as e:
                            print(f"Warning: local HF generation failed: {e}")
                            out = ""
                        responses.append(out)
                else:
                    raw_responses = self.litellm.batch_completion(
                        model=self.model,
                        messages=litellm_requests,
                        max_workers=self.max_litellm_workers,
                        **self.litellm_batch_completion_kwargs
                    )
                    # Extract content, handling errors
                    responses = []
                    for resp in raw_responses:
                        if hasattr(resp, 'choices') and len(resp.choices) > 0:
                            responses.append(resp.choices[0].message.content.strip())
                        else:
                            # Error response (like RateLimitError)
                            print(f"Warning: LLM call failed with: {resp}")
                            responses.append("")  # Empty response will get low score
            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception as e:
            print(f"Error during LLM call: {e}")
            # Return failure scores for all examples in batch
            outputs = [{"full_response": "", "predicted_label": "NOT_SUPPORTED"} for _ in batch]
            scores = [self.failure_score for _ in batch]
            trajectories_out = None if not capture_traces else [
                {"data": data, "full_response": "", "predicted_label": "NOT_SUPPORTED"}
                for data in batch
            ]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories_out)
        
        # Process each response
        for data, response in zip(batch, responses, strict=False):
            predicted_label = self._extract_label(response)
            correct_label = data['answer']

            # Normalize both labels for robust comparison
            norm_pred = predicted_label.strip().upper()
            norm_correct = str(correct_label).strip().upper()

            # Score: 1.0 if normalized labels match, else 0.0
            score = 1.0 if norm_pred == norm_correct else 0.0
            
            output = {
                "full_response": response,
                "predicted_label": predicted_label
            }
            
            outputs.append(output)
            scores.append(score)
            
            if capture_traces:
                trajectories.append({
                    "data": data,
                    "full_response": response,
                    "predicted_label": predicted_label
                })
        
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
    
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[HoVerTrajectory, HoVerRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build reflective dataset for prompt improvement.
        
        This extracts failures and provides detailed feedback for the LLM to reflect on.
        
        Args:
            candidate: Current candidate being evaluated
            eval_batch: Results from evaluate() with trajectories
            components_to_update: Components to generate reflection data for
        
        Returns:
            Dict mapping component names to lists of reflection examples
        """
        ret_d: dict[str, list[dict[str, Any]]] = {}
        
        assert len(components_to_update) == 1, "HoVer adapter currently supports single component"
        comp = components_to_update[0]
        
        items: list[dict[str, Any]] = []
        trace_instances = list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False))
        
        for traj, score, output in trace_instances:
            data = traj['data']
            predicted_label = traj['predicted_label']
            correct_label = data['answer']
            full_response = traj['full_response']
            
            # Simple, direct feedback
            status = "CORRECT" if score > 0.0 else "INCORRECT"
            
            feedback = f"""{status}: Model predicted "{predicted_label}", correct answer is "{correct_label}".

Input: {data['input']}

Response: {full_response}

Notes:
- Keep prompt concise - the task model is ~8B parameters, avoid over-optimization
- Focus on clarity over complexity"""
            
            # Create reflection example
            reflection_item = {
                "Status": status,
                "Input": data['input'],
                "Model Response": full_response,
                "Predicted": predicted_label,
                "Correct Answer": correct_label,
                "Feedback": feedback,
                "Score": score
            }
            
            items.append(reflection_item)
        
        ret_d[comp] = items
        
        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")
        
        return ret_d
