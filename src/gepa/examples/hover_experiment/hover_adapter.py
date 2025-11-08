"""
Custom GEPA Adapter for HoVer Dataset
======================================

This adapter provides custom evaluation and reflection for the HoVer fact verification task.
"""

from typing import Any, Callable, TypedDict
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


_HERE = _PathForSys(__file__).resolve()
_ensure_local_src_on_path(_HERE.parent)

# Optional import of the local HF helper. Prefer a module-local copy in the
# examples folder (so this example runs even if package layout or PYTHONPATH
# differs). Fall back to the package helper if present.
try:
    from hf_local import get_local_hf_model
except Exception:  # pragma: no cover - optional runtime dependency
        get_local_hf_model = None


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
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data['input']}
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
                    if get_local_hf_model is None:
                        raise ImportError(
                            "Local HF helper not available. Install gepa.utils.hf_local or ensure it is in PYTHONPATH."
                        )

                    model_id = self.model.split("/", 1)[1]
                    local_model = get_local_hf_model(model_id)

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
- Consider including few-shot examples if helpful
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
