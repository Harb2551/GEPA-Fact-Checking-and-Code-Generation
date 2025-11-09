"""Few-shot generator utility

Provides a `FewShotGenerator` class that can generate few-shot examples for
any supplied list of examples (train, validation, test). It keeps the same
parsing and CSV output format as the previous script but is reusable from
`train_hover.py` and `evaluate_test_set.py`.

Usage example:
        from generate_fewshot_dataset import FewShotGenerator
        gen = FewShotGenerator(model_name=os.getenv('FEWSHOT_MODEL', 'gpt-5'))
        gen.generate_for_examples(my_examples, out_file='hover_fewshot.csv', max_rows=100)

The generator will try `litellm` first and fall back to the OpenAI
Python client if available.
"""

from __future__ import annotations

import csv
import json
import os
import random
import string
import time
from typing import Any, Dict, List, Optional, Literal

# Optional pydantic for structured validation of LM outputs
try:
    from pydantic import BaseModel, ValidationError, validator
except Exception:
    BaseModel = None
    ValidationError = None
    validator = None


# Small synthetic text generator (no external deps)
WORDS = (
    "the quick brown fox jumps over lazy dog ai model data evidence claim verify"
).split()


def random_sentence(min_words=6, max_words=18) -> str:
    n = random.randint(min_words, max_words)
    return " ".join(random.choice(WORDS) for _ in range(n)).capitalize() + "."


def generate_rows(n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        claim = random_sentence(6, 14)
        context = random_sentence(20, 40)
        answer = random.choice(["SUPPORTED", "NOT_SUPPORTED"])  # synthetic label
        rows.append(
            {
                "id": i,
                "input": f"Claim: {claim}\nContext: {context}",
                "answer": answer,
                "additional_context": {},
            }
        )
    random.shuffle(rows)
    return rows


# Model caller abstraction: try litellm.completion, otherwise OpenAI chat completion

class FewShotGenerator:
    """Generate few-shot examples for a list of examples.

    Methods:
        - generate_for_examples(examples, out_file, max_rows)
    """

    def __init__(self, model_name: str | None = None, sleep: float = 1.0, max_workers: int = 4, chunk_size: int = 32, retries: int = 2):
        self.model_name = model_name or os.getenv("FEWSHOT_MODEL", "gpt-5")
        self.sleep = sleep
        # Parallel generation tuning
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.retries = retries

    def _call_model(self, prompt: str, model_name: str | None = None) -> str:
        model_name = model_name or self.model_name
        # Try litellm first
        try:
            import litellm

            messages = [{"role": "user", "content": prompt}]
            resp = litellm.completion(model=model_name, messages=messages)
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
            return str(resp)
        except Exception:
            pass

        # Fallback: openai
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            choices = resp.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return json.dumps(resp)
        except Exception as e:
            raise RuntimeError(
                "No supported LLM client found (litellm or openai). Install one and set appropriate env vars/API keys. "
                f"Underlying error: {e}"
            )


    def _build_prompt_for_row(self, row: Dict[str, Any]) -> str:
        header = (
            "You are given a claim verification example. Produce exactly 3 few-shot examples (short) "
            "that are formatted as JSON array of objects with keys: 'input' and 'label' where label is either 'SUPPORTED' or 'NOT_SUPPORTED'.\n\n"
        )
        original = "Original example:\n" + row["input"] + "\n\n"
        example_json = (
            "Return only a JSON array, e.g.:\n"
            "[\n"
            "  {\"input\": \"Claim: ... Context: ...\", \"label\": \"SUPPORTED\"},\n"
            "  {\"input\": \"Claim: ... Context: ...\", \"label\": \"NOT_SUPPORTED\"},\n"
            "  {\"input\": \"Claim: ... Context: ...\", \"label\": \"SUPPORTED\"}\n"
            "]\n\n"
        )
        footer = "Keep each example short (one or two sentences per 'input').\n"
        return header + original + example_json + footer


    def _parse_model_response(self, text: str):
        """
        Strictly parse and validate the model response using pydantic.

        This function expects the model output to be a JSON array of objects
        with keys 'input' (str) and 'label' (str). Validation is performed by
        a local pydantic model. If pydantic is not installed or validation
        fails, this function raises an exception — we intentionally do not
        fall back to heuristic parsing here to keep behavior deterministic.
        """
        if BaseModel is None:
            raise RuntimeError(
                "pydantic is required for strict LM output validation."
                " Install it with `pip install pydantic` and retry."
            )

        # Parse the raw text as JSON. If this fails, raise a clear error.
        try:
            parsed = json.loads(text)
        except Exception as e:
            raise ValueError(f"Failed to parse model output as JSON: {e}\nOutput:\n{text}")

        # Validate the structure with pydantic
        class _FewShotExample(BaseModel):
            input: str
            label: str

            @validator("label")
            def _label_must_be_supported(cls, v: str) -> str:
                if v is None:
                    raise ValueError("label missing")
                vv = v.strip().upper()
                if vv not in ("SUPPORTED", "NOT_SUPPORTED"):
                    raise ValueError("label must be SUPPORTED or NOT_SUPPORTED")
                return vv

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array of examples, got: {type(parsed)}")

        validated: List[Dict[str, Any]] = []
        errors: List[str] = []
        for idx, item in enumerate(parsed):
            try:
                m = _FewShotExample.parse_obj(item)
                validated.append({"input": m.input, "label": m.label})
            except ValidationError as ve:
                errors.append(f"item {idx}: {ve}")

        if errors:
            raise ValueError("Few-shot validation failed:\n" + "\n".join(errors))

        return validated

    def _row_to_outrow(self, idx: int, row: Dict[str, Any], few_shot_val) -> Dict[str, Any]:
        return {
            "id": idx,
            "input": row.get("input", ""),
            "answer": row.get("answer"),
            "additional_context": json.dumps(row.get("additional_context", {})),
            "few_shot": json.dumps(few_shot_val) if not isinstance(few_shot_val, str) else json.dumps({"raw": few_shot_val}),
        }

    def generate_for_examples(self, examples: List[Dict[str, Any]], out_file: str | None = None, max_rows: int | None = None, checkpoint_file: str | None = None, testset_file: str | None = None):
        """Generate few-shot entries for the supplied examples.

        examples: list of dicts with at least 'input' and optionally 'answer' and 'additional_context'
        out_file: if provided, write CSV in the same format as before
        max_rows: limit how many rows to process (None -> all)
        Returns list of output rows (dicts)
        """
        if max_rows is None:
            max_rows = len(examples)

        fieldnames = ["id", "input", "answer", "additional_context", "few_shot"]
        out_rows: List[Dict[str, Any]] = []

        # Build or load the testset (so random selection is reproducible across runs)
        if testset_file and os.path.exists(testset_file):
            try:
                with open(testset_file, 'r', encoding='utf-8') as tf:
                    rows_to_process = json.load(tf)
                # Ensure we don't exceed requested max_rows
                rows_to_process = rows_to_process[:max_rows]
            except Exception:
                print(f"Warning: failed to read testset_file {testset_file}; falling back to provided examples")
                rows_to_process = examples[:max_rows]
        else:
            rows_to_process = examples[:max_rows]
            if testset_file:
                try:
                    with open(testset_file, 'w', encoding='utf-8') as tf:
                        json.dump(rows_to_process, tf, ensure_ascii=False, indent=2)
                except Exception:
                    print(f"Warning: failed to write testset_file {testset_file}")

        prompts = [self._build_prompt_for_row(r) for r in rows_to_process]

        outputs: List[str] = []

        # Load checkpoint which now is a JSON file storing processed indices
        processed_indices = set()
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as cf:
                    data = json.load(cf)
                    if isinstance(data, dict) and 'processed_indices' in data:
                        processed_indices = set(data.get('processed_indices', []))
            except Exception:
                print(f"Warning: failed to read checkpoint file {checkpoint_file}; starting fresh")

        # Also detect any already-populated few_shot fields in the testset and mark them processed
        for i, r in enumerate(rows_to_process):
            if r.get('few_shot') is not None:
                processed_indices.add(i)

        # Helper: per-prompt retry using the single-call _call_model (used on
        # parse failures or as a fallback for stubborn rows).
        def _single_retry_generate(prompt_text: str, attempts: int = None) -> str:
            attempts = attempts if attempts is not None else self.retries
            last_exc = None
            for attempt in range(attempts + 1):
                try:
                    return self._call_model(prompt_text)
                except Exception as e:
                    last_exc = e
                    time.sleep(min(2 ** attempt, 5))
            # If all attempts fail, return an error string to be recorded
            return f"ERROR: {last_exc}"

        # Single-call mode: call the model per-prompt using the retry helper.
        # We removed the batched API to keep behavior simple and deterministic.
        outputs = []
        parsed_list: List[Optional[Any]] = []
        total = len(prompts)

        def _print_progress(cur: int, total: int):
            pct = int((cur / total) * 100) if total else 100
            width = 30
            filled = int((pct / 100) * width)
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r[{bar}] {pct:3d}% ({cur}/{total})", end="", flush=True)

        counts = 0
        for idx_prompt, prompt_text in enumerate(prompts, start=1):
            i = idx_prompt - 1
            # progress before sending
            _print_progress(i, total)

            # If this index is already processed (from checkpoint or testset), reuse it
            if i in processed_indices:
                existing = rows_to_process[i].get('few_shot')
                txt = json.dumps(existing, ensure_ascii=False)
                parsed_val = existing
                outputs.append(txt)
                parsed_list.append(parsed_val)
                print(f"\nResuming from checkpoint for {idx_prompt}/{total}: index={i}")
                # progress after sending
                _print_progress(idx_prompt, total)
                counts += 1
                if counts % 10 == 0:
                    print(f"\nGenerated {counts}/{total} prompts (from checkpoint)", flush=True)
                continue

            try:
                txt = _single_retry_generate(prompt_text)
                counts += 1
            except Exception as e:
                txt = f"ERROR: {e}"

            # attempt to parse immediately and show parsed output
            parsed_val: Optional[Any] = None
            try:
                parsed_val = self._parse_model_response(txt)
                try:
                    pretty = json.dumps(parsed_val, indent=2, ensure_ascii=False)
                except Exception:
                    pretty = str(parsed_val)
                print(f"\nParsed response for {idx_prompt}/{total}:\n{pretty}")

                # persist the successful parsed result: update rows_to_process and checkpoint/testset files
                try:
                    rows_to_process[i]['few_shot'] = parsed_val
                    processed_indices.add(i)
                    if checkpoint_file:
                        tmp = checkpoint_file + '.tmp'
                        with open(tmp, 'w', encoding='utf-8') as cf:
                            json.dump({'processed_indices': sorted(list(processed_indices))}, cf, ensure_ascii=False, indent=2)
                        os.replace(tmp, checkpoint_file)
                    if testset_file:
                        tmp2 = testset_file + '.tmp'
                        with open(tmp2, 'w', encoding='utf-8') as tf:
                            json.dump(rows_to_process, tf, ensure_ascii=False, indent=2)
                        os.replace(tmp2, testset_file)
                except Exception as e:
                    print(f"Warning: failed to persist checkpoint/testset for index={i}: {e}")

            except Exception as e:
                print(f"\nParse failed for {idx_prompt}/{total}: {e}\nRaw: {txt}")

            outputs.append(txt)
            parsed_list.append(parsed_val)

            # progress after sending
            _print_progress(idx_prompt, total)
            # notify every 10 generated prompts
            if counts % 10 == 0:
                print(f"\nGenerated {counts}/{total} prompts", flush=True)

        # finish progress line
        print()

        # Post-process outputs and build CSV rows. If parsing returns a raw
        # string (not a JSON array), retry a few times using single-call
        # generator to reduce spurious error captures.
        for i, (row, out_text) in enumerate(zip(rows_to_process, outputs)):
            few_shot_val = None
            try:
                # reuse parsed value if available from generation step
                parsed = parsed_list[i] if i < len(parsed_list) else None
                if parsed is None:
                    parsed = self._parse_model_response(out_text)
                # If parsed is a plain string (i.e., parsing failed), retry
                if isinstance(parsed, str):
                    # Try a few single-call retries
                    for attempt in range(self.retries):
                        retry_text = _single_retry_generate(prompts[i])
                        parsed2 = self._parse_model_response(retry_text)
                        if not isinstance(parsed2, str):
                            parsed = parsed2
                            out_text = retry_text
                            break
                    few_shot_val = parsed
                else:
                    few_shot_val = parsed
            except Exception as e:
                few_shot_val = f"ERROR: {e}"

            out_row = self._row_to_outrow(i, row, few_shot_val)
            out_rows.append(out_row)

            if i % 10 == 0:
                print(f"Processed {i+1}/{min(len(examples), max_rows)} rows")

        if out_file:
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in out_rows:
                    writer.writerow(r)
            print(f"Wrote augmented dataset to {out_file}")

        return out_rows


def main():
    # Backwards-compatible CLI: behave like previous script when executed directly
    gen = FewShotGenerator()
    rows = generate_rows(100)
    checkpoint_json = "hover_fewshot.csv.checkpoint.json"
    testset_json = "hover_fewshot.csv.testset.json"
    gen.generate_for_examples(rows, out_file="hover_fewshot.csv", max_rows=100, checkpoint_file=checkpoint_json, testset_file=testset_json)


if __name__ == "__main__":
    main()
