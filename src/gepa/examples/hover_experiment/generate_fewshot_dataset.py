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
from typing import Any, Dict, List


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

    def __init__(self, model_name: str | None = None, sleep: float = 1.0):
        self.model_name = model_name or os.getenv("FEWSHOT_MODEL", "gpt-5")
        self.sleep = sleep

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
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            import re

            m = re.search(r"(\[.*\])", text, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None

        return parsed if parsed is not None else text

    def _row_to_outrow(self, idx: int, row: Dict[str, Any], few_shot_val) -> Dict[str, Any]:
        return {
            "id": idx,
            "input": row.get("input", ""),
            "answer": row.get("answer"),
            "additional_context": json.dumps(row.get("additional_context", {})),
            "few_shot": json.dumps(few_shot_val) if not isinstance(few_shot_val, str) else json.dumps({"raw": few_shot_val}),
        }

    def generate_for_examples(self, examples: List[Dict[str, Any]], out_file: str | None = None, max_rows: int | None = None):
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

        for i, row in enumerate(examples[:max_rows]):
            prompt = self._build_prompt_for_row(row)
            try:
                text = self._call_model(prompt)
                few_shot_val = self._parse_model_response(text)
            except Exception as e:
                few_shot_val = f"ERROR: {e}"

            out_row = self._row_to_outrow(i, row, few_shot_val)
            out_rows.append(out_row)

            if i % 10 == 0:
                print(f"Processed {i+1}/{min(len(examples), max_rows)} rows")

            time.sleep(self.sleep)

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
    gen.generate_for_examples(rows, out_file="hover_fewshot.csv", max_rows=100)


if __name__ == "__main__":
    main()
