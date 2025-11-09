"""Generate a synthetic HoVer-style dataset and augment it with few-shot examples
obtained from a language model (default: 'gpt-5').

Behavior:
- Generates N random rows (default 100) with fields: input, answer, additional_context
- For each row, calls an LLM to request 3-shot examples (as a JSON array or newline-separated)
  and stores the returned few-shot examples in the 'few_shot' column.
- Writes the result to CSV (default: hover_fewshot.csv) which `train_hover.py` can use
  when a few-shot variable is enabled.

Usage:
  export OPENAI_API_KEY="sk-..."   # or ensure litellm is configured
  export FEWSHOT_MODEL="gpt-5"      # optional, default 'gpt-5'
  python generate_fewshot_dataset.py --rows 100 --out hover_fewshot.csv

Notes:
- This script will attempt to use `litellm.completion` if `litellm` is installed.
  If not, it will try the OpenAI Python package. If neither is available, it will
  raise an instructive error.
- Calls are rate-limited by a short sleep; adjust `SLEEP_BETWEEN` if needed.
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

def call_model_for_fewshot(prompt: str, model_name: str, timeout=60) -> str:
    """Return the model's raw text response for the given prompt and model_name.

    Attempts to use litellm.completion(model=..., messages=...), falling back to
    openai.ChatCompletion if litellm is unavailable.
    """
    # Try litellm first
    try:
        import litellm

        # Build messages as litellm expects
        messages = [{"role": "user", "content": prompt}]
        resp = litellm.completion(model=model_name, messages=messages)
        # litellm completion returns an object; try to extract content
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)
        return str(resp)
    except Exception:
        pass

    # Fallback: try openai package
    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
        # Use chat completions for modern models
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        # extract text
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return json.dumps(resp)
    except Exception as e:
        raise RuntimeError(
            "No supported LLM client found (litellm or openai). Install one and set appropriate env vars/API keys. "
            f"Underlying error: {e}"
        )


def build_prompt_for_row(row: Dict[str, Any]) -> str:
    """Create a prompt asking the model to produce 3-shot examples for the input.

    The prompt requests 3 short example inputs and labels in JSON array form so
    we can store them cleanly.
    """
    # Build the prompt in parts to avoid interpreting JSON braces as f-string
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

    prompt = header + original + example_json + footer
    return prompt


# Configuration variables (edit these directly)
ROWS = 100
OUT = "hover_fewshot.csv"
MODEL = os.getenv("FEWSHOT_MODEL", "gpt-5")
SLEEP = 1.0


def main() -> None:
    rows = generate_rows(ROWS)

    fieldnames = ["id", "input", "answer", "additional_context", "few_shot"]

    print(f"Generating {ROWS} rows, model={MODEL}, out={OUT}, sleep={SLEEP}s")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            prompt = build_prompt_for_row(row)
            try:
                text = call_model_for_fewshot(prompt, MODEL)
                # try to parse JSON out of the model response; be forgiving
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    # attempt to extract the first JSON array in text
                    import re

                    m = re.search(r"(\[.*\])", text, flags=re.S)
                    if m:
                        try:
                            parsed = json.loads(m.group(1))
                        except Exception:
                            parsed = None

                few_shot_val = parsed if parsed is not None else text
            except Exception as e:
                few_shot_val = f"ERROR: {e}"

            out_row = {
                "id": row["id"],
                "input": row["input"],
                "answer": row["answer"],
                "additional_context": json.dumps(row["additional_context"]),
                "few_shot": json.dumps(few_shot_val) if not isinstance(few_shot_val, str) else json.dumps({"raw": few_shot_val}),
            }
            writer.writerow(out_row)

            if i % 10 == 0:
                print(f"Processed {i+1}/{len(rows)} rows")

            time.sleep(SLEEP)

    print(f"Wrote augmented dataset to {OUT}")


if __name__ == "__main__":
    main()
