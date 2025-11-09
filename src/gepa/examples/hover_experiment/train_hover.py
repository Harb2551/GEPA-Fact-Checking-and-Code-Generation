"""
HoVer Dataset Optimization with GEPA
=====================================

This script optimizes prompts for fact verification on the HoVer dataset using GEPA.

Usage:
    python train_hover.py
"""

from datasets import load_dataset
import os
import random
import sys
from pathlib import Path as _PathForSys

# When running this example directly (python train_hover.py) ensure the
# project's `src/` directory is on sys.path so `import gepa` and
# `from gepa.utils.hf_local import ...` work even when this script's CWD is
# the examples folder. This mirrors running the code as an installed package.
_HERE = _PathForSys(__file__).resolve()
_SRC_DIR = _HERE.parents[3]  # ../../.. -> project `src` directory
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gepa
from hover_adapter import HoVerAdapter
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from experiment_logger import ExperimentLogger
from evaluation.data_formatter import DataFormatter
import csv
import json

# Load environment variables from .env file
# Load environment variables from repo root (searching upward)
def _load_env():
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            break
    else:
        # Fallback: default behavior (will look at CWD)
        load_dotenv()

_load_env()

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================

# Set to True to use custom adapter, False to use default adapter
USE_CUSTOM_ADAPTER = True

# Checkpoint behavior
USE_TIMESTAMP_DIR = True  # Set to False to resume from existing directory
RESUME_FROM_CHECKPOINT = True  # Set to True to resume from checkpoint

# Dataset sizes
TRAIN_SIZE = 100  # Smaller training set to avoid overfitting
VAL_SIZE = 30  # Smaller validation set

MAX_METRIC_CALLS = 200  # Reduced budget to prevent over-optimization
REFLECTION_MINIBATCH_SIZE = 5  # Smaller batches for gentler optimization

LITELLM_MAX_WORKERS = 2  # Reduce parallel requests to avoid rate limits

TASK_LM = "gpt-4.1-mini"  # LLM for task execution
REFLECTION_LM = "gpt-5"  # LLM for reflection

# Few-shot configuration: when True, train_hover will look for FEWSHOT_FILE;
# if it's missing it will run the local generator to create it, then attach
# the few-shot examples to training rows under the 'few_shot' key.
USE_FEWSHOT = True
FEWSHOT_FILE = "hover_fewshot.csv"

# Set seed for reproducibility
random.seed(42)

def hover_to_gepa_format(example):
    """Convert a HoVer example to GEPA format using the shared DataFormatter.

    This ensures label mapping and evidence formatting are consistent between
    training and evaluation (0 -> SUPPORTED, 1 -> NOT_SUPPORTED, with robust
    handling of variants).
    """
    return DataFormatter.hover_to_gepa_format(example)


def main():
    # 1. Load HoVer dataset from Hugging Face
    print("Loading HoVer dataset...")
    ds = load_dataset("Dzeniks/hover")
    
    # Get the dataset (adjust based on available splits)
    full_data = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
    print(f"Total examples: {len(full_data)}")
    
    # 2. Convert all examples
    print("Converting to GEPA format...")
    all_examples = [hover_to_gepa_format(ex) for ex in full_data]
    
    # 3. Shuffle and split 80/20
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.8)
    
    trainset = all_examples[:split_idx]
    valset = all_examples[split_idx:]
    
    print(f"Training examples: {len(trainset)}")
    print(f"Validation examples: {len(valset)}")

    # Track whether a few-shot file was used/generated for this run. This
    # will be stored in the experiment config so runs can record if few-shot
    # was active and whether the CSV was present or created.
    few_shot_file_used = False

    # If few-shot mode is enabled, prefer using the FEWSHOT_FILE as the
    # authoritative source for the training set so few-shot rows map exactly
    # to the examples they were generated for. If the file is missing, run the
    # generator to create it.
    if USE_FEWSHOT:
        if not os.path.exists(FEWSHOT_FILE):
            print(f"Few-shot file '{FEWSHOT_FILE}' not found — generating it now...")
            try:
                # Use the reusable FewShotGenerator class
                from generate_fewshot_dataset import FewShotGenerator

                gen = FewShotGenerator(model_name=os.getenv("FEWSHOT_MODEL", None), sleep=1.0)
                # Generate few-shot examples based on the canonical all_examples list
                # Persist a JSON checkpoint and a deterministic testset JSON alongside the CSV
                checkpoint_json = FEWSHOT_FILE + ".checkpoint.json"
                testset_json = FEWSHOT_FILE + ".testset.json"
                gen.generate_for_examples(
                    all_examples,
                    out_file=FEWSHOT_FILE,
                    max_rows=TRAIN_SIZE,
                    checkpoint_file=checkpoint_json,
                    testset_file=testset_json,
                )
            except Exception as e:
                print(f"Failed to generate few-shot dataset: {e}")
                raise
            # If generation succeeded, mark the CSV as used/created
            if os.path.exists(FEWSHOT_FILE):
                few_shot_file_used = True
        else:
            # File already present and will be used
            few_shot_file_used = True

        # Load CSV rows produced by the generator. Expect columns: id,input,answer,additional_context,few_shot
        csv_rows = []
        with open(FEWSHOT_FILE, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                try:
                    few = json.loads(r.get("few_shot", "null"))
                except Exception:
                    few = {"raw": r.get("few_shot", "")}
                csv_rows.append({
                    "input": r.get("input", ""),
                    "answer": r.get("answer", ""),
                    "additional_context": json.loads(r.get("additional_context", "{}")) if r.get("additional_context") else {},
                    "few_shot": few,
                })

        # record how many rows came from the few-shot CSV
        few_shot_rows = len(csv_rows)

        # Build lookup from HF-derived examples by exact input match
        input_to_example = {ex['input']: ex for ex in all_examples}

        # Construct trainset from CSV rows, preferring the canonical HF example
        trainset = []
        for row in csv_rows:
            inp = row['input']
            if inp in input_to_example:
                ex = input_to_example[inp].copy()
                ex['few_shot'] = row.get('few_shot')
            else:
                ex = {
                    'input': inp,
                    'answer': row.get('answer'),
                    'additional_context': row.get('additional_context', {}),
                    'few_shot': row.get('few_shot')
                }
            trainset.append(ex)

        # Validation set: examples not used in trainset
        train_inputs = {ex['input'] for ex in trainset}
        valset = [ex for ex in all_examples if ex['input'] not in train_inputs]

        print(f"Training examples (from CSV): {len(trainset)}")
        print(f"Validation examples (remaining): {len(valset)}")
    else:
        few_shot_rows = 0
        few_shot_file_used = False
    
    # 4. Preview one example
    print("\n" + "="*60)
    print("EXAMPLE:")
    print("="*60)
    print(f"Input:\n{trainset[0]['input']}\n")
    print(f"Expected Answer: {trainset[0]['answer']}")
    print("="*60 + "\n")
    
    # 5. Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # 6. Define seed prompt (this is what GEPA will optimize)
    seed_prompt = {
        "system_prompt": """Given a claim and an evidence Answer SUPPORTED or NOT_SUPPORTED."""
    }
    
    # 7. Set up adapter
    if USE_CUSTOM_ADAPTER:
        print(f"\n{'='*60}")
        print("Using CUSTOM HoVerAdapter with:")
        print("  - Custom label extraction")
        print("  - Detailed feedback generation")
        print("  - Enhanced reflection data")
        print(f"  - Max workers: {LITELLM_MAX_WORKERS}")
        print(f"{'='*60}\n")
        adapter = HoVerAdapter(
            model=TASK_LM,
            max_litellm_workers=LITELLM_MAX_WORKERS
        )
    else:
        print(f"\n{'='*60}")
        print("Using DEFAULT adapter")
        print(f"{'='*60}\n")
        adapter = None  # Will use DefaultAdapter
    
    # 8. Determine run directory
    if USE_TIMESTAMP_DIR:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"./results/gepa_hover_results_{timestamp}"
        print(f"Creating new run directory: {run_dir}")
    else:
        run_dir = "./results/gepa_hover_results_20251108_005848"
        if RESUME_FROM_CHECKPOINT and os.path.exists(f"{run_dir}/gepa_state.bin"):
            print(f"Resuming from checkpoint in: {run_dir}")
        elif os.path.exists(f"{run_dir}/gepa_state.bin"):
            print(f"WARNING: Checkpoint exists but RESUME_FROM_CHECKPOINT=False")
            print(f"Deleting old checkpoint to start fresh...")
            import shutil
            shutil.rmtree(run_dir)
        else:
            print(f"Starting fresh in: {run_dir}")
    
    # 9. Run GEPA optimization
    print("Starting GEPA optimization...")
    print(f"Using: {TRAIN_SIZE} train, {VAL_SIZE} val")
    print(f"Minibatch size: {REFLECTION_MINIBATCH_SIZE} examples per iteration")
    print(f"Budget: {MAX_METRIC_CALLS} metric calls")
    print(f"Task LM: {TASK_LM}")
    print(f"Reflection LM: {REFLECTION_LM}\n")
    
    optimize_kwargs = {
        "seed_candidate": seed_prompt,
        "trainset": trainset[:TRAIN_SIZE],
        "valset": valset[:VAL_SIZE],
        "reflection_lm": REFLECTION_LM,
        "reflection_minibatch_size": REFLECTION_MINIBATCH_SIZE,
        "max_metric_calls": MAX_METRIC_CALLS,
        "run_dir": run_dir,
        "display_progress_bar": True
    }
    
    if adapter is not None:
        # Use custom adapter
        optimize_kwargs["adapter"] = adapter
    else:
        # Use default adapter
        optimize_kwargs["task_lm"] = TASK_LM
    
    result = gepa.optimize(**optimize_kwargs)
    
    # 10. Display results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\nOriginal Prompt:")
    print("-" * 60)
    print(seed_prompt['system_prompt'])
    print("\n" + "="*60)
    print(f"\nOptimized Prompt:")
    print("-" * 60)
    print(result.best_candidate['system_prompt'])
    print("\n" + "="*60)
    
    # Get best score using the correct attribute
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest Score: {best_score:.2%}")
    print(f"Total Metric Calls: {result.total_metric_calls}")
    print(f"Number of Candidates Evaluated: {result.num_candidates}")
    print("="*60)
    
    # 11. Save the optimized prompt
    output_file = f"{run_dir}/optimized_prompt.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(result.best_candidate['system_prompt'])
    print(f"\nOptimized prompt saved to: {output_file}")
    
    # 12. Save hyperparameters and results to JSON
    hyperparams = {
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "reflection_minibatch_size": REFLECTION_MINIBATCH_SIZE,
        "max_metric_calls": MAX_METRIC_CALLS,
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "use_custom_adapter": USE_CUSTOM_ADAPTER,
        "litellm_max_workers": LITELLM_MAX_WORKERS,
        # Few-shot metadata
        "use_few_shot": USE_FEWSHOT,
        "few_shot_file_used": few_shot_file_used,
        "few_shot_rows": few_shot_rows,
    }
    
    # Create logger instance and save configuration
    logger = ExperimentLogger(run_dir)
    logger.save_experiment_config(result, hyperparams, seed_prompt)


if __name__ == "__main__":
    main()
