"""
Test Set Evaluation Script
===========================

Evaluates both seed and optimized prompts on the HoVer test set and saves comparison results.
Implements SOLID principles and checkpoint support for interruption recovery.

Usage:
    python evaluate_test_set.py
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from pathlib import Path

from hover_adapter import HoVerAdapter
from evaluation import (
    DataFormatter,
    CheckpointManager,
    PromptLoader,
    PromptEvaluator,
    ReportGenerator
)

# Load environment variables
# Portable .env loading (search upward for .env)
def _load_env():
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            break
    else:
        load_dotenv()

_load_env()


# =====================
# USER CONFIGURABLE VARIABLES
# =====================
RUN_DIR = "./results/gepa_hover_results_20251109_003303"
# Optional: path to a separate run directory that contains a few-shot-optimized
# optimized_prompt.txt. If set, that optimized prompt will also be evaluated on
# the same test set and results saved alongside the main run's reports.
RUN_DIR_FEWSHOT = "./results/gepa_hover_results_20251109_004549"
TEST_SIZE = 500
TASK_LM = "gpt-4.1-mini"
MAX_WORKERS = 2
BATCH_SIZE = 50
# =====================


# ============================================================================
# Main Orchestrator (Dependency Inversion Principle)
# ============================================================================

class TestEvaluationOrchestrator:
    """Main orchestrator for test set evaluation"""
    
    def __init__(
        self,
        run_dir: str,
        test_size: int,
        task_lm: str,
        max_workers: int,
        batch_size: int = 50,
        fewshot_run_dir: Optional[str] = None,
    ):
        """
        Initialize orchestrator
        
        Args:
            run_dir: Directory containing prompts and where results will be saved
            test_size: Number of test examples to evaluate
            task_lm: Model name for evaluation
            max_workers: Max parallel workers for litellm
            batch_size: Examples per checkpoint
        """
        self.run_dir = Path(run_dir)
        self.fewshot_run_dir = Path(fewshot_run_dir) if fewshot_run_dir else None
        self.test_size = test_size
        self.task_lm = task_lm
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialize components (Dependency Injection)
        self.data_formatter = DataFormatter()
        self.checkpoint_manager = CheckpointManager(self.run_dir / "eval_checkpoints")
        self.prompt_loader = PromptLoader()
        self.report_generator = ReportGenerator(self.run_dir)
    
    def run(self):
        """Execute the full evaluation pipeline"""
        # keep a reference to few-shot results (if any) so we can print them later
        fewshot_results = None
        
        # Step 1: Load test data
        test_examples = self._load_test_data()
        
        # Step 2: Load prompts
        seed_prompt, optimized_prompt = self._load_prompts()
        
        # Step 3: Initialize adapter and evaluator
        adapter = HoVerAdapter(model=self.task_lm, max_litellm_workers=self.max_workers)
        evaluator = PromptEvaluator(adapter, self.checkpoint_manager)
        
        # Step 4: Evaluate seed prompt
        print("\n" + "="*70)
        print("EVALUATING SEED PROMPT")
        print("="*70)
        seed_results = evaluator.evaluate_with_checkpoints(
            seed_prompt, test_examples, "seed", self.batch_size
        )
        print(f"✓ Seed Prompt Accuracy: {seed_results['accuracy']:.2%}")
        
        # Step 5: Evaluate optimized prompt
        print("\n" + "="*70)
        print("EVALUATING OPTIMIZED PROMPT")
        print("="*70)
        optimized_results = evaluator.evaluate_with_checkpoints(
            optimized_prompt, test_examples, "optimized", self.batch_size
        )
        print(f"✓ Optimized Prompt Accuracy: {optimized_results['accuracy']:.2%}")

        # Optional: if a separate few-shot run dir is provided, evaluate its
        # optimized prompt on the same test set for direct comparison.
        if self.fewshot_run_dir is not None and self.fewshot_run_dir.exists():
            try:
                print("\n" + "="*70)
                print("EVALUATING FEW-SHOT OPTIMIZED PROMPT (from different run_dir)")
                print("="*70)
                fewshot_opt_file = self.fewshot_run_dir / "optimized_prompt.txt"
                if not fewshot_opt_file.exists():
                    raise FileNotFoundError(f"Optimized prompt not found in few-shot run dir: {fewshot_opt_file}")

                from evaluation.prompt_loader import PromptLoader as _PL
                fewshot_prompt = _PL.load_from_file(fewshot_opt_file)

                fewshot_results = evaluator.evaluate_with_checkpoints(
                    fewshot_prompt, test_examples, "fewshot_optimized", self.batch_size
                )
                print(f"✓ Few-shot Optimized Prompt Accuracy: {fewshot_results['accuracy']:.2%}")

                # Save the few-shot evaluation results to a JSON file inside the
                # few-shot run directory (so results live with that experiment).
                import json, time
                ts = time.strftime("%Y%m%d_%H%M%S")
                target_dir = self.fewshot_run_dir or self.run_dir
                out_file = target_dir / f"fewshot_optimized_results_{ts}.json"
                with open(out_file, 'w', encoding='utf-8') as of:
                    json.dump(fewshot_results, of, indent=2)
                print(f"✓ Few-shot evaluation results saved to: {out_file} (inside few-shot run dir)")

            except Exception as e:
                print(f"Warning: failed to evaluate few-shot optimized prompt: {e}")
        
        # Step 6: Generate and save report
        print("\n" + "="*70)
        print("GENERATING EVALUATION REPORT")
        print("="*70)
        report = self.report_generator.generate_report(
            seed_results, optimized_results, len(test_examples)
        )
        self.report_generator.save_report(report, seed_results, optimized_results)
        
        # Step 7: Print summary
        self.report_generator.print_summary(report)

        # Extra: print few-shot summary if we evaluated a few-shot optimized prompt
        if fewshot_results is not None:
            try:
                print("\n" + "="*70)
                print("FEW-SHOT OPTIMIZED PROMPT SUMMARY")
                print("="*70)
                acc = fewshot_results.get("accuracy")
                correct = fewshot_results.get("correct")
                total = fewshot_results.get("total")
                print(f"Accuracy: {acc:.2%}" if acc is not None else f"Accuracy: {acc}")
                print(f"Correct: {correct} / {total}")

                # Show improvement over baseline if baseline is available
                try:
                    baseline_acc = seed_results.get("accuracy")
                    if baseline_acc is not None and acc is not None:
                        delta = acc - baseline_acc
                        pct = (delta / baseline_acc * 100.0) if baseline_acc else None
                        print(f"Improvement vs baseline: {delta:+.4f} ({pct:+.2f}% )" if pct is not None else f"Improvement vs baseline: {delta:+.4f}")
                except Exception:
                    pass

                # Print a short excerpt of the few-shot optimized prompt (if available)
                try:
                    if self.fewshot_run_dir is not None:
                        opt_prompt_file = self.fewshot_run_dir / "optimized_prompt.txt"
                        if opt_prompt_file.exists():
                            txt = opt_prompt_file.read_text(encoding='utf-8').strip()
                            excerpt = txt[:400].replace('\n', ' ')
                            print("\nFew-shot optimized prompt excerpt:")
                            print(excerpt + ("..." if len(txt) > len(excerpt) else ""))
                except Exception:
                    pass

                # Optionally print per-example brief results (predicted labels)
                try:
                    dr = fewshot_results.get('detailed_results')
                    if dr:
                        preds = [d.get('predicted_label') or d.get('full_response') for d in dr]
                        print("\nPredicted labels (few-shot):", preds)
                except Exception:
                    pass

            except Exception as e:
                print(f"Warning: failed to print few-shot summary: {e}")
        
        # Step 8: Final comparison JSON (baseline vs optimized, and vs few-shot if present)
        try:
            import time
            ts = time.strftime("%Y%m%d_%H%M%S")
            comparison = {
                "timestamp": ts,
                "test_size": len(test_examples),
                "baseline": {
                    "accuracy": seed_results.get("accuracy"),
                    "correct": seed_results.get("correct"),
                    "total": seed_results.get("total"),
                },
                "optimized": {
                    "accuracy": optimized_results.get("accuracy"),
                    "correct": optimized_results.get("correct"),
                    "total": optimized_results.get("total"),
                },
            }

            # Compute deltas against baseline
            try:
                b_acc = float(seed_results.get("accuracy", 0.0))
                o_acc = float(optimized_results.get("accuracy", 0.0))
                comparison["optimized"]["delta_absolute"] = o_acc - b_acc
                comparison["optimized"]["delta_percent"] = ((o_acc - b_acc) / b_acc * 100.0) if b_acc else None
                comparison["optimized"]["additional_correct"] = int(optimized_results.get("correct", 0) - seed_results.get("correct", 0))
            except Exception:
                # best-effort; if keys missing, skip numeric deltas
                comparison["optimized"]["delta_absolute"] = None
                comparison["optimized"]["delta_percent"] = None
                comparison["optimized"]["additional_correct"] = None

            # If few-shot evaluation happened, include it and compute deltas
            if 'fewshot_results' in locals() and fewshot_results is not None:
                f_res = fewshot_results
                f_acc = float(f_res.get("accuracy", 0.0))
                comparison["fewshot_optimized"] = {
                    "accuracy": f_acc,
                    "correct": f_res.get("correct"),
                    "total": f_res.get("total"),
                    "delta_absolute": (f_acc - b_acc),
                    "delta_percent": ((f_acc - b_acc) / b_acc * 100.0) if b_acc else None,
                    "additional_correct": int(f_res.get("correct", 0) - seed_results.get("correct", 0)),
                }

            # Save comparison JSON to main run_dir
            import json
            comp_file = self.run_dir / f"comparison_results_{ts}.json"
            with open(comp_file, 'w', encoding='utf-8') as cf:
                json.dump(comparison, cf, indent=2)
            print(f"✓ Final comparison saved to: {comp_file}")

            # Also save a copy into the few-shot run dir if present
            if self.fewshot_run_dir is not None:
                try:
                    copy_path = self.fewshot_run_dir / f"comparison_results_{ts}.json"
                    with open(copy_path, 'w', encoding='utf-8') as cf2:
                        json.dump(comparison, cf2, indent=2)
                    print(f"✓ Final comparison saved to few-shot run dir: {copy_path}")
                except Exception as e:
                    print(f"Warning: failed to write comparison into few-shot dir: {e}")

        except Exception as e:
            print(f"Warning: failed to write final comparison JSON: {e}")
    
    def _load_test_data(self) -> List[Dict]:
        """Load and convert test data"""
        print("Loading HoVer dataset...")
        ds = load_dataset("Dzeniks/hover")
        
        # Use validation split as test set (or test split if available)
        if 'test' in ds:
            test_data = ds['test']
        elif 'validation' in ds:
            test_data = ds['validation']
        else:
            full_data = ds['train']
            test_data = full_data.select(range(len(full_data) - self.test_size, len(full_data)))
        
        print(f"Available test examples: {len(test_data)}")
        
        # Shuffle the test data for randomized evaluation
        test_data_list = list(test_data)
        random.shuffle(test_data_list)
        print("✓ Test data shuffled for randomized evaluation")
        
        # Convert to GEPA format
        print("Converting to GEPA format...")
        test_examples = [self.data_formatter.hover_to_gepa_format(ex) for ex in test_data_list]
        test_examples = test_examples[:self.test_size]
        print(f"✓ Using {len(test_examples)} test examples")
        
        return test_examples
    
    def _load_prompts(self) -> Tuple[Dict, Dict]:
        """Load seed and optimized prompts with UTF-8 encoding"""
        print(f"\nLoading prompts from {self.run_dir}...")
        
        seed_prompt_file = self.run_dir / "seed_prompt.txt"
        optimized_prompt_file = self.run_dir / "optimized_prompt.txt"
        
        # Load or extract seed prompt
        if not seed_prompt_file.exists():
            print("Seed prompt not found, extracting from config...")
            config_file = self.run_dir / "experiment_config.json"
            seed_prompt = self._extract_seed_from_config(config_file, seed_prompt_file)
            if not seed_prompt:
                raise FileNotFoundError(f"Could not find or extract seed prompt")
        else:
            seed_prompt = self._load_prompt_from_file(seed_prompt_file)
        
        # Load optimized prompt
        if not optimized_prompt_file.exists():
            raise FileNotFoundError(f"Optimized prompt not found at {optimized_prompt_file}")
        
        optimized_prompt = self._load_prompt_from_file(optimized_prompt_file)
        
        print("✓ Prompts loaded successfully")
        return seed_prompt, optimized_prompt
    
    def _load_prompt_from_file(self, file_path: Path) -> Dict[str, str]:
        """Load a prompt from a text file with UTF-8 encoding"""
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        return {"system_prompt": prompt_text}
    
    def _extract_seed_from_config(self, config_file: Path, output_file: Path) -> Dict[str, str]:
        """Extract seed prompt from experiment config and save to file with UTF-8 encoding"""
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            seed_prompt_text = config.get('seed_prompt', {}).get('system_prompt', '')
            if not seed_prompt_text:
                return None
            
            # Save for future use
            with open(output_file, 'w', encoding='utf-8') as sf:
                sf.write(seed_prompt_text)
            
            print(f"  ✓ Extracted seed prompt from config")
            return {"system_prompt": seed_prompt_text}
        except Exception as e:
            print(f"  ⚠ Error extracting seed prompt: {e}")
            return None


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Main entry point"""
    # Use variables above instead of CLI args
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        return
    try:
        orchestrator = TestEvaluationOrchestrator(
            run_dir=RUN_DIR,
            test_size=TEST_SIZE,
            task_lm=TASK_LM,
            max_workers=MAX_WORKERS,
            batch_size=BATCH_SIZE,
            fewshot_run_dir=RUN_DIR_FEWSHOT or None,
        )
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user. Progress has been saved.")
        print("Run the same command again to resume from checkpoint.")
    except Exception as e:
        print(f"\n\n✗ Error during evaluation: {e}")
        print("Progress has been saved. Run again to resume.")


if __name__ == "__main__":
    main()
