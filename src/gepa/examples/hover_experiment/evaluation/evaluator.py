"""
Prompt Evaluator Module
========================
Evaluates prompts with checkpoint support
"""

from typing import Dict, List, Any
from hover_adapter import HoVerAdapter

from .checkpoint_manager import CheckpointManager


class PromptEvaluator:
    """Evaluates prompts with checkpoint support"""
    
    def __init__(self, adapter: HoVerAdapter, checkpoint_manager: CheckpointManager):
        """
        Initialize evaluator
        
        Args:
            adapter: HoVerAdapter instance for running evaluations
            checkpoint_manager: CheckpointManager for saving/loading progress
        """
        self.adapter = adapter
        self.checkpoint_manager = checkpoint_manager
    
    def evaluate_with_checkpoints(
        self,
        prompt: Dict,
        test_examples: List[Dict],
        prompt_type: str,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt with checkpoint support
        
        Args:
            prompt: Prompt dictionary with 'system_prompt' key
            test_examples: List of test examples in GEPA format
            prompt_type: 'seed' or 'optimized'
            batch_size: Number of examples to process before checkpointing
        
        Returns:
            Dict with evaluation results including accuracy and detailed results
        """
        print(f"\nEvaluating {prompt_type} prompt on {len(test_examples)} examples...")
        
        # Try to resume from checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(prompt_type)
        
        if checkpoint:
            results = checkpoint['results']
            start_idx = checkpoint['progress']
            print(f"  ↻ Resuming from example {start_idx}")
        else:
            results = []
            start_idx = 0
            print(f"  ▸ Starting fresh evaluation")
        
        # Process in batches with checkpointing
        try:
            for batch_start in range(start_idx, len(test_examples), batch_size):
                batch_end = min(batch_start + batch_size, len(test_examples))
                batch = test_examples[batch_start:batch_end]
                
                print(f"  Processing examples {batch_start+1}-{batch_end}...")
                
                # Evaluate batch
                batch_results = self.adapter.evaluate(
                    batch=batch,
                    candidate=prompt
                )
                results.extend(batch_results.outputs)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(prompt_type, results, len(results))
            
            # Calculate final metrics
            metrics = self._calculate_metrics(results, test_examples)
            
            # Clear checkpoint on successful completion
            self.checkpoint_manager.clear_checkpoint(prompt_type)
            
            return {
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "predicted_distribution": metrics["predicted_distribution"],
                "actual_distribution": metrics["actual_distribution"],
                "detailed_results": results
            }
        
        except KeyboardInterrupt:
            print(f"\n  ⚠ Evaluation interrupted! Progress saved to checkpoint.")
            print(f"  Run again to resume from example {len(results)}")
            raise
        except Exception as e:
            print(f"\n  ✗ Error during evaluation: {e}")
            print(f"  Progress saved. Run again to resume.")
            raise
    
    def _calculate_metrics(self, results: List[Dict], test_examples: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            results: List of evaluation results
            test_examples: Original test examples
        
        Returns:
            Dict with calculated metrics
        """
        # Adapt to HoVerAdapter output structure:
        # Each result item has keys: 'full_response', 'predicted_label'
        # Original test_examples have key 'answer'. Compute correctness here.
        total = len(results)
        correct = 0
        predicted_supported = 0
        predicted_not_supported = 0

        for res, ex in zip(results, test_examples, strict=False):
            pred = res.get('predicted_label')
            gold = ex.get('answer')
            if pred == 'SUPPORTED':
                predicted_supported += 1
            elif pred == 'NOT_SUPPORTED':
                predicted_not_supported += 1
            # correctness
            if isinstance(pred, str) and isinstance(gold, str) and pred.strip().upper() == gold.strip().upper():
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        actual_supported = sum(1 for ex in test_examples if ex.get('answer') == 'SUPPORTED')
        actual_not_supported = sum(1 for ex in test_examples if ex.get('answer') == 'NOT_SUPPORTED')

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predicted_distribution': {
                'SUPPORTED': predicted_supported,
                'NOT_SUPPORTED': predicted_not_supported,
            },
            'actual_distribution': {
                'SUPPORTED': actual_supported,
                'NOT_SUPPORTED': actual_not_supported,
            },
        }
