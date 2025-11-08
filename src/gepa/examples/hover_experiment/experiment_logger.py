"""
Experiment Logger for saving hyperparameters and results
"""

import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Handles saving experiment configuration and results to JSON"""
    
    def __init__(self, run_dir: str):
        """
        Initialize the experiment logger
        
        Args:
            run_dir: Directory where results will be saved
        """
        self.run_dir = run_dir
        
    def save_experiment_config(
        self,
        result,
        hyperparameters: dict,
        seed_prompt: dict
    ):
        """
        Save experiment configuration and results to JSON
        
        Args:
            result: GEPAResult object from gepa.optimize()
            hyperparameters: Dict of hyperparameters used in the experiment
            seed_prompt: Original seed prompt dict
        """
        # Calculate Pareto front score
        pareto_score = self._calculate_pareto_score(result)
        
        # Get seed score (first candidate)
        seed_score = result.val_aggregate_scores[0] if len(result.val_aggregate_scores) > 0 else 0.0
        
        # Get best score
        best_score = result.val_aggregate_scores[result.best_idx]
        
        # Build configuration dict
        experiment_config = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "run_directory": self.run_dir,
            },
            "hyperparameters": hyperparameters,
            "results": {
                "best_prompt_index": result.best_idx,
                "best_prompt_score": best_score,
                "seed_prompt_score": seed_score,
                "improvement_over_seed": best_score - seed_score,
                "improvement_percentage": ((best_score - seed_score) / seed_score * 100) if seed_score > 0 else 0.0,
                "pareto_front_score": pareto_score,
                "pareto_improvement": pareto_score - best_score,
                "total_iterations": result.num_candidates,
                "total_metric_calls": result.total_metric_calls,
                "num_candidates_evaluated": result.num_candidates,
                "num_full_val_evals": result.num_full_val_evals,
            },
            "all_candidate_scores": {
                f"candidate_{i}": score 
                for i, score in enumerate(result.val_aggregate_scores)
            }
        }
        
        # Add seed prompt to config for downstream evaluation scripts
        experiment_config["seed_prompt"] = seed_prompt

        # Save to JSON
        config_file = Path(self.run_dir) / "experiment_config.json"
        with open(config_file, "w") as f:
            json.dump(experiment_config, f, indent=2)
        
        print(f"Experiment configuration saved to: {config_file}")
        
        return config_file
    
    def _calculate_pareto_score(self, result) -> float:
        """Calculate Pareto front score from result"""
        if not result.per_val_instance_best_candidates:
            return 0.0
        
        # Count how many validation examples have at least one solver
        examples_with_solver = sum(
            1 for candidates in result.per_val_instance_best_candidates.values() 
            if len(candidates) > 0
        )
        
        total_examples = len(result.per_val_instance_best_candidates)
        
        return examples_with_solver / total_examples if total_examples > 0 else 0.0
    
    def save_optimized_prompt(self, result):
        """
        Save the optimized prompt to a text file
        
        Args:
            result: GEPAResult object from gepa.optimize()
        """
        output_file = Path(self.run_dir) / "optimized_prompt.txt"
        with open(output_file, "w") as f:
            f.write(result.best_candidate['system_prompt'])
        
        print(f"Optimized prompt saved to: {output_file}")
        
        return output_file
    
    def print_summary(self, result):
        """
        Print a summary of the optimization results
        
        Args:
            result: GEPAResult object from gepa.optimize()
        """
        best_score = result.val_aggregate_scores[result.best_idx]
        seed_score = result.val_aggregate_scores[0] if len(result.val_aggregate_scores) > 0 else 0.0
        pareto_score = self._calculate_pareto_score(result)
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Best Prompt Index: {result.best_idx}")
        print(f"Best Score: {best_score:.2%}")
        print(f"Seed Score: {seed_score:.2%}")
        print(f"Improvement: {best_score - seed_score:+.2%}")
        print(f"Pareto Front Score: {pareto_score:.2%}")
        print(f"Total Metric Calls: {result.total_metric_calls}")
        print(f"Candidates Evaluated: {result.num_candidates}")
        print("="*60)
