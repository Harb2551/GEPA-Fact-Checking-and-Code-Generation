"""
Report Generator Module
=======================
Generates and saves evaluation reports
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class ReportGenerator:
    """Generates and saves evaluation reports"""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory where reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        seed_results: Dict,
        optimized_results: Dict,
        test_size: int
    ) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            seed_results: Results from seed prompt evaluation
            optimized_results: Results from optimized prompt evaluation
            test_size: Number of test examples evaluated
        
        Returns:
            Complete report dictionary
        """
        
        report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "run_directory": str(self.output_dir),
                "test_set_size": test_size
            },
            "seed_prompt_performance": {
                "accuracy": seed_results["accuracy"],
                "correct": seed_results["correct"],
                "total": seed_results["total"],
                "predicted_distribution": seed_results["predicted_distribution"],
                "actual_distribution": seed_results["actual_distribution"]
            },
            "optimized_prompt_performance": {
                "accuracy": optimized_results["accuracy"],
                "correct": optimized_results["correct"],
                "total": optimized_results["total"],
                "predicted_distribution": optimized_results["predicted_distribution"],
                "actual_distribution": optimized_results["actual_distribution"]
            },
            "comparison": {
                "accuracy_improvement": optimized_results["accuracy"] - seed_results["accuracy"],
                "accuracy_improvement_percentage": (
                    (optimized_results["accuracy"] - seed_results["accuracy"]) / seed_results["accuracy"] * 100
                    if seed_results["accuracy"] > 0 else 0.0
                ),
                "additional_correct_predictions": optimized_results["correct"] - seed_results["correct"],
                "relative_improvement": (
                    f"{optimized_results['accuracy']:.2%} vs {seed_results['accuracy']:.2%}"
                )
            },
            "summary": {
                "seed_accuracy": f"{seed_results['accuracy']:.2%}",
                "optimized_accuracy": f"{optimized_results['accuracy']:.2%}",
                "improvement": f"{(optimized_results['accuracy'] - seed_results['accuracy']):.2%}",
                "verdict": self._get_verdict(seed_results["accuracy"], optimized_results["accuracy"])
            }
        }
        
        return report
    
    def save_report(self, report: Dict, seed_results: Dict, optimized_results: Dict):
        """
        Save report and detailed results to files with timestamps
        
        Args:
            report: Report dictionary
            seed_results: Seed prompt results
            optimized_results: Optimized prompt results
        
        Returns:
            Path to saved report file
        """
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main report
        report_file = self.output_dir / f"test_evaluation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Evaluation report saved to: {report_file}")
        
        # Save detailed results
        detailed_file = self.output_dir / f"test_detailed_results_{timestamp}.json"
        detailed_data = {
            "seed_prompt_results": seed_results["detailed_results"],
            "optimized_prompt_results": optimized_results["detailed_results"]
        }
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        print(f"✓ Detailed results saved to: {detailed_file}")
        
        return report_file
    
    @staticmethod
    def _get_verdict(seed_acc: float, optimized_acc: float) -> str:
        """
        Determine verdict based on accuracy comparison
        
        Args:
            seed_acc: Seed prompt accuracy
            optimized_acc: Optimized prompt accuracy
        
        Returns:
            Verdict string: IMPROVED, NO_CHANGE, or DEGRADED
        """
        if optimized_acc > seed_acc:
            return "IMPROVED"
        elif optimized_acc == seed_acc:
            return "NO_CHANGE"
        else:
            return "DEGRADED"
    
    @staticmethod
    def print_summary(report: Dict):
        """
        Print formatted summary of evaluation
        
        Args:
            report: Report dictionary to display
        """
        print("\n" + "="*70)
        print("TEST SET EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nTest Set Size: {report['evaluation_info']['test_set_size']} examples")
        
        print("\n" + "-"*70)
        print("SEED PROMPT PERFORMANCE")
        print("-"*70)
        print(f"Accuracy: {report['seed_prompt_performance']['accuracy']:.2%}")
        print(f"Correct: {report['seed_prompt_performance']['correct']} / {report['seed_prompt_performance']['total']}")
        
        print("\n" + "-"*70)
        print("OPTIMIZED PROMPT PERFORMANCE")
        print("-"*70)
        print(f"Accuracy: {report['optimized_prompt_performance']['accuracy']:.2%}")
        print(f"Correct: {report['optimized_prompt_performance']['correct']} / {report['optimized_prompt_performance']['total']}")
        
        print("\n" + "-"*70)
        print("COMPARISON")
        print("-"*70)
        print(f"Accuracy Improvement: {report['comparison']['accuracy_improvement']:+.2%}")
        print(f"Improvement Percentage: {report['comparison']['accuracy_improvement_percentage']:+.2f}%")
        print(f"Additional Correct: {report['comparison']['additional_correct_predictions']:+d}")
        print(f"Verdict: {report['summary']['verdict']}")
        
        print("\n" + "="*70)
