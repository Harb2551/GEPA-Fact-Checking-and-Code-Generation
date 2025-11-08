"""
Checkpoint Manager Module
=========================
Manages saving and loading evaluation checkpoints for interruption recovery
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class CheckpointManager:
    """Manages saving and loading evaluation checkpoints"""
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory where checkpoints will be saved
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, prompt_type: str) -> Path:
        """
        Get checkpoint file path for a specific prompt type
        
        Args:
            prompt_type: Either 'seed' or 'optimized'
        
        Returns:
            Path to checkpoint file
        """
        return self.checkpoint_dir / f"eval_checkpoint_{prompt_type}.json"
    
    def save_checkpoint(self, prompt_type: str, results: List[Dict], progress: int):
        """
        Save evaluation checkpoint
        
        Args:
            prompt_type: Either 'seed' or 'optimized'
            results: List of evaluation results so far
            progress: Number of examples completed
        """
        checkpoint_data = {
            "prompt_type": prompt_type,
            "results": results,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = self.get_checkpoint_path(prompt_type)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"  ✓ Checkpoint saved: {progress} examples completed")
    
    def load_checkpoint(self, prompt_type: str) -> Optional[Dict]:
        """
        Load evaluation checkpoint if exists
        
        Args:
            prompt_type: Either 'seed' or 'optimized'
        
        Returns:
            Checkpoint data dict or None if not found
        """
        checkpoint_path = self.get_checkpoint_path(prompt_type)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"  ✓ Found checkpoint: {checkpoint_data['progress']} examples completed")
            return checkpoint_data
        except Exception as e:
            print(f"  ⚠ Error loading checkpoint: {e}")
            return None
    
    def clear_checkpoint(self, prompt_type: str):
        """
        Delete checkpoint after successful completion
        
        Args:
            prompt_type: Either 'seed' or 'optimized'
        """
        checkpoint_path = self.get_checkpoint_path(prompt_type)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  ✓ Checkpoint cleared")
