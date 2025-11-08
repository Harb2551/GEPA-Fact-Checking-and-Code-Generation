"""
Prompt Loader Module
====================
Handles loading prompts from files and configuration
"""

import json
from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """Handles loading prompts from files and config"""
    
    @staticmethod
    def load_from_file(file_path: Path) -> Dict[str, str]:
        """
        Load a prompt from a text file
        
        Args:
            file_path: Path to prompt text file
        
        Returns:
            Dict with 'system_prompt' key
        """
        with open(file_path, 'r') as f:
            prompt_text = f.read().strip()
        return {"system_prompt": prompt_text}
    
    @staticmethod
    def extract_seed_from_config(config_file: Path, output_file: Path) -> Optional[Dict[str, str]]:
        """
        Extract seed prompt from experiment config and save to file
        
        Args:
            config_file: Path to experiment_config.json
            output_file: Path where seed prompt should be saved
        
        Returns:
            Dict with 'system_prompt' key or None if extraction fails
        """
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            seed_prompt_text = config.get('seed_prompt', {}).get('system_prompt', '')
            if not seed_prompt_text:
                return None
            
            # Save for future use
            with open(output_file, 'w') as sf:
                sf.write(seed_prompt_text)
            
            print(f"  ✓ Extracted seed prompt from config")
            return {"system_prompt": seed_prompt_text}
        except Exception as e:
            print(f"  ⚠ Error extracting seed prompt: {e}")
            return None
