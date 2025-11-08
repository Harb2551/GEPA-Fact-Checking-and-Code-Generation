# Evaluation Module

A modular, SOLID-compliant evaluation system with checkpoint support for test set evaluation.

## Structure

```
evaluation/
├── __init__.py              # Package exports
├── data_formatter.py        # Data format conversion
├── checkpoint_manager.py    # Checkpoint save/load
├── prompt_loader.py         # Prompt loading utilities
├── evaluator.py            # Evaluation engine with checkpoints
└── report_generator.py     # Report generation and formatting
```

## Components

### DataFormatter
- **Responsibility**: Convert HoVer dataset format to GEPA format
- **Key Method**: `hover_to_gepa_format(example)`

### CheckpointManager
- **Responsibility**: Save and load evaluation checkpoints
- **Key Methods**:
  - `save_checkpoint(prompt_type, results, progress)`
  - `load_checkpoint(prompt_type)`
  - `clear_checkpoint(prompt_type)`

### PromptLoader
- **Responsibility**: Load prompts from files or extract from config
- **Key Methods**:
  - `load_from_file(file_path)`
  - `extract_seed_from_config(config_file, output_file)`

### PromptEvaluator
- **Responsibility**: Evaluate prompts with automatic checkpointing
- **Key Method**: `evaluate_with_checkpoints(prompt, test_examples, prompt_type, batch_size)`
- **Features**:
  - Auto-saves progress every N examples (batch_size)
  - Auto-resumes from checkpoint on interruption
  - Clears checkpoint on successful completion

### ReportGenerator
- **Responsibility**: Generate and save evaluation reports
- **Key Methods**:
  - `generate_report(seed_results, optimized_results, test_size)`
  - `save_report(report, seed_results, optimized_results)`
  - `print_summary(report)`

## Usage

```python
from evaluation import (
    DataFormatter,
    CheckpointManager,
    PromptLoader,
    PromptEvaluator,
    ReportGenerator
)

# Initialize components
data_formatter = DataFormatter()
checkpoint_manager = CheckpointManager("./checkpoints")
prompt_loader = PromptLoader()
report_generator = ReportGenerator("./results")

# Use in your evaluation pipeline
```

## Checkpoint System

### How It Works
1. Evaluation runs in batches (default: 50 examples)
2. After each batch, progress is saved to checkpoint file
3. If interrupted, rerun the same command to resume
4. Checkpoint is automatically deleted on successful completion

### Checkpoint Location
`{run_dir}/eval_checkpoints/eval_checkpoint_{prompt_type}.json`

### Checkpoint Contents
- `prompt_type`: "seed" or "optimized"
- `results`: List of evaluation results so far
- `progress`: Number of examples completed
- `timestamp`: When checkpoint was created

## SOLID Principles

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to extend without modifying existing code
- **Liskov Substitution**: Components can be swapped with compatible implementations
- **Interface Segregation**: Clean, focused interfaces
- **Dependency Inversion**: High-level orchestrator depends on abstractions

## Error Handling

All components handle interruptions gracefully:
- `KeyboardInterrupt`: Saves progress and provides resume instructions
- `Exception`: Saves progress and allows retry
- File errors: Clear error messages with troubleshooting hints
