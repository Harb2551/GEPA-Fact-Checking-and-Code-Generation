# HoVer Dataset Optimization with GEPA

This example demonstrates using GEPA to optimize prompts for fact verification on the HoVer dataset.

## Files

- `train_hover.py`: Main training script
- `hover_adapter.py`: Custom GEPA adapter for HoVer with enhanced feedback
- `README.md`: This file

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage (Default Adapter)

Run with the default GEPA adapter (from repository root or this directory):
```bash
cd src/gepa/examples/hover_experiment
python train_hover.py
```

### With Custom Adapter (Recommended)

Run with the custom HoVer adapter for better feedback:
```bash
python train_hover.py --use-custom-adapter
```

### Advanced Options

```bash
# Full dataset with custom adapter
python train_hover.py --use-custom-adapter --train-size 1000 --val-size 200

# Increase optimization budget
python train_hover.py --use-custom-adapter --max-metric-calls 500

# Use different models
python train_hover.py --use-custom-adapter \
    --task-lm "openai/gpt-4o" \
    --reflection-lm "openai/gpt-4o"

# Quick test run
python train_hover.py --use-custom-adapter --train-size 20 --val-size 5 --max-metric-calls 50
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-custom-adapter` | False | Use custom HoVerAdapter instead of DefaultAdapter |
| `--train-size` | 100 | Number of training examples |
| `--val-size` | 20 | Number of validation examples |
| `--max-metric-calls` | 200 | Maximum number of evaluations |
| `--task-lm` | `openai/gpt-4o-mini` | Model to optimize |
| `--reflection-lm` | `openai/gpt-4o` | Model for reflection |

## Custom Adapter Benefits

The `HoVerAdapter` provides:

1. **Better Label Extraction**: Handles variations like "SUPPORTS", "REFUTES", "NOT SUPPORTED"
2. **Detailed Feedback**: Provides specific analysis of why predictions failed
3. **Enhanced Reflection**: Gives the reflection LM more context about errors
4. **Custom Scoring**: Exact match on predicted vs ground truth labels

## Results

Results will be saved to `./gepa_hover_results/`:
- `gepa_state.bin`: Optimization state (can resume if interrupted)
- `optimized_prompt.txt`: The best prompt found
- `generated_best_outputs_valset/`: Best outputs for each validation example

## Dataset

The HoVer dataset contains claims that need to be verified as SUPPORTED or NOT_SUPPORTED based on provided context from Wikipedia.

**Example:**
- **Claim**: "Skagen Painter, who painted the 1893 painting Roses, favored naturalism."
- **Context**: Multiple sentences from Wikipedia articles
- **Label**: SUPPORTED or NOT_SUPPORTED

## How GEPA Works

1. Evaluates seed prompt on validation set
2. Each iteration:
   - Selects a candidate from Pareto front
   - Tests on training minibatch (3 examples)
   - Uses LLM reflection to propose improved prompt
   - Re-evaluates new prompt on same minibatch
   - If improved → evaluate on full validation set
3. Returns best prompt based on validation scores
