# Examples

Example scripts for the nonescape Python library.

## Scripts

### `basic_inference.py`
Basic image classification example.

```bash
python basic_inference.py model.safetensors image.jpg
```

### `aria_test.py`
ARIA dataset evaluation script with multi-GPU support.

```bash
# Single GPU
python aria_test.py model.safetensors --data-path ./aria_dataset

# Multi-GPU
torchrun --nproc_per_node=gpu aria_test.py model.safetensors --data-path ./aria_dataset
```

## Model Setup

Download models before running examples:

```bash
# Full model
wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors

# Small model  
wget https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-small-v0.safetensors
```
