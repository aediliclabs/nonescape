# Nonescape Python

AI-generated image detection using nonescape deep classifiers.

## Installation

```bash
pip install -e .
```

## Usage

```python
from nonescape import NonescapeClassifier

# Load model
classifier = NonescapeClassifier("nonescape-v0.safetensors")

# Single image
result = classifier.predict("image.jpg")
print(f"AI Generated: {result.is_synthetic}, Confidence: {result.confidence}")

# Multiple images
results = classifier.predict(["img1.jpg", "img2.jpg"])
```

## Models

### Full Model (`NonescapeClassifier`)
- Vision Transformer + EfficientNet
- Higher accuracy
- Download: [nonescape-v0.safetensors](https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-v0.safetensors)

### Small Model (`NonescapeClassifierSmall`) 
- EfficientNet only
- Faster inference
- Download: [nonescape-small-v0.safetensors](https://nonescape.sfo2.cdn.digitaloceanspaces.com/nonescape-small-v0.safetensors)

## Examples

See [`examples/`](examples/README.md) for detailed usage examples.

## Browser Usage

For client-side detection, use the [JavaScript library](../javascript/README.md).
