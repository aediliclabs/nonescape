# Nonescape JavaScript

AI-generated image detection for browsers and Node.js using [nonescape](https://www.nonescape.com) 
deep classifiers.

## Installation

```bash
npm install @aedilic/nonescape
```

## Quick Start

### Local Classification

```javascript
import { LocalClassifier } from '@aedilic/nonescape';

const classifier = new LocalClassifier({
  onProgress: (progress) => console.log(`Loading: ${Math.round(progress.current / progress.total * 100)}%`)
});

await classifier.initialize();

// Single image
const result = await classifier.predict(imageFile);
console.log(`AI Generated: ${result.isSynthetic}, Confidence: ${result.confidence}`);

// Multiple images
const results = await classifier.predict([image1, image2, image3]);

classifier.dispose();
```

### Remote API

```javascript
import { RemoteClassifier } from '@aedilic/nonescape';

const classifier = new RemoteClassifier({ apiToken: 'your-token' });
const result = await classifier.predict(imageFile);
```

## API

### LocalClassifier

**Options**
- `modelPath?: string` - Custom model URL
- `onProgress?: (progress) => void` - Loading progress callback
- `threshold?: number` - Classification threshold (default: 0.5)

**Methods**
- `initialize()` - Load model
- `predict(image|images[])` - Classify image(s)
- `dispose()` - Clean up resources

### RemoteClassifier

**Options**
- `apiToken?: string` - API bearer token
- `baseUrl?: string` - Custom API URL
- `threshold?: number` - Classification threshold (default: 0.5)

**Methods**
- `predict(image|images[])` - Classify image(s)

**Input types:** `File | HTMLImageElement | string`

**Result:** `{ confidence: number, isSynthetic: boolean }`

