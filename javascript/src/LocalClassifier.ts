// Copyright 2025 Aedilic Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import * as ort from "onnxruntime-web";
import Classifier, {
  ClassificationResult,
  InputFileType,
  isBatchInput,
  loadImage,
  reweighThreshold,
} from "./Classifier";

/** Progress information for model loading */
export interface LoadingProgress {
  /** Bytes loaded so far */
  current: number;
  /** Total bytes to load */
  total: number;
}

/** Configuration options for Nonescape classifier */
export interface LocalClassifierOptions {
  /** Custom model URL or path */
  modelPath?: string;
  /** Callback for loading progress updates */
  onProgress?: (progress: LoadingProgress) => void;
  /** Threshold for classifying as synthetic (default: 0.5) */
  threshold?: number;
}

/** Synthetic content classifier using local ONNX models */
class LocalClassifier implements Classifier {
  private defaultModelPath = "https://nonescape.sfo2.digitaloceanspaces.com/nonescape-small-v0.onnx";
  private imageChannels = 3;
  private imageSize = 224;

  private session: ort.InferenceSession | null = null;
  private isLoading = false;

  constructor(private options: LocalClassifierOptions = {}) {}

  /** Initializes the ONNX model for inference. Must be called before first prediction. */
  async initialize(): Promise<void> {
    if (this.session || this.isLoading) return;
    this.isLoading = true;

    const modelPath = this.options.modelPath ?? this.defaultModelPath;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

    try {
      this.session = await this.loadModel(modelPath, { executionProviders: ["wasm"], logSeverityLevel: 2 });
    } catch (error) {
      throw new Error(`Failed to initialize model: ${error}`);
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Loads ONNX model with optional progress tracking
   *
   * @param modelPath URL or path to model file
   * @param sessionOptions ONNX session configuration
   * @returns Initialized inference session
   */
  private async loadModel(
    modelPath: string,
    sessionOptions: ort.InferenceSession.SessionOptions
  ): Promise<ort.InferenceSession> {
    if (!this.options.onProgress) return await ort.InferenceSession.create(modelPath, sessionOptions);

    const response = await fetch(modelPath);
    const contentLength = response.headers.get("content-length") || "0";

    const totalSize = parseInt(contentLength, 10);
    const reader = response.body?.getReader();
    if (!reader) throw new Error("Model file invalid");

    const chunks: Uint8Array[] = [];
    let loadedSize = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loadedSize += value.length;

      this.options.onProgress?.({ current: loadedSize, total: totalSize || loadedSize });
    }

    const arrayBuffer = new Uint8Array(loadedSize);
    let offset = 0;
    for (const chunk of chunks) {
      arrayBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    return await ort.InferenceSession.create(arrayBuffer.buffer, sessionOptions);
  }

  async predict(input: InputFileType): Promise<ClassificationResult>;
  async predict(input: InputFileType[]): Promise<ClassificationResult[]>;
  async predict(input: InputFileType | InputFileType[]): Promise<ClassificationResult | ClassificationResult[]> {
    if (!this.session) throw new Error("Model not initialized. Call initialize() first.");

    const isBatch = isBatchInput(input);
    const images = isBatch ? input : [input];

    const imageElements = await Promise.all(images.map((img) => loadImage(img)));
    const batchData = Float32Array.from(imageElements.flatMap((img) => Array.from(this.preprocessImage(img))));

    const tensor = new ort.Tensor("float32", batchData, [
      images.length,
      this.imageChannels,
      this.imageSize,
      this.imageSize,
    ]);

    const results = await this.session.run({ x: tensor });
    const logits = results.linear_1.data as Float32Array;
    const threshold = this.options.threshold ?? 0.5;

    const classificationResults = Array.from({ length: images.length }, (_, i) => {
      const imageLogits = logits.slice(i * 2, i * 2 + 2);
      const probs = this.softmax(Float32Array.from(imageLogits));
      return reweighThreshold(probs[1], threshold);
    });

    return isBatch ? classificationResults : classificationResults[0];
  }

  dispose(): void {
    if (!this.session) return;

    this.session.release?.();
    this.session = null;
  }

  /**
   * Preprocesses image for model inference
   *
   * @param imageElement Source image element
   * @returns Normalized float array in CHW format
   */
  private preprocessImage(imageElement: HTMLImageElement): Float32Array {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;

    const size = this.imageSize;
    const resizeSize = 256;

    canvas.width = size;
    canvas.height = size;

    // Scale and crop image
    const scale = resizeSize / Math.min(imageElement.width, imageElement.height);
    const scaledWidth = imageElement.width * scale;
    const scaledHeight = imageElement.height * scale;

    const cropX = (scaledWidth - size) / 2;
    const cropY = (scaledHeight - size) / 2;

    ctx.drawImage(imageElement, -cropX, -cropY, scaledWidth, scaledHeight);

    const imageData = ctx.getImageData(0, 0, size, size);
    const data = imageData.data;

    // Normalize image
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const pixelCount = size * size;

    const rgbArray = Float32Array.from({ length: this.imageChannels * pixelCount }, (_, idx) => {
      const channel = Math.floor(idx / pixelCount);
      const pixelIdx = idx % pixelCount;
      const pixelValue = data[pixelIdx * 4 + channel] / 255.0;
      return (pixelValue - mean[channel]) / std[channel];
    });

    return rgbArray;
  }

  /**
   * Applies softmax activation to logits
   *
   * @param arr Raw model output logits
   * @returns Probability distribution
   */
  private softmax(arr: Float32Array): Float32Array {
    const max = Math.max(...arr);
    const exps = arr.map((v) => Math.exp(v - max));
    const norm = exps.reduce((a, b) => a + b, 0);

    return exps.map((e) => e / norm);
  }
}

export default LocalClassifier;
