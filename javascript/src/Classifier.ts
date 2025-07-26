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

/** Result of content classification */
export interface ClassificationResult {
  /** Confidence score between 0 and 1 */
  confidence: number;
  /** True if content is artificially generated */
  isSynthetic: boolean;
}

export type InputFileType = File | HTMLImageElement | string;
export const isBatchInput = (input: InputFileType | InputFileType[]): input is Array<InputFileType> =>
  Array.isArray(input);

/**
 * Loads image from various input types
 * @param input File, HTMLImageElement, or URL string
 * @returns HTMLImageElement ready for processing
 */
export async function loadImage(input: File | HTMLImageElement | string): Promise<HTMLImageElement> {
  if (input instanceof HTMLImageElement) return input;

  return new Promise((resolve, reject) => {
    const img = new Image();
    let objectUrl: string | null = null;

    const cleanup = () => objectUrl && URL.revokeObjectURL(objectUrl);

    img.onload = () => {
      cleanup();
      resolve(img);
    };
    img.onerror = () => {
      cleanup();
      reject();
    };

    if (typeof input === "string") {
      img.src = input;
    } else {
      objectUrl = URL.createObjectURL(input);
      img.src = objectUrl;
    }
  });
}

/**
 * Calculates confidence and isSynthetic from AI probability
 * @param prob Probability that content is synthetic (0-1)
 * @param threshold Threshold for classifying as synthetic
 * @returns Classification result
 */
export function reweighThreshold(prob: number, threshold: number = 0.5): ClassificationResult {
  const isSynthetic = prob > threshold;
  const confidence = 0.5 + 0.5 * (isSynthetic ? (prob - threshold) / (1 - threshold) : (threshold - prob) / threshold);
  return { confidence, isSynthetic };
}

/** Synthetic content classifier using ONNX models */
export default interface Classifier {
  /**
   * Predicts if a single image is synthetic.
   *
   * @param input Image to classify (File, HTMLImageElement, or URL string)
   * @returns Classification result with confidence score
   */
  predict(input: InputFileType): Promise<ClassificationResult>;

  /**
   * Predicts if multiple images are synthetic.
   *
   * @param input Array of images to classify
   * @returns Array of classification results with confidence scores
   */
  predict(input: InputFileType[]): Promise<ClassificationResult[]>;

  /** Releases model resources */
  dispose(): void;
}
