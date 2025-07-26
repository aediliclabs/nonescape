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

import Classifier, {
  ClassificationResult,
  InputFileType,
  isBatchInput,
  loadImage,
  reweighThreshold,
} from "./Classifier";

/** Configuration options for Nonescape classifier */
export interface RemoteClassifierOptions {
  /** API token */
  apiToken?: string;
  /** Custom API base URL */
  baseUrl?: string;
  /** Threshold for classifying as synthetic (default: 0.5) */
  threshold?: number;
}

interface APIResponse {
  results: { ai_prob: number | null; filename: string }[];
}

/** Synthetic content classifier using remote API */
class RemoteClassifier implements Classifier {
  private defaultBaseUrl = "https://api.nonescape.com/v1";

  constructor(private options: RemoteClassifierOptions = {}) {}

  async predict(input: InputFileType): Promise<ClassificationResult>;
  async predict(input: InputFileType[]): Promise<ClassificationResult[]>;
  async predict(input: InputFileType | InputFileType[]): Promise<ClassificationResult | ClassificationResult[]> {
    const baseUrl = this.options.baseUrl ?? this.defaultBaseUrl;
    const threshold = this.options.threshold ?? 0.5;

    const isBatch = isBatchInput(input);
    const result = await this.predictBatch(isBatch ? input : [input], baseUrl, threshold);
    return isBatch ? result : result[0];
  }

  private async predictBatch(
    inputs: InputFileType[],
    baseUrl: string,
    threshold: number
  ): Promise<ClassificationResult[]> {
    const formData = new FormData();
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      let file: File;

      if (input instanceof File) {
        file = input;
      } else {
        const imageElement = await loadImage(input);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d")!;

        canvas.width = imageElement.width;
        canvas.height = imageElement.height;
        ctx.drawImage(imageElement, 0, 0);

        const blob = await new Promise<Blob>((resolve, reject) =>
          canvas.toBlob(
            (blob) => (blob ? resolve(blob) : reject(new Error("Failed to create blob from canvas"))),
            "image/jpeg",
            0.9
          )
        );
        file = new File([blob], `image_${i}.jpg`, { type: "image/jpeg" });
      }

      formData.append("files", file);
    }

    const headers: HeadersInit = {};
    if (this.options.apiToken) headers.Authorization = `Bearer ${this.options.apiToken}`;

    const response = await fetch(`${baseUrl}/predict`, { body: formData, headers, method: "POST" });
    if (!response.ok) throw new Error(`API request failed: ${response.status} ${response.statusText}`);

    const data: APIResponse = await response.json();

    const errors = data.results.filter((r) => r.ai_prob === null);
    if (errors.length > 0) throw new Error(`Failed to process image(s): ${errors.map((e) => e.filename).join(", ")}`);

    return data.results.map((r) => reweighThreshold(r.ai_prob!, threshold));
  }

  dispose(): void {}
}

export default RemoteClassifier;
