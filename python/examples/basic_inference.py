#!/usr/bin/env python3
#
# Copyright 2025 Aedilic Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
from PIL import Image
from nonescape import NonescapeClassifier, NonescapeClassifierSmall, preprocess_image


def main():
    parser = argparse.ArgumentParser(description="Classify image as authentic or AI-generated")
    parser.add_argument("model_path", help="Path to model file (.safetensors)")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--small", action="store_true", help="Use small model variant")
    args = parser.parse_args()

    model = (NonescapeClassifierSmall if args.small else NonescapeClassifier).from_pretrained(args.model_path)
    model.eval()

    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    tensor = preprocess_image(image)

    with torch.no_grad():
        probs = model(tensor.unsqueeze(0))
        authentic_prob = probs[0][0].item()
        ai_prob = probs[0][1].item()

    print(f"Image: {args.image_path}")
    print(f"Authentic probability: {authentic_prob:.2%}")
    print(f"Synthetic probability: {ai_prob:.2%}")

    if ai_prob > 0.5:
        print("Classification: Synthetic")
    else:
        print("Classification: Authentic")


if __name__ == "__main__":
    main()
