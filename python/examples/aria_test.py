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

"""Multi-GPU evaluation script for nonescape classifier on ARIA dataset.

Paper: https://arxiv.org/pdf/2404.14581

Single GPU/CPU:
`python aria_test.py model.safetensors --data-path ./aria_dataset`

Multi-GPU with torchrun:
`torchrun --nproc_per_node=8 aria_test.py model.safetensors --data-path ./aria_dataset`
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from nonescape import NonescapeClassifier, NonescapeClassifierSmall, preprocess_image

AI_DIRS = ["DALL-E", "DreamStudio", "Midjourney", "StarryAI"]
IMG_TYPES = ["T2I", "IT2I"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

REAL_CATEGORY = 0
CATEGORIES = {REAL_CATEGORY: "Real"}
for i, ai_dir in enumerate(AI_DIRS):
    for j, img_type in enumerate(IMG_TYPES):
        cat_id = 1 + i * len(IMG_TYPES) + j
        CATEGORIES[cat_id] = f"{ai_dir}_{img_type}"

CAT_TO_ID = {name: cat_id for cat_id, name in CATEGORIES.items()}


class ARIADataset(Dataset):
    def __init__(self, image_paths_and_categories: List[Tuple[str, int]]):
        self.items = image_paths_and_categories

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, category_id = self.items[idx]
        with Image.open(img_path) as image:
            tensor = preprocess_image(image.convert("RGB"))
        return tensor, category_id


def setup_distributed():
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        return 0, 1, device

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    return rank, world_size, device


def cleanup_distributed():
    if "WORLD_SIZE" in os.environ:
        dist.destroy_process_group()


def gather_results(
    scores: List[float], categories: List[int], rank: int, world_size: int, device: str
) -> Tuple[List[float], List[int]]:
    if world_size == 1:
        return scores, categories

    num_samples = torch.tensor(len(scores), dtype=torch.long, device=device)
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(world_size)]
    dist.all_gather(all_num_samples, num_samples)
    all_num_samples = [t.item() for t in all_num_samples]
    max_samples = max(all_num_samples)

    padded_scores = scores + [0.0] * (max_samples - len(scores))
    scores_tensor = torch.tensor(padded_scores, dtype=torch.float32, device=device)
    all_scores = [torch.zeros_like(scores_tensor) for _ in range(world_size)]
    dist.all_gather(all_scores, scores_tensor)

    padded_categories = categories + [0] * (max_samples - len(categories))
    categories_tensor = torch.tensor(padded_categories, dtype=torch.long, device=device)
    all_categories = [torch.zeros_like(categories_tensor) for _ in range(world_size)]
    dist.all_gather(all_categories, categories_tensor)

    if rank != 0:
        return [], []

    final_scores = []
    final_categories = []
    for scores_t, cats_t, n_samples in zip(all_scores, all_categories, all_num_samples):
        final_scores.extend(scores_t[:n_samples].cpu().tolist())
        final_categories.extend(cats_t[:n_samples].cpu().tolist())

    return final_scores, final_categories


CORRUPTED_IMAGES = ["StarryAI/IT2I/ins/1489777465165169105.png"]  # Dataset contains corrupted images


def collect_images(data_path: Path, max_samples: int = None) -> List[Tuple[str, int]]:
    images = []

    real_dir = data_path / "REAL"
    if real_dir.exists():
        for img_path in real_dir.rglob("*"):
            if img_path.suffix.lower() in IMG_EXTENSIONS:
                images.append((str(img_path), REAL_CATEGORY))

    for ai_dir in AI_DIRS:
        ai_path = data_path / ai_dir
        if not ai_path.exists():
            continue

        for img_type in IMG_TYPES:
            type_path = ai_path / img_type
            if not type_path.exists():
                continue

            categori_id = CAT_TO_ID[f"{ai_dir}_{img_type}"]
            for img_path in type_path.rglob("*"):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    images.append((str(img_path), categori_id))

    images = [i for i in images if str(Path(i[0]).relative_to(data_path)) not in CORRUPTED_IMAGES]
    if max_samples and len(images) > max_samples:
        images.sort()
        images = random.sample(images, max_samples)

    return images


def evaluate_model(model, dataloader: DataLoader, device: str, rank: int = 0) -> Tuple[List[float], List[int]]:
    all_scores, all_categories = [], []

    model.eval()
    with torch.no_grad():
        for batch_tensors, batch_categories in tqdm(dataloader, desc="Evaluating", disable=(rank != 0)):
            probs = model(batch_tensors.to(device))
            synth_probs = probs[:, 1].cpu().tolist()

            all_scores.extend(synth_probs)
            all_categories.extend(batch_categories.tolist())

    return all_scores, all_categories


def calculate_metrics(scores: List[float], categories: List[int], threshold: float = 0.5) -> dict:
    if not scores or not categories:
        raise Exception("Cannot calculate metrics for empty scores and/or categories")

    y_true = [0 if cat == REAL_CATEGORY else 1 for cat in categories]
    y_pred = [1 if score > threshold else 0 for score in scores]

    total_acc = accuracy_score(y_true, y_pred)
    total_ap = average_precision_score(y_true, scores)

    category_stats = {}
    for cat_id in set(categories):
        cat_name = CATEGORIES[cat_id]
        cat_indices = [i for i, cat in enumerate(categories) if cat == cat_id]
        cat_true = [y_true[i] for i in cat_indices]
        cat_pred = [y_pred[i] for i in cat_indices]

        accuracy = sum(1 for t, p in zip(cat_true, cat_pred) if t == p) / len(cat_true)
        category_stats[cat_name] = {"accuracy": accuracy, "count": len(cat_indices)}

    return {
        "total_accuracy": total_acc,
        "total_ap": total_ap,
        "category_stats": category_stats,
        "num_real": sum(1 for cat in categories if cat == REAL_CATEGORY),
        "num_ai": sum(1 for cat in categories if cat != REAL_CATEGORY),
    }


def main():
    parser = argparse.ArgumentParser(description="Test nonescape classifier on ARIA dataset")
    parser.add_argument("model_path", help="Path to model file (.safetensors)")
    parser.add_argument("--data-path", default="./aria_dataset", help="Path to dataset directory")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to use")
    parser.add_argument("--small", action="store_true", help="Use small model variant")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for predictions (for multiple GPUs, this is per GPU)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for image loading (for multiple GPUs, this is per GPU)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    rank, world_size, device = setup_distributed()

    def log(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    data_path = Path(args.data_path)
    if not data_path.exists() or not (data_path / "REAL").exists():
        log(f"Dataset not found at {data_path}")
        log("Run the following command to download and prepare the dataset:")
        log(
            f"mkdir -p {data_path} && cd {data_path} && "
            f"wget https://storage.googleapis.com/nonescape-public/aria_dataset.tar.gz && "
            f"tar -xzf aria_dataset.tar.gz && mv aria_dataset/* . && rm -r aria_dataset aria_dataset.tar.gz &&"
            f"find . -name '*.zip' -execdir unzip -qq -o '{{}}' \\; -exec rm '{{}}' \\; &&"
            f"find . -type d -name __MACOSX -exec rm -r '{{}}' \\; 2>/dev/null"
        )
        cleanup_distributed()
        return

    try:
        model = (NonescapeClassifierSmall if args.small else NonescapeClassifier).from_pretrained(args.model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        log(f"Error loading model: {e}")
        cleanup_distributed()
        return

    log(f"Using {device} on {world_size} process(es)")
    log("Collecting images (types: T2I, IT2I)...")

    images = collect_images(data_path, args.max_samples)
    dataset = ARIADataset(images)

    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    log(f"Evaluating model on {len(images)} images across {world_size} process(es)...")

    local_scores, local_categories = evaluate_model(model, dataloader, device, rank)
    final_scores, final_categories = gather_results(local_scores, local_categories, rank, world_size, device)

    if rank != 0:
        cleanup_distributed()
        return

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    for threshold in [0.5, 0.65, 0.8]:
        results = calculate_metrics(final_scores, final_categories, threshold)

        print(("=" * 16) + f" threshold: {threshold} " + ("=" * 16))
        print(f"Total samples:    {results['num_real'] + results['num_ai']:,}")
        print(f"Real images:      {results['num_real']:,}")
        print(f"Synthetic images: {results['num_ai']:,}")
        print()
        print("OVERALL METRICS:")
        print(f"  Total accuracy:     {results['total_accuracy']:.3f}")
        print(f"  Average precision:  {results['total_ap']:.3f}")
        print()
        print("PER-CATEGORY ACCURACY:")

        cats = list(results["category_stats"].keys())
        sorted_cats = [c for c in cats if c == "Real"] + sorted([c for c in cats if c != "Real"])
        for category in sorted_cats:
            stats = results["category_stats"][category]
            print(f"  {category:<20} {stats['accuracy']:.3f} ({stats['count']:,} samples)")

    cleanup_distributed()


if __name__ == "__main__":
    main()
