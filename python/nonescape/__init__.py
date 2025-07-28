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

from __future__ import annotations
import torch
from torch import Tensor, nn
import torchvision.models as models
import torchvision.transforms.v2 as T
from transformers import Dinov2Model, Dinov2Config
from safetensors.torch import load_file
from PIL import Image


def preprocess_image(image: Image.Image) -> Tensor:
    """Preprocess image for Nonescape models.

    Args:
        image: PIL Image

    Returns:
        Preprocessed tensor ready for model input
    """
    transform = T.Compose(
        [
            T.ToImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.JPEG(quality=100),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


class NonescapeClassifier(nn.Module):
    """ViT/EfficientNet-based fake image detector

    Uses a vision transformer as backbone with EfficientNet v2 to compute an attention maps for the ViT features.
    """

    def __init__(self, num_classes: int = 2, num_heads=16, num_queries: int = 128):
        super().__init__()

        self.embedding_size = 1024
        self.num_queries = num_queries

        vit_backbone = Dinov2Model(Dinov2Config.from_pretrained("facebook/dinov2-large"))
        efficientnet = models.efficientnet_v2_l(weights=None, num_classes=num_queries * self.embedding_size)
        self.vit_backbone = vit_backbone
        self.query_net = efficientnet
        self.key_net = nn.Linear(self.embedding_size, self.embedding_size)
        self.value_net = nn.Linear(self.embedding_size, self.embedding_size)
        self.attention = nn.MultiheadAttention(self.embedding_size, num_heads=num_heads, batch_first=True)
        self.head = nn.Linear(self.embedding_size, num_classes)

        self.register_buffer("_input_mean", torch.empty((3, 1, 1)))
        self.register_buffer("_input_std", torch.empty((3, 1, 1)))

    @classmethod
    def from_pretrained(cls, path: str) -> NonescapeClassifier:
        state_dict = load_file(path)

        model = cls()
        model.load_state_dict(state_dict)

        return model

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        with torch.no_grad():
            vit_output = self.vit_backbone(x)
            vit_features = vit_output.last_hidden_state  # CLS token [B, embed_dim]
        q = self.query_net.forward(x).reshape(B, self.num_queries, -1)  # [B, num_queries, embed_dim]
        k = self.key_net.forward(vit_features)  # [B, D, embed_dim]
        v = self.value_net.forward(vit_features)  # [B, D, embed_dim]

        emb, _ = self.attention.forward(q, k, v)  # [B, num_queries, embed_dim]
        emb = emb.mean(dim=1)
        logits = self.head(emb.squeeze(1))
        probs = nn.functional.softmax(logits, dim=-1)

        return probs


class NonescapeClassifierMini(nn.Module):
    """EfficientNet-based fake image detector"""

    in_features = None  # Override

    def __init__(self, num_classes: int = 2, embedding_size: int = 1024, dropout: float = 0.2):
        super().__init__()

        self.backbone = models.efficientnet_v2_s(weights=None, num_classes=embedding_size, dropout=dropout)
        self.head = nn.Linear(embedding_size, num_classes)

    @classmethod
    def from_pretrained(cls, path: str) -> NonescapeClassifierMini:
        state_dict = load_file(path)

        model = cls()
        model.load_state_dict(state_dict)

        return model

    def forward(self, x: Tensor) -> Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        probs = nn.functional.softmax(logits, dim=-1)

        return probs


__all__ = ["NonescapeClassifier", "NonescapeClassifierMini", "preprocess_image"]
