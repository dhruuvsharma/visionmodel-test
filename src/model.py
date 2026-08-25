import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


def create_backbone(backbone_name="resnet18"):
    """
    Create a torchvision backbone with no pretrained weights.

    Supported:
        resnet18
        resnet34
        resnet50
    """
    backbone_name = backbone_name.lower()

    if backbone_name == "resnet18":
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=False)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, feature_dim

    if backbone_name == "resnet34":
        try:
            backbone = models.resnet34(weights=None)
        except TypeError:
            backbone = models.resnet34(pretrained=False)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, feature_dim

    if backbone_name == "resnet50":
        try:
            backbone = models.resnet50(weights=None)
        except TypeError:
            backbone = models.resnet50(pretrained=False)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, feature_dim

    raise ValueError(f"Unsupported backbone: {backbone_name}")


class ShirtEncoder(nn.Module):
    """
    Shirt image encoder.

    Returns:
        embedding
        classification logits
    """

    def __init__(
        self,
        num_classes,
        embedding_dim=128,
        backbone_name="resnet18"
    ):
        super().__init__()

        self.backbone, feature_dim = create_backbone(backbone_name)

        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)

        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, dim=-1)

        logits = self.classifier(features)

        return embedding, logits