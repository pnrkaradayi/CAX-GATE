"""Classifier backbones.

"Proposed CNN": ConvNeXt-Tiny (ImageNet-1k pre-trained) whose last linear layer is replaced by
Dropout(p) + Linear(768, 11). The dropout layer is kept active at test time for Monte Carlo
dropout. Baselines: ResNet-50 and EfficientNet-B0 with their standard heads.
All models take images in [0, 1]; normalisation is part of the model.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def arch_of(name: str, cfg) -> str:
    return cfg.proposed_arch if name == "proposed" else name


def build_backbone(arch: str, num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    tv = torchvision.models
    if arch == "convnext_tiny":
        m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif arch == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(arch)
    return m


class Classifier(nn.Module):
    def __init__(self, arch, num_classes, mean, std, pretrained=True, dropout=0.2):
        super().__init__()
        self.arch = arch
        self.norm = Normalize(mean, std)
        self.net = build_backbone(arch, num_classes, pretrained, dropout)

    def forward(self, x):
        return self.net(self.norm(x))

    def target_layer(self) -> nn.Module:
        """Last convolutional block, used by Grad-CAM."""
        if self.arch == "resnet50":
            return self.net.layer4
        return self.net.features[-1]


def enable_mc_dropout(model: nn.Module):
    """Evaluation mode everywhere except the dropout layers."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def checkpoint_path(cfg, name: str, seed: int, split: str = "grouped") -> Path:
    return cfg.sub("models", split) / f"{name}_seed{seed}.pt"


def save_classifier(path, model: Classifier, mean, std, extra=None):
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "mean": mean, "std": std,
                "extra": extra or {}}, path)


def load_classifier(path, num_classes, device="cpu") -> Classifier:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Classifier(ck["arch"], num_classes, ck["mean"], ck["std"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
