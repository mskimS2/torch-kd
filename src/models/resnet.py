import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from models.utils import init_weights


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock, self).__init__()
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual_layer(x)
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, in_channels: List[int], layers: List[int], num_classes: int = 10, use_featuremap: bool = True):
        super(ResNet, self).__init__()
        assert len(in_channels) == len(layers), "`in_channels` and `layers` must have the same length"
        self.use_featuremap = use_featuremap
        self.initial_channels = in_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_channels),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            stride = 2 if i > 0 else 1  # Apply stride of 2 for all layers except the first
            self.layers.append(
                self._make_layer(in_channels[i], in_channels[i] if i == 0 else in_channels[i - 1], layers[i], stride)
            )

        self.classifier = nn.Linear(in_channels[-1], num_classes)

        self.apply(init_weights)

    def _make_layer(self, in_channels: int, prev_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResNetBlock(prev_channels, in_channels, stride)]
        layers += [ResNetBlock(in_channels, in_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.conv1(x)

        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)

        out = F.avg_pool2d(x, 4).view(x.size(0), -1)
        logits = self.classifier(out)

        if self.use_featuremap:
            return logits, feature_maps

        return logits, []


# test code
if __name__ == "__main__":
    in_channels = [16, 32, 64, 128]
    layers = [3, 4, 6, 3]
    num_classes = 10

    model = ResNet(in_channels, layers, num_classes)

    input_tensor = torch.randn(1, 3, 32, 32)

    logits, feature_maps = model(input_tensor)

    print("Logits shape:", logits.shape)
    for i, fmap in enumerate(feature_maps):
        print(f"Feature map layer{i+1} shape:", fmap.shape)

    print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
