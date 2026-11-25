"""Shared DFH backbone and training objective."""

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels, out_channels, kernel_size, *, padding=0, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class DFHNet(nn.Module):
    """Eight convolutional layers and two fully connected layers for square RGB input."""

    def __init__(self, code_bits=96, image_size=56):
        super().__init__()
        self.attention_conv1 = _conv_block(3, 32, 3, padding=1)
        self.attention_conv2 = _conv_block(32, 64, 3, padding=1)
        self.attention_conv3 = _conv_block(64, 128, 3, padding=1)
        self.attention_conv4 = _conv_block(128, 3, 1)
        self.features = nn.Sequential(
            *_conv_block(3, 32, 3, pool=True).children(),
            *_conv_block(32, 64, 2, pool=True).children(),
            *_conv_block(64, 128, 2, pool=True).children(),
        )
        self.conv4 = _conv_block(128, 256, 2)
        feature_size = (image_size - 2) // 2
        feature_size = (feature_size - 1) // 2
        feature_size = (feature_size - 1) // 2
        fusion_size = 128 * feature_size**2 + 256 * (feature_size - 1)**2
        self.face_features_layer = nn.Sequential(
            nn.Linear(fusion_size, 8192), nn.BatchNorm1d(8192), nn.ReLU(inplace=True)
        )
        self.hash_layer = nn.Sequential(nn.Linear(8192, code_bits), nn.BatchNorm1d(code_bits))

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        trunk = self.attention_conv1(images)
        trunk = self.attention_conv2(trunk)
        trunk = self.attention_conv3(trunk)
        trunk = self.attention_conv4(trunk)
        features = self.features(trunk)
        fusion = torch.cat((features.flatten(1), self.conv4(features).flatten(1)), dim=1)
        return self.hash_layer(self.face_features_layer(fusion))


class DFHLoss(nn.Module):
    """Shared class-wise loss and original discrete target update."""

    def __init__(self, num_classes, code_bits, variance, eta, objective):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, code_bits))
        self.variance = variance
        self.eta = eta
        self.objective = objective

    def _classwise(self, outputs, labels):
        squared_norms = (
            outputs.square().sum(dim=1, keepdim=True)
            + self.centers.square().sum(dim=1)
        )
        distances_squared = squared_norms.addmm_(outputs, self.centers.T, alpha=-2)
        weights = (-0.5 * (1.0 / self.variance) * distances_squared).exp()
        probabilities = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        classes = torch.arange(self.centers.size(0), device=outputs.device)
        label_mask = labels.reshape(-1, 1) == classes
        return -((probabilities + 1e-6).log() * label_mask).sum() / len(labels)

    def _regression_values(self, outputs, labels):
        one_hot = F.one_hot(labels, num_classes=self.centers.size(0)).to(outputs.dtype)
        regression = torch.pinverse(one_hot.T @ one_hot) @ one_hot.T @ outputs.sign()
        return one_hot @ regression

    def forward(self, outputs, labels):
        classwise = self._classwise(outputs, labels)
        if self.objective == "classwise":
            return classwise

        with torch.no_grad():
            if self.objective == "classwise-quantization":
                target = (outputs > 0).to(outputs.dtype).mul_(2).sub_(1)
            else:
                target = (self._regression_values(outputs, labels) + self.eta * outputs).sign()

        quantization = (outputs - target).square().sum(dim=1).mean()
        return classwise + self.eta * quantization
