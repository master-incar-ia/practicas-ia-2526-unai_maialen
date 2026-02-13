import torch
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """VGG‑like network with two convolutional blocks and softmax output.

    Each block follows the VGG pattern: conv→ReLU→pool. The first block
    produces 16 channels, the second 32. After pooling, a small classifier
    flattens the features and applies two linear layers, finishing with a
    softmax to yield class probabilities.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # two VGG blocks
        self.features = nn.Sequential(
            # block 1: 3 -> 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # classifier after two pools: feature map size 32 x 8 x 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Cifar10CNN()
    print(model)
    x = torch.randn(1, 3, 32, 32)
    print(model(x))
