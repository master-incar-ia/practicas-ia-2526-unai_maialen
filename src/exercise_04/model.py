import torch
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """CNN simple para clasificacion de CIFAR-10 (3x32x32)."""

    def __init__(self, num_classes=10):
        super().__init__()

        # feature extractor reduced to two conv layers
        # first layer outputs 16 channels, second layer outputs 32 channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # adjust classifier input to match feature output size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # after one pooling, input spatial dims 16x16 -> 32 channels
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = Cifar10CNN()
    print(model)
    x = torch.randn(1, 3, 32, 32)
    print(model(x))
