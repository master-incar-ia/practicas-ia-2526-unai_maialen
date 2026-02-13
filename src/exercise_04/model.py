import torch
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """CNN optimizado para sistemas con recursos limitados usando Global Average Pooling."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extractor con menos canales para reducir parámetros
        # Reducimos de 16->32 canales a 8->16 canales
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling para reducir drasticamente los parámetros
        # Convierte feature maps de [B, 16, 16, 16] a [B, 16, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador mucho más simple y liviano
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 16, 1, 1] -> [B, 16]
            nn.Dropout(p=0.2),  # Menor dropout porque el modelo es más simple
            nn.Linear(16, num_classes),  # Directamente de 16 a 10 clases
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)  # Reduce spatial dimensions to 1x1
        return self.classifier(x)


if __name__ == "__main__":
    model = Cifar10CNN()
    print(model)
    x = torch.randn(1, 3, 32, 32)
    print(model(x))
