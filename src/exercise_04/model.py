import torch
import torch.nn as nn


class Cifar10CNN(nn.Module):
    """CNN siguiendo arquitectura VGGnet con dos capas convolucionales y Softmax final."""

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extractor estilo VGGnet con 16 y 32 canales
        self.features = nn.Sequential(
            # Primera capa convolucional: 3 → 16 canales
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Segunda capa convolucional: 16 → 32 canales
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # MaxPooling para reducir dimensionalidad espacial
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Clasificador estilo VGGnet
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Después del pooling: 32 canales × 16×16 = 8192 features
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
            # Softmax para probabilidades de clasificación
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = Cifar10CNN()
    print(model)
    x = torch.randn(1, 3, 32, 32)
    print(model(x))
