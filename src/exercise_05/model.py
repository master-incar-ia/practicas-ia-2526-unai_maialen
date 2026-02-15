import torch
import torch.nn as nn


class MultiPerceptron(nn.Module):
    """Red neuronal multicapa para aprender funciones no lineales"""

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Crear capas ocultas con activación ReLU
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Capa de salida sin activación (para regresión)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x, use_activation=True):
        return self.network(x)


if __name__ == "__main__":
    model = MultiPerceptron(1, [256, 128, 64], 1)
    print(model)
    x = torch.tensor([[1.0]])
    print(model(x))
