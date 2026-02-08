import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset


class NoisyRegressionDataset(Dataset):
    def __init__(self, noise_std=20, size=100, seed=42, normalize_x=False, normalize_y=False):
        np.random.seed(seed)
        x = np.random.uniform(0, 100, size=(size,))
        delta = np.random.normal(0, noise_std, size=(size,))
        y = 100 * np.sin(8 * numpy.pi * x / 100) + 2 + delta

        # Keep raw values for visualization or inverse transforms.
        self.x_raw = x
        self.y_raw = y

        # Create a DataFrame for visualization
        df = pd.DataFrame(data=np.array([x, y]).transpose(), columns=["x", "y"])
        self.df = df

        if normalize_x:
            self.x_min = float(x.min())
            self.x_max = float(x.max())
            x = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        else:
            self.x_min = None
            self.x_max = None

        if normalize_y:
            self.y_mean = float(y.mean())
            self.y_std = float(y.std())
            y = (y - self.y_mean) / (self.y_std + 1e-8)
        else:
            self.y_mean = None
            self.y_std = None

        # Reshape for PyTorch compatibility
        self.x = x.reshape((-1, 1))
        self.y = y.reshape((-1, 1))

    def plot(self, filepath):
        ax = sns.scatterplot(self.df, x="x", y="y")
        ax.set_title("Synthetic noisy data of y=5*x+2")
        plt.savefig(filepath)
        plt.show()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    dataset = NoisyRegressionDataset()
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    # save the plot
    dataset.plot(output_folder / "plot_dataset_example.png")
    dataset.plot(output_folder / "plot_dataset_example.png")
    dataset.plot(output_folder / "plot_dataset_example.png")
