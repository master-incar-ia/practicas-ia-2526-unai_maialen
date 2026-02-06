from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset


class NoisyRegressionDataset(Dataset):
    def __init__(self, noise_std=20, size=100, seed=42, normalize=True):
        np.random.seed(seed)
        self.x = np.random.uniform(0, 100, size=(size,))
        self.delta = np.random.normal(0, noise_std, size=(size,))
        self.y = -3 * self.x * self.x + 5 * self.x + self.delta

        # Create a DataFrame for visualization
        df = pd.DataFrame(data=np.array([self.x, self.y]).transpose(), columns=["x", "y"])
        self.df = df

        # Reshape for PyTorch compatibility
        self.x = self.x.reshape((-1, 1))
        self.y = self.y.reshape((-1, 1))

        # Normalize features/targets if requested
        self.normalize = normalize
        if self.normalize:
            self.x_mean = self.x.mean()
            self.x_std = self.x.std() + 1e-8
            self.y_mean = self.y.mean()
            self.y_std = self.y.std() + 1e-8
            self.x = (self.x - self.x_mean) / self.x_std
            self.y = (self.y - self.y_mean) / self.y_std

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
