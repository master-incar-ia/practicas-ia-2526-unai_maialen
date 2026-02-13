from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from .dataset import CIFAR10Dataset
from .model import Cifar10CNN


def _confusion_matrix(num_classes: int, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    return cm


def _precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return precision, recall, f1


def evaluate_and_plot(loader, model, dataset_name, class_names, output_folder, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = _confusion_matrix(len(class_names), all_preds, all_targets)
    precision, recall, f1 = _precision_recall_f1(cm)
    accuracy = (all_preds == all_targets).mean()

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }

    print(f"Evaluation metrics for {dataset_name} dataset:")
    print(metrics)

    # Confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.tight_layout()
    plt.savefig(output_folder / f"{dataset_name}_confusion_matrix.png")
    plt.close()

    # Per-class metrics table
    per_class = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        index=class_names,
    )
    per_class.to_csv(output_folder / f"{dataset_name}_per_class_metrics.csv")

    return metrics


def save_metrics_as_picture(metrics, filepath):
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 3 decimal places
    df = df.round(3)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(8, 2))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )

    # Save the plot as an image
    plt.savefig(filepath)


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    # Set the seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    # Datasets
    root = Path(__file__).parent.parent.parent / "data"
    train_full_eval = CIFAR10Dataset(root, train=True, transform=eval_transform, download=False)
    test_dataset = CIFAR10Dataset(root, train=False, transform=eval_transform, download=False)

    # Split train into train/val with shared indices
    train_size = int(0.9 * len(train_full_eval))
    val_size = len(train_full_eval) - train_size
    train_subset, val_subset = random_split(
        range(len(train_full_eval)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset = Subset(train_full_eval, train_subset.indices)
    val_dataset = Subset(train_full_eval, val_subset.indices)

    # DataLoaders optimizados para recursos limitados
    pin_memory = device.type == "cuda"
    batch_size = 32  # Reducido de 256 a 32 para evitar problemas de memoria

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,  # 0 para evitar problemas de multiprocessing en Windows
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=0
    )

    # Load the best model weights
    model = Cifar10CNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(output_folder / "best_model.pth", map_location=device))

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    class_names = train_full_eval.data.classes
    metrics["train"] = evaluate_and_plot(
        train_loader, model, "train", class_names, output_folder, device
    )
    metrics["validation"] = evaluate_and_plot(
        val_loader, model, "validation", class_names, output_folder, device
    )
    metrics["test"] = evaluate_and_plot(
        test_loader, model, "test", class_names, output_folder, device
    )

    # save  metrics as csv
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics, output_folder / "metrics.png")

    print("Evaluation complete!")
