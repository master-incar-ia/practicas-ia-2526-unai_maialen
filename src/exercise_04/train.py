from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from tqdm import tqdm

from .dataset import CIFAR10Dataset
from .model import Cifar10CNN


def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option.

    force: 'auto'|'cpu'|'cuda' - when 'auto' will pick cuda if available.
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.numel()


def train_model(output_folder: Path, device: torch.device):
    # Data transforms
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    # Datasets
    root = Path(__file__).parent.parent.parent / "data"
    train_full = CIFAR10Dataset(root, train=True, transform=train_transform, download=True)
    train_full_eval = CIFAR10Dataset(root, train=True, transform=eval_transform, download=False)
    test_dataset = CIFAR10Dataset(root, train=False, transform=eval_transform, download=False)

    # Split train into train/val with shared indices
    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size
    train_subset, val_subset = random_split(
        range(len(train_full)), [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_dataset = Subset(train_full, train_subset.indices)
    val_dataset = Subset(train_full_eval, val_subset.indices)

    # DataLoaders optimizados para recursos limitados
    pin_memory = device.type == "cuda"
    batch_size = 16  # Reducido de 32 a 16 para usar menos memoria

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=0,  # 0 para evitar problemas de multiprocessing en Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )

    # Model, loss, optimizer optimizados para recursos limitados
    model = Cifar10CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # Learning rate más bajo para modelo más pequeño
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)

    # Menos épocas para entrenamiento más rápido
    num_epochs = 60
    best_val_acc = 0.0
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += _accuracy(logits, targets)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_acc += _accuracy(logits, targets)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
            )

    print(f"Best validation acc: {best_val_acc:.3f}, Model saved to {best_model_path}")

    # Test accuracy
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            test_acc += _accuracy(logits, targets)
    test_acc /= len(test_loader)
    print(f"Test Acc: {test_acc:.3f}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(output_folder / "loss_plot.png")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_accs, label="Train Acc")
    plt.plot(range(num_epochs), val_accs, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(output_folder / "accuracy_plot.png")
    plt.close()


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")  # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")
    train_model(output_folder, device=device)
