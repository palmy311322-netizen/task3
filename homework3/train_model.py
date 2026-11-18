import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import CNN, MLP, ResNet18
from trainer import Trainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
DATASET = "cifar10"  # or "fashionmnist"
MODEL = 'resnet18'  # or "cnn" or "mlp"
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
SEEDS = [0, 1, 2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def data_loaders(dataset_name):
    if dataset_name == "fashionmnist":
        cls = datasets.FashionMNIST
        norm = transforms.Normalize((0.2860,), (0.3530,))
    else:
        cls = datasets.CIFAR10
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))
    tf = transforms.Compose([transforms.ToTensor(), norm])

    train_full = cls(DATA_ROOT, train=True, download=False, transform=tf)
    test_set = cls(DATA_ROOT, train=False, download=False, transform=tf)

    n_val = int(len(train_full) * 0.1)
    train_set, val_set = random_split(
        train_full, [len(train_full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))

    sample, _ = train_full[0]
    return (DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_set, batch_size=512),
            DataLoader(test_set, batch_size=512),
            sample.shape[0],
            sample.shape[1])


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
            total_correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, total_correct / total


def run_seed(seed, dataset_name):
    set_seed(seed)
    train_loader, val_loader, test_loader, in_channels, input_size = \
        data_loaders(dataset_name)

    if MODEL == 'cnn':
        model = CNN(in_channels, input_size)
    elif MODEL == 'resnet18':
        model = ResNet18(in_channels, input_size)
    else:
        model = MLP(in_channels * input_size * input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model, optimizer, criterion, device=DEVICE,
        save_path=PROJECT_ROOT / 'homework3' / f'best_{MODEL}_seed{seed}.pth',
        monitor='val_acc', patience=8,
        plot_path=PROJECT_ROOT / 'homework3' / f'learning_curves_{MODEL}_seed{seed}.png',
    )
    print(f'\n===== seed = {seed}, model = {MODEL} =====')
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, plot=False)

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'seed {seed}: test loss {test_loss:.4f}, acc {test_acc:.2%}')

    trainer.plot_curves(show=False)
    return test_acc


def main():
    accs = [run_seed(s, DATASET) for s in SEEDS]

    mean_acc = statistics.mean(accs)
    std_acc = statistics.stdev(accs)

    print('\n各次结果:', ', '.join(f'{a:.2%}' for a in accs))
    print(f'Test accuracy = {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%')

if __name__ == '__main__':
    main()
