"""在你自己的照片上微调 MNIST 预训练模型。

前置步骤：
    1. python homework1/tutorial.py           # 训练并保存 homework1/mnist_cnn.pth
    2. 拍照 -> homework1/raw_photos/<标签>/    # 标签就是文件夹名 0~9
    3. python homework1/preprocess_photos.py  # 转成 homework1/my_data/<标签>/*.png

然后运行：
    python homework1/finetune.py
    python homework1/finetune.py --freeze     # 只训练最后一层，数据极少时更稳

输出：
    homework1/cnn_custom.pth                 微调后的权重
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    """结构必须与 tutorial.py 中的 CNN 完全一致，否则权重加载不进来。"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def evaluate(model: nn.Module, loader: Data.DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)[0].argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.size(0)
    return correct / max(total, 1)


def mnist_accuracy(model: nn.Module, n: int = 2000) -> float | None:
    """在 MNIST 官方测试集上抽查，确认微调没有把原有能力忘掉。"""
    root = HERE.parent / "mnist"
    if not root.exists():
        return None
    test = torchvision.datasets.MNIST(root=str(root), train=False)
    x = test.data[:n].unsqueeze(1).float().to(DEVICE) / 255.0
    y = test.targets[:n].to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(x)[0].argmax(dim=1)
    return float((pred == y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="用自己的照片微调 MNIST 模型")
    parser.add_argument("--data", type=Path, default=HERE / "my_data")
    parser.add_argument("--weights", type=Path, default=HERE / "mnist_cnn.pth")
    parser.add_argument("--out", type=Path, default=HERE / "cnn_custom.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--freeze", action="store_true", help="冻结卷积层，只训练最后的全连接层")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(
            f"找不到数据目录：{args.data}\n先运行 preprocess_photos.py 生成 28x28 图片。"
        )
    if not args.weights.exists():
        raise SystemExit(
            f"找不到预训练权重：{args.weights}\n先运行 tutorial.py 训练 MNIST 模型。"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            # 轻微几何扰动，模拟写字位置/大小的差异，缓解小样本过拟合
            transforms.RandomAffine(
                degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=0
            ),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )

    # 同一批文件建两个数据集，让训练/验证用不同的 transform
    plain = torchvision.datasets.ImageFolder(str(args.data))
    if len(plain) == 0:
        raise SystemExit(f"{args.data} 里没有图片")
    print("标签映射:", plain.class_to_idx)

    indices = list(range(len(plain)))
    random.Random(args.seed).shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_ratio))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    if not train_idx:
        raise SystemExit("数据太少，至少需要 2 张图片")

    ds_train = torchvision.datasets.ImageFolder(str(args.data), transform=train_tf)
    ds_val = torchvision.datasets.ImageFolder(str(args.data), transform=val_tf)
    train_set = Data.Subset(ds_train, train_idx)
    val_set = Data.Subset(ds_val, val_idx)

    batch_size = min(args.batch_size, len(train_set))
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print(
        f"训练 {len(train_set)} 张 / 验证 {len(val_set)} 张，"
        f"batch={batch_size}，device={DEVICE}"
    )

    cnn = CNN().to(DEVICE)
    cnn.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    if args.freeze:
        for p in cnn.conv1.parameters():
            p.requires_grad = False
        for p in cnn.conv2.parameters():
            p.requires_grad = False
        print("已冻结卷积层，只训练 out 层")

    optimizer = torch.optim.Adam(
        (p for p in cnn.parameters() if p.requires_grad), lr=args.lr
    )
    loss_func = nn.CrossEntropyLoss()

    baseline = mnist_accuracy(cnn)
    if baseline is not None:
        print(f"微调前 MNIST 测试准确率: {baseline:.4f}")

    best_acc = -1.0
    best_state = copy.deepcopy(cnn.state_dict())
    for epoch in range(1, args.epochs + 1):
        cnn.train()
        running = 0.0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
            loss = loss_func(cnn(b_x)[0], b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * b_y.size(0)

        train_loss = running / len(train_set)
        val_acc = evaluate(cnn, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(cnn.state_dict())
        print(
            f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val accuracy {val_acc:.4f}"
        )

    cnn.load_state_dict(best_state)
    torch.save(cnn.state_dict(), args.out)

    after = mnist_accuracy(cnn)
    print(f"\n最佳验证准确率: {best_acc:.4f}")
    if after is not None:
        print(f"微调后 MNIST 测试准确率: {after:.4f}")
    print(f"权重已保存: {args.out}")


if __name__ == "__main__":
    main()
