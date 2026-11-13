"""用自己拍的真实照片训练一个图像分类模型（迁移学习）。

照片怎么放（标签就是文件夹名，一张照片一个类别）：
    homework1/photos/
        cat/     IMG_0001.jpg  IMG_0002.png ...
        dog/     ...
        flower/  ...
        ...

用法：
    python homework1/train_photos.py                     # 训练
    python homework1/train_photos.py --freeze            # 只训练最后的分类层
    python homework1/train_photos.py --predict 某张照片.jpg
    python homework1/train_photos.py --predict 某个文件夹

特点：
    * 用 ImageNet 预训练的 ResNet18，把最后的全连接层换成你的类别数；
    * 自动做 Resize/裁剪/翻转/旋转/颜色扰动，适配手机照片；
    * 自动读取 EXIF 方向（手机竖拍不会躺倒）；
    * 自动跳过损坏的图片；
    * 按类别分层划分训练/验证集，保存验证集上最好的权重。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN = [0.485, 0.456, 0.406]  # ImageNet 统计量，迁移学习必须用同一套
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


class PhotoDataset(Data.Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(im), label


def scan_dataset(root: Path):
    """遍历 photos/<类别>/<图片>，顺便校验图片能否打开。"""
    if not root.is_dir():
        raise SystemExit(
            f"找不到照片目录：{root}\n"
            f"请按类别建子文件夹，例如 {root / 'cat'}、{root / 'dog'}"
        )
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(classes) < 2:
        raise SystemExit(f"{root} 下至少需要 2 个类别文件夹，目前是 {classes}")

    samples: list[tuple[str, int]] = []
    bad: list[tuple[Path, str]] = []
    counts: dict[str, int] = {}
    for label, name in enumerate(classes):
        files = sorted(
            p for p in (root / name).iterdir() if p.suffix.lower() in IMG_EXTS
        )
        counts[name] = len(files)
        for path in files:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as exc:
                bad.append((path, str(exc)))
                continue
            samples.append((str(path), label))
    return classes, samples, bad, counts


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                # 先随机裁一块再缩放，等价于让模型看不同位置/大小的物体
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 256 / 224)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def stratified_split(samples, num_classes: int, val_ratio: float, seed: int):
    """每个类别按比例分，避免验证集里没有某个类别。"""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        by_class[label].append(index)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label, ids in by_class.items():
        rng.shuffle(ids)
        if len(ids) >= 2:
            n_val = max(1, int(round(len(ids) * val_ratio)))
            n_val = min(n_val, len(ids) - 1)
        else:
            n_val = 0  # 只有一张的类别只能放进训练集
        val_idx.extend(ids[:n_val])
        train_idx.extend(ids[n_val:])
    return train_idx, val_idx


def build_model(num_classes: int, pretrained: bool = True, freeze: bool = False):
    if pretrained:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            model = torchvision.models.resnet18(weights=weights)
        except Exception as exc:
            print(f"[警告] 预训练权重下载失败（{exc}），改为随机初始化从头训练。")
            model = torchvision.models.resnet18(weights=None)
    else:
        model = torchvision.models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model


def evaluate(model: nn.Module, loader: Data.DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.size(0)
    return correct / max(total, 1)


def load_checkpoint(path: Path):
    if not path.exists():
        raise SystemExit(f"找不到模型权重：{path}，先训练一次。")
    ckpt = torch.load(path, map_location=DEVICE)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["classes"]


def predict(path: Path, weights: Path, topk: int = 3) -> None:
    model, classes = load_checkpoint(weights)
    tf = build_transforms(train=False)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        files = [path]
    if not files:
        raise SystemExit(f"{path} 下没有图片")

    for file in files:
        with Image.open(file) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
            x = tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = model(x).softmax(dim=1)[0]
        k = min(topk, len(classes))
        top_p, top_i = probs.topk(k)
        print(file.name)
        for prob, idx in zip(top_p.tolist(), top_i.tolist()):
            print(f"    {classes[idx]:>12s}  {prob:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="用真实照片训练图像分类模型")
    parser.add_argument("--data", type=Path, default=HERE / "photos")
    parser.add_argument("--out", type=Path, default=HERE / "photo_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None, help="默认冻结时 1e-3，全量微调 1e-4")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--freeze", action="store_true", help="冻结骨干网络，只训练分类层")
    parser.add_argument("--workers", type=int, default=0, help="数据加载进程数，Windows 建议 0~4")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 权重")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--predict", type=Path, default=None, help="训练完用模型预测图片或文件夹")
    args = parser.parse_args()

    if args.predict is not None:
        predict(args.predict, args.out)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes, samples, bad, counts = scan_dataset(args.data)
    print("类别及数量:", {k: v for k, v in counts.items()})
    if bad:
        print(f"跳过 {len(bad)} 张无法读取的图片:")
        for path, reason in bad[:10]:
            print(f"  {path.name}: {reason}")
    if len(samples) < 4:
        raise SystemExit("有效图片太少（至少 4 张），无法划分训练/验证集")

    train_idx, val_idx = stratified_split(samples, len(classes), args.val_ratio, args.seed)
    if not val_idx:
        raise SystemExit("验证集为空：请确保每个类别至少有 2 张照片")
    if not train_idx:
        raise SystemExit("训练集为空")

    train_set = PhotoDataset([samples[i] for i in train_idx], build_transforms(True))
    val_set = PhotoDataset([samples[i] for i in val_idx], build_transforms(False))
    batch_size = min(args.batch_size, len(train_set))
    train_loader = Data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = Data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=DEVICE.type == "cuda",
    )
    print(f"训练 {len(train_set)} 张 / 验证 {len(val_set)} 张，batch={batch_size}，device={DEVICE}")

    model = build_model(
        len(classes), pretrained=not args.no_pretrained, freeze=args.freeze
    ).to(DEVICE)
    lr = args.lr if args.lr is not None else (1e-3 if args.freeze else 1e-4)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_func = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = loss_func(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum())
            total += y.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | train loss {running / len(train_set):.4f}"
            f" | train acc {correct / max(total, 1):.3f} | val acc {val_acc:.3f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "arch": "resnet18",
                    "img_size": IMG_SIZE,
                },
                args.out,
            )

    print(f"\n最佳验证准确率: {best_acc:.3f}")
    print(f"权重已保存: {args.out}")
    print(f"预测单张图片: python {Path(__file__).name} --predict 你的照片.jpg")


if __name__ == "__main__":
    main()
