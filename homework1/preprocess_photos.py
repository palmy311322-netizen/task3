"""把拍摄/扫描的单个数字照片转成 MNIST 风格的 28x28 灰度图。

为什么要转换：
    MNIST 的样本有固定的"样子"——28x28、白字黑底、数字长边约 20 像素、
    重心位于画面中心。手机照片是彩色的、黑字白底、分辨率各不相同，
    如果不先对齐到同样的样子，模型看到的分布和训练时不同，准确率会崩。

目录约定（标签就是文件夹名）：
    homework1/raw_photos/
        0/  IMG_0001.jpg  IMG_0002.png ...
        1/  ...
        ...
        9/

用法（在任何目录下运行都可以）：
    python homework1/preprocess_photos.py
    python homework1/preprocess_photos.py --src homework1/raw_photos --dst homework1/my_data

输出：
    homework1/my_data/<标签>/<原文件名>.png   28x28 灰度图，0=黑底，255=笔画
    homework1/my_data/preview.png             拼图，用来肉眼检查每张是否处理正确
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

HERE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def otsu_threshold(gray: np.ndarray) -> int:
    """经典 Otsu 阈值：让前景/背景两类之间的方差最大。"""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(gray.size)
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return int(best_t)


def _shift(img: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """将图像平移 (dr, dc) 像素，越界部分裁掉（不循环）。"""
    out = np.zeros_like(img)
    h, w = img.shape
    sy0, sy1 = max(0, -dr), min(h, h - dr)
    sx0, sx1 = max(0, -dc), min(w, w - dc)
    if sy0 >= sy1 or sx0 >= sx1:
        return out
    out[sy0 + dr:sy1 + dr, sx0 + dc:sx1 + dc] = img[sy0:sy1, sx0:sx1]
    return out


def to_mnist_style(img: Image.Image, size: int = 28) -> np.ndarray:
    """返回 uint8 [size, size]：0=背景，255=笔画（白字黑底，与 MNIST 一致）。"""
    img = ImageOps.exif_transpose(img)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    # 轻度模糊，压掉纸纹、噪点和压缩伪影
    gray = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32
    )

    thr = otsu_threshold(gray.astype(np.uint8))
    dark = gray < thr
    # 数字一般只占画面的一小部分，取"少数派"作为笔画；
    # 这样白纸黑字、黑板白字两种极性都能处理。
    ink = dark if dark.mean() < 0.5 else ~dark

    if ink.sum() < 10:
        raise ValueError("几乎没有找到笔画像素，检查照片是否包含单个数字")
    if ink.mean() > 0.5:
        raise ValueError(
            "笔画像素超过一半，画面里可能不只有数字（背景/桌面被算进去了），"
            "请让白纸占满整个画面"
        )

    ys, xs = np.nonzero(ink)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    digits = ink[y0:y1, x0:x1].astype(np.float32)

    # 保持长宽比，把长边缩放到 20 像素（MNIST 数字的大致尺寸）
    h, w = digits.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = (
        np.asarray(
            Image.fromarray((digits * 255).astype(np.uint8)).resize(
                (nw, nh), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        / 255.0
    )

    # 放进 28x28 画布中央
    canvas = np.zeros((size, size), dtype=np.float32)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    # 按质心平移，让数字重心落在画布中心（MNIST 的预处理方式）
    total = float(canvas.sum())
    if total > 0:
        ys, xs = np.nonzero(canvas)
        wgt = canvas[ys, xs]
        cy = float((ys * wgt).sum() / total)
        cx = float((xs * wgt).sum() / total)
        target = (size - 1) / 2.0
        canvas = _shift(canvas, int(round(target - cy)), int(round(target - cx)))

    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def make_preview(images: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    """把处理结果拼成一张图，方便肉眼检查。"""
    if not images:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 10
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, arr) in zip(axes.ravel(), images):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="把数字照片转换成 MNIST 风格 28x28 图")
    parser.add_argument("--src", type=Path, default=HERE / "raw_photos")
    parser.add_argument("--dst", type=Path, default=HERE / "my_data")
    parser.add_argument("--size", type=int, default=28)
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(
            f"找不到照片目录：{args.src}\n"
            "请先按标签建立子文件夹，例如 raw_photos/0/、raw_photos/1/ … raw_photos/9/"
        )

    label_dirs = sorted(p for p in args.src.iterdir() if p.is_dir())
    if not label_dirs:
        raise SystemExit(f"{args.src} 下没有标签子文件夹（0~9）")

    total, failed = 0, 0
    preview: list[tuple[str, np.ndarray]] = []
    for label_dir in label_dirs:
        label = label_dir.name
        files = sorted(p for p in label_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"[{label}] 没有图片，跳过")
            continue
        out_dir = args.dst / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            try:
                with Image.open(path) as im:
                    arr = to_mnist_style(im, size=args.size)
            except Exception as exc:  # 单张失败不影响其它图片
                failed += 1
                print(f"[跳过] {path.name}: {exc}")
                continue
            Image.fromarray(arr, mode="L").save(out_dir / f"{path.stem}.png")
            total += 1
            if len(preview) < 40:
                preview.append((label, arr))
        print(f"[{label}] 完成 {len(files)} 张")

    if total == 0:
        raise SystemExit("没有任何图片转换成功")

    make_preview(preview, args.dst / "preview.png")
    print(f"\n共生成 {total} 张 28x28 图 -> {args.dst}")
    if failed:
        print(f"失败 {failed} 张，见上面的提示")
    print(f"检查拼图：{args.dst / 'preview.png'}")


if __name__ == "__main__":
    main()
