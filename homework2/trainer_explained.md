# trainer.py 逐段详解

> 这份文档按代码顺序把每一段、每一个"看着陌生"的写法拆开讲。
> 建议对照文件 `trainer.py` 的左侧行号一起看。

---

## 0. 先看全局：这个文件到底在干嘛

`trainer.py` 只有一个类 `Trainer`，它的职责是**把 PyTorch 训练模型时每次都要手写的那套循环包起来**。

标准 PyTorch 训练循环本来长这样：

```python
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            ...
```

这套代码每次实验都要重写一遍，而且还要额外处理：验证集评估、保存最优模型、早停、打印日志、画学习曲线。`Trainer` 把这些全部封装成了两个方法：

| 想做的事 | 调用的方法 |
|---|---|
| 训练 + 验证 + 保存 + 早停 | `fit()` |
| 单独画学习曲线 | `plot_curves()` |
| 训练一个 epoch 的内部细节 | `_run_epoch()`（私有） |

调用起来就三行：

```python
trainer = Trainer(model, optimizer, criterion, patience=10)
history = trainer.fit(train_loader, val_loader, epochs=50)
```

---

## 1. 导入部分（第 1–14 行）

```python
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
```

- **`import copy`**：Python 标准库。这里只用到一个函数 `copy.deepcopy()`，作用是"深拷贝"（后面第 109 行会讲为什么必须深拷贝）。
- **`from pathlib import Path`**：比字符串更好用的路径对象。`Path('a/b/model.pth')` 可以 `.parent` 取父目录、`.mkdir()` 建目录，不用手动拼 `os.path.join('a/b', 'model.pth')`。
- **`import matplotlib.pyplot as plt`**：画图库，`plt` 是社区约定俗成的别名。
- **`import torch`**：PyTorch 本体，负责张量运算、保存模型、设备（GPU/CPU）管理。

接下来 8 个 `COLOR_xxx = '#xxxxxx'` 只是**画图用的颜色常量**（十六进制 RGB 字符串），不影响训练逻辑。单独定义常量的好处是：想改配色只改这一处。

- `#2a78d6` 蓝色 → 训练曲线
- `#eb6834` 橙色 → 验证曲线
- 其余是背景色、网格线、坐标轴、文字、弱化文字的颜色

---

## 2. `__init__` 构造函数（第 16–56 行）

### 2.1 函数签名

```python
def __init__(self, model, optimizer, criterion, device=None,
             save_path='best_model.pth', patience=10, min_delta=0.0,
             monitor='val_acc', plot_path='learning_curves.png'):
```

- `self` 指"当前这个对象自己"，是 Python 实例方法固定的第一个参数。
- 带 `=xxx` 的参数是**默认参数**：不传就用默认值。所以最简调用只需要传前三个：
  `Trainer(model, optimizer, criterion)`。
- `device=None` 表示"调用者没指定设备"，后面会自动判断用 GPU 还是 CPU。

### 2.2 存到 `self.xxx` 上

```python
self.model = model
self.optimizer = optimizer
self.criterion = criterion
```

`self.xxx = ...` 就是把值挂到对象身上，**之后所有方法都能通过 `self.xxx` 访问到**。这是面向对象里"保存状态"的基本写法。

### 2.3 自动选设备

```python
self.device = device or torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
self.model.to(self.device)
```

拆成两层理解：

1. 内层 `'cuda' if torch.cuda.is_available() else 'cpu'` 是**条件表达式**（三目运算符），读作"如果 CUDA 可用就用 'cuda'，否则用 'cpu'"。它是 `if/else` 写在一行里的简写。
2. `torch.device('cuda')` 把字符串变成 PyTorch 的设备对象。
3. 外层 `device or ...`：Python 的 `or` **短路求值**——左边为"真"就直接返回左边，左边是 `None`（假值）才返回右边。所以效果是：**传了 device 就用传的，没传就自动选**。

`self.model.to(self.device)` 把模型的所有参数搬到对应设备上（GPU 或 CPU）。**模型和数据必须在同一个设备上**，否则前向计算会报错。

### 2.4 路径处理

```python
self.save_path = Path(save_path)
...
self.plot_path = Path(plot_path)
```

把传入的普通字符串 `'best_model.pth'` 统一转成 `Path` 对象。好处是后面可以写：

```python
self.save_path.parent.mkdir(parents=True, exist_ok=True)  # 第 91 行
```

- `.parent`：父目录。`best_model.pth` 没有目录部分，`Path` 会把它当成当前目录 `.`，`mkdir` 对已存在的 `.` 不会报错。
- `.mkdir(parents=True, exist_ok=True)`：
  - `parents=True`：父目录不存在就**逐级创建**（相当于 `mkdir -p`）。
  - `exist_ok=True`：目录已存在**不报错**。
  - 合起来就是"确保这个目录存在"。

### 2.5 早停与监控指标

```python
self.patience = patience   # 连续多少轮不提升就停；None 表示不早停
self.min_delta = min_delta # 至少提升这么多才算"有提升"
self.monitor = monitor     # 'val_acc' 或 'val_loss'
self.mode = 'max' if monitor == 'val_acc' else 'min'
```

- 准确率**越大越好**，所以 `mode='max'`；loss **越小越好**，所以 `mode='min'`。这个 `mode` 变量后面判断"有没有变好"时会用到。
- `min_delta` 的作用：验证指标在噪声里小幅抖动（比如 0.9001 → 0.9002）不算真的进步。设 `min_delta=0.001` 就要求"至少提升 0.001"才算。

第 21–22 行还有个**参数校验**：

```python
if monitor not in ('val_acc', 'val_loss'):
    raise ValueError("monitor 只能是 'val_acc' 或 'val_loss'")
```

`raise ValueError(...)` 是主动抛异常：参数写错时**立刻**报错，而不是训练 10 个 epoch 后才发现逻辑错了。这叫"fail fast"。

### 2.6 初始化记录容器

```python
self.history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
}
```

一个字典，4 个键，每个值都是**空列表**。每训练一轮就往对应列表 `append` 一个数，最后形成 4 条时间序列，供画图使用。

```python
init_best = -float('inf') if self.mode == 'max' else float('inf')
self.best_metric = init_best
```

`float('inf')` 是浮点数的无穷大，`-float('inf')` 是负无穷。

- 如果是"越大越好"，初始最优值设成 **负无穷**，这样第一轮的任何结果都能"超过"它。
- 如果是"越小越好"，初始设成 **正无穷**，同理。

```python
self.best_epoch = 0    # 目前最优是第几轮
self.best_state = None # 目前最优的模型权重（state_dict 的深拷贝）
self.no_improve = 0    # 连续多少轮没提升了
```

---

## 3. `_run_epoch`：跑完一个 epoch（第 58–83 行）

方法名前面加一个下划线 `_run_epoch` 是 Python 的**约定**：表示"这是内部方法，外面别直接调"，但语法上并没有强制限制。

### 3.1 切换 train/eval 模式

```python
self.model.train(train)
```

`train(True)` 等价于 `model.train()`，`train(False)` 等价于 `model.eval()`。两者是同一个方法的布尔参数。

**为什么必须切换？** 有些层在训练和推理时行为不同：

- `Dropout` 训练时随机丢神经元，评估时*不*丢；
- `BatchNorm` 训练时用当前 batch 的均值/方差并更新滑动统计量，评估时用滑动统计量。

不切换会导致验证结果不可信，甚至训练不稳定。

### 3.2 累加器

```python
total_loss = 0.0
total_correct = 0
total_samples = 0
```

因为一个 epoch 里有很多 batch，不能只记最后一个 batch 的 loss，要**累加**整个 epoch 的结果，最后再求平均。

### 3.3 梯度开关：`with ... if ... else ...`

```python
with torch.enable_grad() if train else torch.no_grad():
    for x, y in loader:
```

这行有点绕，拆开看：

- `torch.no_grad()` 和 `torch.enable_grad()` 都是**上下文管理器**（配合 `with` 使用的对象），作用是临时修改"要不要记录计算图"的全局开关。
  - `no_grad()`：**关掉**梯度记录 → 省显存、速度快。验证/推理时用。
  - `enable_grad()`：**打开**梯度记录 → 训练需要。写它是因为它显式、不依赖外部状态。
- `A if 条件 else B` 是条件表达式，所以整句是：**训练就开梯度，验证就关梯度**。
- `with X:` 保证进入这个代码块时生效、**离开时自动恢复原状态**，即使中途抛异常也会恢复。

等价于：

```python
if train:
    with torch.enable_grad():
        ...
else:
    with torch.no_grad():
        ...
```

作者只是写得更紧凑。

### 3.4 遍历 DataLoader

```python
for x, y in loader:
```

`DataLoader` 每次产出一个 batch，是一个二元组 `(inputs, labels)`。`for x, y in ...` 是**元组解包**：把元组第一个元素给 `x`、第二个给 `y`。

`x` 的形状通常是 `(batch_size, C, H, W)`（图像）或 `(batch_size, features)`（向量），`y` 形状是 `(batch_size,)`。

### 3.5 数据搬到设备

```python
x, y = x.to(self.device), y.to(self.device)
```

**模型在 GPU 上，输入数据也必须在 GPU 上**。`.to(device)` 返回新张量（若已在该设备上则原样返回）。标签 `y` 也要搬，因为计算 loss 时两者会在同一设备上运算。

### 3.6 前向计算

```python
output = self.model(x)
if isinstance(output, (tuple, list)):
    output = output[0]
```

- `self.model(x)` 调用了模型的 `forward`，得到输出 logits，形状 `(batch_size, num_classes)`。
- 有些模型（如 HuggingFace 的模型、带辅助输出的网络）返回的是元组，比如 `(main_output, auxiliary_output)`。`isinstance(obj, (A, B))` 意思是"obj 是不是 A 或 B 的实例"。这里做**兼容**：如果是元组/列表，就只取第一个（主输出）。

### 3.7 计算损失

```python
loss = self.criterion(output, y)
```

`criterion` 是外部传入的损失函数（如 `nn.CrossEntropyLoss()`）。这里直接调用它，所以**任何符合 `criterion(output, target)` 签名的损失函数都能用**，这就是"通用"的体现。

### 3.8 训练三步曲

```python
if train:
    self.optimizer.zero_grad()  # 1. 清空上一步残留的梯度
    loss.backward()             # 2. 反向传播，计算每个参数的梯度
    self.optimizer.step()       # 3. 按梯度更新参数
```

这三行的顺序是固定的，解释：

1. **`zero_grad()`**：PyTorch 默认会**累加**梯度（`p.grad += new_grad`），不会自动清零。不清零的话，这次梯度会叠加上次的，结果错误。所以每次更新前要先清零。
2. **`backward()`**：从 `loss` 出发沿计算图反向求导，把每个参数的梯度写进 `p.grad`。
3. **`step()`**：优化器读取 `p.grad`，按自己算法（SGD、Adam…）更新参数。

> 为什么验证时没有这三行？因为 `train=False` 直接跳过，只做前向，不更新参数。

### 3.9 累计指标

```python
batch_size = y.size(0)
total_loss += loss.item() * batch_size
total_correct += (output.argmax(dim=1) == y).sum().item()
total_samples += batch_size
```

逐个拆：

- **`y.size(0)`**：batch 里有多少个样本。`size(0)` 就是 `shape[0]`。

- **`loss.item()`**：把只有一个元素的张量转成 Python 浮点数。注意它也会**脱离计算图**（不需要梯度，纯读数值）。

- **`loss.item() * batch_size`**：关键细节。`loss` 本身是这个 batch 的**平均值**。如果直接 `total_loss += loss.item()` 再除以 batch 数，那么最后一个不满的 batch（比如 32 个里只剩 10 个）会被高估权重。乘以 `batch_size` 转成"这个 batch 的总损失"，最后除以 `total_samples`，才是**真正的按样本加权平均**。

- **`output.argmax(dim=1)`**：在 `dim=1`（类别维度）上取最大值的下标，得到每个样本预测的类别。形状从 `(N, C)` 变成 `(N,)`。

- **`... == y`**：逐元素比较，得到布尔张量 `(N,)`，预测对了是 `True`。

- **`.sum().item()`**：`True` 当 1、`False` 当 0 求和，再转成 Python 整数 → 这个 batch 预测对了几个。

- **`total_samples += batch_size`**：累计样本数量，作为最终分母。

### 3.10 返回

```python
return total_loss / total_samples, total_correct / total_samples
```

返回一个元组 `(平均 loss, 准确率)`，准确率范围是 0~1（第 155 行画图时才乘 100 转百分比）。

---

## 4. `_is_better`：判断"有没有变好"（第 85–88 行）

```python
def _is_better(self, metric):
    if self.mode == 'max':
        return metric > self.best_metric + self.min_delta
    return metric < self.best_metric - self.min_delta
```

- `mode='max'`（准确率）：新指标必须 **明显大于** 历史最优（超过 `min_delta` 才算）。
- `mode='min'`（loss）：新指标必须 **明显小于** 历史最优。

用 `+ min_delta` / `- min_delta` 而不是 `>=`，是为了过滤掉噪声级别的小幅抖动。`min_delta=0.0` 时它就退化成普通的"严格更好"判断。

---

## 5. `fit`：训练主循环（第 90–143 行）

### 5.1 签名与建目录

```python
def fit(self, train_loader, val_loader, epochs, verbose=True, plot=True):
    self.save_path.parent.mkdir(parents=True, exist_ok=True)
```

`verbose` 控制是否打印日志，`plot` 控制是否画曲线。逻辑与调试分离，很常用。

### 5.2 主循环

```python
for epoch in range(1, epochs + 1):
```

`range(1, epochs + 1)` 生成 `1, 2, ..., epochs`（左闭右开）。从 1 开始是因为"第几轮"对人来说习惯从 1 数。

```python
train_loss, train_acc = self._run_epoch(train_loader, train=True)
val_loss, val_acc = self._run_epoch(val_loader, train=False)
```

**同一套代码跑两次**，只是 `train` 参数不同。这是 `_run_epoch` 设计成通用方法的回报——避免训练/验证各写一遍几乎相同的循环。

```python
self.history['train_loss'].append(train_loss)
...
```

每轮把 4 个指标追加到历史列表，供画图和第 147 行计算轮数。

### 5.3 挑出本轮监控指标

```python
metric = val_acc if self.monitor == 'val_acc' else val_loss
improved = self._is_better(metric)
```

`monitor` 决定拿哪个指标比较。`improved` 是布尔值，表示本轮是否刷新了最优。

### 5.4 有提升时：存权重 + 落盘

```python
if improved:
    self.best_metric = metric
    self.best_epoch = epoch
    self.no_improve = 0
    self.best_state = copy.deepcopy(self.model.state_dict())
    torch.save({
        'state_dict': self.best_state,
        'epoch': epoch,
        'monitor': self.monitor,
        'metric': self.best_metric,
    }, self.save_path)
```

**这是整个文件最容易出错、也最值得理解的一处。**

- **`model.state_dict()`**：返回一个字典，键是参数名，值是参数张量。**注意：这些张量和模型里的参数共享底层内存**，不是副本。

- **为什么必须 `copy.deepcopy()`？** 因为后面每一轮 `optimizer.step()` 都会**原地修改**模型参数。如果不深拷贝，`self.best_state` 里存的"最优权重"会跟着一起被改掉——最后恢复的时候拿到的是**最后一轮**的权重，而不是最优轮的。`deepcopy` 会递归复制所有张量，得到一份真正独立的快照。

- **`torch.save(对象, 路径)`**：把任意 Python 对象（这里是字典）序列化保存，得到的就是 `best_model.pth`。存成字典而不是裸 state_dict，是为了**同时记录这是第几轮、用哪个指标选的、指标值多少**，以后加载时可以判断这个 checkpoint 的来历。

- `self.no_improve = 0`：这次进步了，连续未提升计数清零。

### 5.5 没提升时

```python
else:
    self.no_improve += 1
```

`+=` 是"加上并赋值"，即 `self.no_improve = self.no_improve + 1`。

### 5.6 打印日志

```python
if verbose:
    flag = ' *' if improved else ''
    print('Epoch %3d | train loss %.4f acc %.2f%% | '
          'val loss %.4f acc %.2f%%%s'
          % (epoch, train_loss, train_acc * 100,
             val_loss, val_acc * 100, flag))
```

- 这里是 **C 语言风格的 `%` 格式化**（和 f-string 不同，但功能类似）：
  - `%3d`：整数，宽度至少 3，右对齐 → 让轮次数字对齐。
  - `%.4f`：浮点数保留 4 位小数。
  - `%.2f`：保留 2 位小数。
  - `%%`：**输出一个真正的 `%` 字符**（因为 `%` 本身是格式符的引导符，想打印它必须写两次）。
  - `%s`：直接转成字符串插入，这里是 `' *'` 或 `''`。
- `train_acc * 100`：准确率从 0~1 转成百分比。
- `flag = ' *' if improved else ''`：刷新最优的那一轮行尾加个星号，方便一眼看出。

### 5.7 早停

```python
if self.patience is not None and self.no_improve >= self.patience:
    if verbose:
        print('早停：验证集 %s 连续 %d 轮未提升，第 %d 轮停止' ...)
    break
```

- 先判断 `patience is not None`：告诉用户"传 None 就不早停"。注意这里**必须用 `is not None`**，不能写 `if patience`，否则 `patience=0` 会被误判成假值。
- `break` 立即跳出 `for` 循环，后面的 epoch 不再训练。
- 打印用 `%d` 占位整数。

### 5.8 循环之后：恢复最优权重

```python
if self.best_state is not None:
    self.model.load_state_dict(self.best_state)
    self.model.eval()
```

- 训练结束时，模型参数是**最后一轮**的（可能已经过拟合了）。把之前深拷贝保存的**最优轮快照**加载回去，交付的才是最好的模型。
- `self.model.eval()` 把模型切到评估模式（关掉 Dropout、BN 用滑动统计量），这样交给用户后直接就能推理。

### 5.9 画图 + 返回

```python
if plot:
    self.plot_curves()
return self.history
```

`self.history` 是字典，调用方可以继续拿它做分析，比如挑选 `val_acc` 最高的 epoch。

---

## 6. `plot_curves`：画学习曲线（第 145–195 行）

### 6.1 准备

```python
save_path = Path(save_path or self.plot_path)
epochs = list(range(1, len(self.history['train_loss']) + 1))
```

- `save_path or self.plot_path`：利用 `or` 短路——调用方传了路径就用传的，传 `None` 就用构造函数里设置的默认路径。
- `range(1, N+1)` 转成列表，得到横轴 `[1, 2, ..., N]`，N 是实际训练了多少轮（可能因为早停而少于 `epochs` 参数）。

### 6.2 创建画布

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
fig.patch.set_facecolor(COLOR_SURFACE)
```

- `plt.subplots(1, 2)`：建一个 **1 行 2 列**的子图网格。返回 `fig`（整张画布）和 `axes`（子图数组，可以 `axes[0]`、`axes[1]` 访问）。
- `figsize=(11, 4.2)`：画布尺寸，宽 11 英寸、高 4.2 英寸。
- `dpi=150`：分辨率，越高越清晰、文件越大。
- `fig.patch` 是画布的"背景矩形"，`set_facecolor` 给它涂上背景色。

### 6.3 要画的两组数据

```python
panels = [
    (axes[0], 'Loss',
     self.history['train_loss'], self.history['val_loss']),
    (axes[1], 'Accuracy (%)',
     [v * 100 for v in self.history['train_acc']],
     [v * 100 for v in self.history['val_acc']]),
]
```

一个列表，每个元素是一个四元组：`(子图, 标题, 训练序列, 验证序列)`。

`[v * 100 for v in self.history['train_acc']]` 是**列表推导式**，一行完成"遍历 + 乘 100 + 组成新列表"，等价于：

```python
tmp = []
for v in self.history['train_acc']:
    tmp.append(v * 100)
```

### 6.4 逐个面板绘制

```python
for ax, title, train_values, val_values in panels:
```

元组解包 + 循环，一次处理一个子图。

```python
ax.set_facecolor(COLOR_SURFACE)
ax.plot(epochs, train_values, color=COLOR_TRAIN,
        linewidth=2, label='Train')
ax.plot(epochs, val_values, color=COLOR_VAL,
        linewidth=2, label='Validation')
```

- `ax.plot(x, y, ...)` 画折线图；`color` 颜色、`linewidth` 线宽、`label` 图例名字（`label` 是给 `legend()` 用的）。
- 一个 `ax` 上画两条线：训练和验证。故意用不同颜色，方便对比。

```python
if self.best_epoch > 0:
    ax.axvline(self.best_epoch, color=COLOR_MUTED,
               linestyle=':', linewidth=1)
```

`axvline(x)` 在 x=`best_epoch` 处画一条**竖直参考线**，标记最优轮次。`linestyle=':'` 是点线。

```python
ax.set_title(title, fontsize=12, color=COLOR_TEXT, loc='left')
ax.set_xlabel('Epoch', fontsize=10, color=COLOR_TEXT)
ax.grid(True, color=COLOR_GRID, linewidth=0.8)
ax.set_axisbelow(True)
```

- `set_title` / `set_xlabel`：标题和 x 轴标签。`loc='left'` 让标题左对齐（一种偏现代的排版）。
- `grid(True, ...)`：打开网格线。
- `set_axisbelow(True)`：网格线画在**数据下面**，不遮挡曲线。

### 6.5 美化边框（spines）

```python
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(COLOR_AXIS)
```

**spines 就是绘图区四周的边框线**，一共四条：top / right / left / bottom。

- 隐藏上、右边框；
- 左、下边框换成浅灰色。

这是数据可视化里常见的"去框"风格，减少视觉干扰。

```python
ax.tick_params(colors=COLOR_MUTED, labelsize=9)
```

统一设置刻度线/刻度标签的颜色和字号。

```python
ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_TEXT)
```

显示图例。`frameon=False` 去掉图例外面的方框。

### 6.6 标注最优轮次

```python
axes[0].annotate('best epoch %d' % self.best_epoch,
                 xy=(self.best_epoch, 1), xycoords=('data', 'axes fraction'),
                 xytext=(4, -12), textcoords='offset points',
                 fontsize=9, color=COLOR_MUTED, va='top')
```

`annotate` 在图上加一段带位置的文字，参数含义：

| 参数 | 含义 |
|---|---|
| 第一个位置参数 | 要显示的文字 |
| `xy=(x, y)` | 文字**指向**的坐标 |
| `xycoords=('data', 'axes fraction')` | **混合坐标系**：x 用数据坐标（= 最优轮次），y 用"轴内比例"（1 = 子图顶边）。这样标注始终贴在顶部，不会随 loss 数值大小跑出画面 |
| `xytext=(4, -12)` | 文字相对指向点的偏移 |
| `textcoords='offset points'` | 偏移单位是"点"（打印单位） |
| `va='top'` | 垂直对齐方式：文字顶部对齐到锚点 |

这行只加在左图 `axes[0]` 上，避免重复。

### 6.7 收尾保存

```python
fig.tight_layout()
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
print('学习曲线已保存到:', save_path)
```

- `tight_layout()`：自动调整子图间距，避免标题和轴标签互相重叠。
- `savefig(...)`：
  - `facecolor=fig.get_facecolor()`：保存时沿用画布背景色（否则默认白色，背景色会丢）。
  - `bbox_inches='tight'`：裁掉周围多余空白，图片更紧凑。
- `Path` 对象可以直接传给 `savefig`。

```python
if show:
    plt.show()
else:
    plt.close(fig)
```

- `plt.show()`：弹出窗口显示图片（**会阻塞**，直到手动关掉窗口）。
- `plt.close(fig)`：只保存不显示时，主动释放画布资源。不关闭的话，在循环里反复画图会**累积内存**。

---

## 7. 执行流程总览

```
Trainer(...)                        ← 只做配置，不训练
   │
   └─ fit(train_loader, val_loader, epochs)
        │
        ├─ mkdir(save_path 的父目录)
        │
        └─ for epoch in 1..epochs:
             ├─ _run_epoch(train_loader, train=True)   → 更新参数
             ├─ _run_epoch(val_loader, train=False)    → 只评估
             ├─ history 追加 4 个指标
             ├─ 本轮更好？ → deepcopy 权重 / torch.save / no_improve=0
             │              否则 → no_improve += 1
             ├─ 连续 patience 轮没提升？ → break（早停）
             └─ 打印一行日志
        │
        ├─ load_state_dict(best_state)  ← 恢复最优权重
        ├─ model.eval()
        ├─ plot_curves()                ← 画 loss/acc 双图
        └─ return history
```

---

## 8. 必须记住的 5 个坑

1. **`zero_grad()` 不能省**：PyTorch 梯度默认是累加的，不清零等于训练错了。
2. **`best_state` 必须深拷贝**：`state_dict()` 是"视图"不是副本，不 `deepcopy` 最优权重会被后续更新覆盖。
3. **`min_delta` 用 `+`/`-` 而不是直接比较**：防止验证指标在噪声上"假提升"。
4. **`patience is not None` 别写成 `if patience`**：`0` 是合法值但会被当成假。
5. **`plt.show()` 会阻塞**：批量/无界面环境用 `fit(..., plot=False)` 或 `plot_curves(show=False)`。

---

## 9. 想练手的话

改这几处就能检验自己是否真的看懂了：

1. 让 `plot_curves` 支持**单独指定**保存路径（现在参数位置有点隐蔽：`plot_curves(save_path=...)`）。
2. 在 `history` 里额外记录 `lr`，并在学习曲线里加第三个子图。
3. 把 `_run_epoch` 的准确率统计改成支持**多标签分类**（提示：用 `sigmoid > 0.5` 再比较）。
4. 给 `fit` 加一个 `scheduler` 参数，每轮 `scheduler.step()`，观察学习率衰减的效果。
5. 大模型场景下 `best_state` 改为"只记最好的一轮，磁盘保存"，避免内存翻倍——提示：训练结束后从磁盘 `torch.load`。
