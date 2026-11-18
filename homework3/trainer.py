import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# 学习曲线配色：蓝/橙
COLOR_TRAIN = '#2a78d6'
COLOR_VAL = '#eb6834'
COLOR_SURFACE = '#fcfcfb'
COLOR_GRID = '#e1e0d9'
COLOR_AXIS = '#c3c2b7'
COLOR_TEXT = '#52514e'
COLOR_MUTED = '#898781'

class Trainer:
    def __init__(self, model, optimizer, criterion, device=None,
                 save_path='best_model.pth', patience=10, min_delta=0.0,
                 monitor='val_acc', plot_path='learning_curves.png'):
        
        if monitor not in ('val_acc', 'val_loss'):
            raise ValueError("monitor 只能是 'val_acc' 或 'val_loss'")

        self.model = model
        #model: 任意 nn.Module
        self.optimizer = optimizer
        #optimizer: 任意 torch.optim.Optimizer
        self.criterion = criterion
        #criterion: 任意损失函数，调用方式
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        #device: torch.device，默认自动选择 GPU/CPU
        self.model.to(self.device)

        self.save_path = Path(save_path)
        #save_path: 最优模型的保存路径
        self.patience = patience
        #patience: 早停轮数；验证集指标连续这么多轮没提升就停止，None 表示不早停
        self.min_delta = min_delta
        #min_delta: 指标至少提升这么多才算“有提升”，避免在噪声上来回触发
        self.monitor = monitor
        #monitor: 用来挑选最优模型的验证集指标，'val_acc'（越大越好）或 'val_loss'（越小越好）
        self.mode = 'max' if monitor == 'val_acc' else 'min'
        self.plot_path = Path(plot_path)
        #plot_path: 学习曲线图片的保存路径

        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        }
        init_best = -float('inf') if self.mode == 'max' else float('inf')
        self.best_metric = init_best
        self.best_epoch = 0
        self.best_state = None
        self.no_improve = 0

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.enable_grad() if train else torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = self.criterion(output, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (output.argmax(dim=1) == y).sum().item()
                total_samples += batch_size

        return total_loss / total_samples, total_correct / total_samples

    def _is_better(self, metric):
        if self.mode == 'max':
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta

    def fit(self, train_loader, val_loader, epochs, verbose=True, plot=True):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            metric = val_acc if self.monitor == 'val_acc' else val_loss
            improved = self._is_better(metric)

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
            else:
                self.no_improve += 1

            if verbose:
                flag = ' *' if improved else ''
                print('Epoch %3d | train loss %.4f acc %.2f%% | '
                      'val loss %.4f acc %.2f%%%s'
                      % (epoch, train_loss, train_acc * 100,
                         val_loss, val_acc * 100, flag))

            if self.patience is not None and self.no_improve >= self.patience:
                if verbose:
                    print('早停：验证集 %s 连续 %d 轮未提升，第 %d 轮停止'
                          % (self.monitor, self.patience, epoch))
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.eval()
            if verbose:
                print('已恢复第 %d 轮的最优权重（%s = %.4f）'
                      % (self.best_epoch, self.monitor, self.best_metric))
                print('最优模型已保存到:', self.save_path)

        if plot:
            self.plot_curves()

        return self.history

    def plot_curves(self, save_path=None, show=True):
        save_path = Path(save_path or self.plot_path)
        epochs = list(range(1, len(self.history['train_loss']) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        fig.patch.set_facecolor(COLOR_SURFACE)

        panels = [
            (axes[0], 'Loss',
             self.history['train_loss'], self.history['val_loss']),
            (axes[1], 'Accuracy (%)',
             [v * 100 for v in self.history['train_acc']],
             [v * 100 for v in self.history['val_acc']]),
        ]

        for ax, title, train_values, val_values in panels:
            ax.set_facecolor(COLOR_SURFACE)
            ax.plot(epochs, train_values, color=COLOR_TRAIN,
                    linewidth=2, label='Train')
            ax.plot(epochs, val_values, color=COLOR_VAL,
                    linewidth=2, label='Validation')

            if self.best_epoch > 0:
                ax.axvline(self.best_epoch, color=COLOR_MUTED,
                           linestyle=':', linewidth=1)

            ax.set_title(title, fontsize=12, color=COLOR_TEXT, loc='left')
            ax.set_xlabel('Epoch', fontsize=10, color=COLOR_TEXT)
            ax.grid(True, color=COLOR_GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(COLOR_AXIS)
            ax.tick_params(colors=COLOR_MUTED, labelsize=9)
            ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_TEXT)

        axes[0].annotate('best epoch %d' % self.best_epoch,
                         xy=(self.best_epoch, 1), xycoords=('data', 'axes fraction'),
                         xytext=(4, -12), textcoords='offset points',
                         fontsize=9, color=COLOR_MUTED, va='top')

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print('学习曲线已保存到:', save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)
