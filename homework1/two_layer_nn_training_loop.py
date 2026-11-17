from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

EPOCH = 100
BATCH_SIZE = 64
LR = 0.1           # SGD 的步长比 Adam 大很多，0.001 会慢得几乎不动
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
SEED = 42
# 用脚本所在目录拼路径，这样不管在哪个目录下运行都能找到文件
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'wine_quality_dataset.csv'
MODEL_PATH = BASE_DIR / 'wine_quality_mlp.pth'


torch.manual_seed(SEED)
np.random.seed(SEED)

#读取数据集
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['quality']).values.astype(np.float32)
classes, Y = np.unique(df['quality'].values, return_inverse=True)
Y = Y.astype(np.int64)                                      
NUM_FEATURES = X.shape[1]
NUM_CLASSES = len(classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.fc(x)
        return x, x

def train_model(train_X, train_Y):
    mlp = MLP().to(device)
    optimizer = torch.optim.SGD(
        mlp.parameters(), 
        lr=LR, momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    loss_func = nn.CrossEntropyLoss()
    loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_X, train_Y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    mlp.train()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(loader):
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = mlp(b_x)[0]
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return mlp

def accuracy(model, X_, Y_):
    model.eval()
    with torch.no_grad():
        pred_y = model(X_)[0].argmax(dim=1)
        return (pred_y == Y_).float().mean().item()

#划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, stratify=Y, random_state=SEED
)

#只用训练集的均值/方差做标准化，避免测试集信息泄露到训练里
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

train_X = torch.from_numpy((X_train - mean) / std).to(device)
train_Y = torch.from_numpy(Y_train).to(device)
test_X = torch.from_numpy((X_test - mean) / std).to(device)
test_Y = torch.from_numpy(Y_test).to(device)

#训练并评估
print('device:', device)
mlp = train_model(train_X, train_Y)
train_acc = accuracy(mlp, train_X, train_Y)
test_acc = accuracy(mlp, test_X, test_Y)

print('train accuracy: %.2f | test accuracy: %.2f' % (train_acc, test_acc))

mlp.eval()
with torch.no_grad():
    pred_y = mlp(test_X)[0].argmax(dim=1).cpu().numpy()
print('测试集前 10 个预测分数:', classes[pred_y[:10]])
print('测试集前 10 个真实分数:', classes[Y_test[:10]])

#保存权重、标准化参数和标签映射，推理时新数据要用同样的 mean/std 处理，
torch.save({
    'state_dict': mlp.state_dict(),
    'mean': mean,
    'std': std,
    'classes': classes,
}, MODEL_PATH)
print('权重已保存:', MODEL_PATH)
