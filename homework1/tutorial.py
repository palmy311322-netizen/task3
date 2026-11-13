from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR=0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),#(0,1)
    download=DOWNLOAD_MNIST
)

#print(train_data.data.size())
#print(train_data.targets.size())
#plt.imshow(train_data.data[0].numpy(), cmap='gray')
#plt.title('%i' % train_data.targets[0])
#plt.show()

train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 有 GPU 就用 GPU，否则用 CPU
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data=torchvision.datasets.MNIST(root='./mnist/', train=False)
# 测试集不做反向传播，用 no_grad 代替旧版的 volatile
with torch.no_grad():
    # (2000, 28, 28) -> (2000, 1, 28, 28)，归一化到 [0, 1]
    test_X=test_data.data[:2000].unsqueeze(1).float()/255.
    test_Y=test_data.targets[:2000]
test_X, test_Y=test_X.to(device), test_Y.to(device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,#前面得到层高
                out_channels=16,#输出的层高
                kernel_size=5,#扫描5*5的区域
                stride=1,#步长
                padding=2#填充2个像素
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out=nn.Linear(32*7*7, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0), -1)
        output=self.out(x)
        return output, x

cnn=CNN().to(device)
optimizer=torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func=nn.CrossEntropyLoss()

def test_accuracy():
    cnn.eval()
    with torch.no_grad():
        test_output=cnn(test_X)[0]
        pred_y=test_output.argmax(dim=1)
        acc=(pred_y == test_Y).float().mean().item()
    cnn.train()
    return acc

print('device:', device)
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y=b_x.to(device), b_y.to(device)
        output=cnn(b_x)[0]        # forward 返回 (output, x)，训练只用 output
        loss=loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            acc=test_accuracy()
            print('Epoch: %d | step: %4d | train loss: %.4f | test accuracy: %.2f'
                  % (epoch, step, loss.item(), acc))

cnn.eval()
with torch.no_grad():
    pred_y=cnn(test_X)[0].argmax(dim=1)
print('Final test accuracy: %.2f' % (pred_y == test_Y).float().mean().item())
print('前 10 个预测:', pred_y[:10].cpu().numpy())
print('前 10 个真实:', test_Y[:10].cpu().numpy())

# 保存权重，供 finetune.py 在你自己的照片上继续训练
weight_path = Path(__file__).with_name('mnist_cnn.pth')
torch.save(cnn.state_dict(), weight_path)
print('权重已保存:', weight_path)