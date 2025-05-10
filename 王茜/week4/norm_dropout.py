"""
1. 搭建的神经网络，使用olivettiface数据集进行训练。
2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
3. 尝试不同optimizer对模型进行训练，观察对比loss结果。
4. 注册kaggle并尝试激活Accelerator，使用GPU加速模型训练。# 无需提交
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ---------------------- 自定义 Dataset 类 ----------------------
class OlivettiDataset(Dataset):
    def __init__(self, transform=None):
        # 加载 Olivetti Faces 数据集
        self.data = fetch_olivetti_faces(shuffle=True, random_state=42)
        self.images = self.data.data  # 转换为 [C, H, W]
        self.targets = self.data.target
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


class Net0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 512)
        # self.norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 216)
        # self.norm2 = nn.BatchNorm1d(216)
        self.fc3 = nn.Linear(216, 40)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.norm1(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.norm2(x)
        x = self.act(x)
        # x = self.droput(x)
        x = self.fc3(x)
        return x


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 512)
        self.norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 216)
        self.norm2 = nn.BatchNorm1d(216)
        self.fc3 = nn.Linear(216, 40)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(model, batch_data, epochs, loss_fn, optimizer):
    model.train()
    loss_history = []
    for i in tqdm.tqdm(range(epochs), total=epochs):
        t1 = time.time()
        for train, target in batch_data:
            output = model(train)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time.time()
        print(f'Epoch [{i + 1}/{epochs}], Loss: {loss.item():.4f}, Time: {t2 - t1:.5f}')
        loss_history.append(loss.item())
    # 模型训练和测试
    return loss_history


def predict(model, test_dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dl:
            output = model(data)
            _, predicted = torch.max(output, -1)  # 返回每行最大值和索引
            total += data.shape[0]
            correct += (predicted == target).sum().item()
    print(f'target: {total}, correct: {correct}, Accuracy: {correct / total * 100}%')
    precision = correct / total * 100
    return precision

def plot_loss(loss_history_df):
    loss_history_df.plot()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss history')
    plt.show()
    plt.close()


if __name__ == '__main__':
    device = 'cpu'
    batch_size = 10
    lr0 = 0.01
    lr1 = 0.01
    epochs = 100
    test_size = 0.2

    olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    data = OlivettiDataset()
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    print(type(data))
    train_batch = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_batch = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model0 = Net0().to(device)
    model1 = Net1().to(device)
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
    # torch.optim # 优化器库
    # 只需传入模型参数和学习率即可
    optimize0 = torch.optim.SGD(model0.parameters(), lr=lr0)
    optimize1 = torch.optim.SGD(model1.parameters(), lr=lr1)

    loss_history_dict = {}
    # 训练
    loss0 = train_model(model0, train_batch, epochs, loss_fn, optimize0)
    loss1 = train_model(model1, train_batch, epochs, loss_fn, optimize1)
    # 绘制损失函数对比图像
    loss_history_dict['no_norm_dropout'] = loss0
    loss_history_dict['norm_dropout'] = loss1
    loss_history_df = pd.DataFrame(loss_history_dict)
    plot_loss(loss_history_df)

    # 预测
    print('未标准化和正则化')
    predict(model0, test_batch)
    print('正则化和标准化后')
    predict(model1, test_batch)



