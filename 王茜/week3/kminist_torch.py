import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def load_dataset(device, batch_size=100, shuffle=True):
    # 加载训练集和测试集
    train_data = KMNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_data = KMNIST(root='./test_data', train=False, download=True, transform=ToTensor())
    labels = train_data.class_to_idx
    train_size = train_data[0][0].shape
    train_data = [(x.reshape(-1).to(device), torch.tensor(y).to(device)) for x, y in train_data]
    test_data = [(x.reshape(-1).to(device), torch.tensor(y).to(device)) for x, y in test_data]
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return train_dl, test_dl, labels, train_size



def select_device(device='cpu'):
    if device=='mps' and torch.mps.is_available():
        return torch.device('mps')
    if device=='cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def torch_model(in_linear_features,
                in_linear_out_features,
                out_linear_features,
                out_linear_out_features,
                lr,
                device):
    # 所有结构串联
    model = nn.Sequential(
        nn.Linear(in_linear_features, in_linear_out_features).to(device),
        nn.Sigmoid().to(device),
        nn.Linear(out_linear_features, out_linear_out_features).to(device),
        nn.Sigmoid().to(device)
    )

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
    # torch.optim # 优化器库
    # 只需传入模型参数和学习率即可
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_fn, optimizer


def train_model(model, loss_fn, optimizer, train_dl, epochs, model_path):
    loss_history = []
    for i in range(epochs):
        t1 = time.time()
        for batch_data, batch_target in train_dl:
            # 前向运算
            output = model(batch_data)
            # 计算损失
            loss = loss_fn(output, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time.time()
        print(f'Epoch [{i + 1}/{epochs}], Loss: {loss.item():.4f}, Time: {t2 - t1:.5f}')
        loss_history.append(loss.item())
    torch.save(model.state_dict(), f'{model_path}.pth')
    return model, loss_history


def predict_model(model, test_dl):
    # 测试
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dl:
            output = model(data)
            _, predicted = torch.max(output, 1)  # 返回每行最大值和索引
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'target: {total}, correct: {correct}, Accuracy: {correct / total * 100}%')
    precision = correct / total * 100
    return precision


def main(device_name, lr, epochs, neuron, batch_size, load_local=False):
    # 选择设备类型
    device = select_device(device_name)
    # 加载数据集
    train_dl, test_dl, labels, data_shape = load_dataset(device, batch_size=batch_size, shuffle=True)
    model_name = f'model-{neuron}-{lr}-{batch_size}-{epochs}'
    # 加载模型
    in_linear_features = train_dl.dataset[0][0].shape[0]
    in_linear_out_features = neuron
    out_linear_features = neuron
    out_linear_out_features = len(labels)
    model, loss_fn, optimizer = torch_model(in_linear_features,
                                            in_linear_out_features,
                                            out_linear_features,
                                            out_linear_out_features,
                                            lr,
                                            device)
    precision = None
    loss_history = None
    if load_local and os.path.exists(f'{model_name}.pth'):
        model.load_state_dict(torch.load(f'{model_name}.pth'))
        precision = predict_model(model, test_dl)
    else:
        model, loss_history = train_model(model, loss_fn, optimizer, train_dl, epochs, model_name)
    return loss_history, precision


if __name__ == '__main__':
    use_device = 'cpu'
    device = select_device(use_device)
    lr_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_size = 120
    epochs_list = [10, 100, 1000]
    neurons = [16, 32, 64, 128]
    lr = 0.01
    epochs = 10
    neuron = 16
    # 验证不同学习率
    for lr in lr_rates:
        epochs = 100
        neuron = 16
        main(device, lr, epochs, neuron, batch_size, load_local=True)
        print(f'model_paras: lr: {lr}, batch_size: {batch_size}, epochs: {epochs}, neuron: {neuron}')

    # 验证不同神经元个数
    for neuron in neurons:
        lr = 0.01
        epochs = 100
        main(device, lr, epochs, neuron, batch_size, load_local=True)
        print(f'model_paras: lr: {lr}, batch_size: {batch_size}, epochs: {epochs}, neuron: {neuron}')