import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


class MalwareDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        pkl_path = os.path.join(self.root_dir, f"{self.annotations.iloc[index, 0]}.pkl")  # 使用os.path.join以确保路径正确
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        feature = data['features']
        label = data['class'] - 1  # 假设标签从1开始，调整为从0开始
        return feature, label


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # 移除了全局平均池化层
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= size
    correct /= size
    print(f"Train Loss: {train_loss:>8f}, Accuracy: {(100 * correct):>0.2f}%")


def ceshi(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Loss: {test_loss:>8f}, Accuracy: {(100 * correct):>0.2f}%")


csv_file = 'D:\恶意软件数据集\最新数据集\扩充9902pic_updated.csv'
root_dir = 'D:\恶意软件数据集\最新数据集\融合特征'

batch_size = 64
learning_rate = 1e-3
epochs = 100

dataset = MalwareDataset(csv_file=csv_file, root_dir=root_dir)

train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 构建模型
model = NeuralNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和测试模型
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    ceshi(test_loader, model, loss_fn)
print("Done!")
