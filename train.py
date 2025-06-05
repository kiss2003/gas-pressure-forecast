import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from math import sqrt
import os
from net import *

# 设置训练参数
batch_size = 256  # 批次大小
num_epochs = 100  # 训练轮数
learn_rate = 0.001 # 学习率
normal = False #数据是否需要归一化
# 设置模型参数
dims = 50,10  # 隐藏层维度
seq_length = 6 * 1440  # 序列长度
num_layers = 2  # LSTM层数

loss_file_name = 'LSTM_Day64_10_slide_cut6ms_diff.txt'

# 指定使用第一块 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 初始化模型，并将其移动到 GPU
torch.set_default_dtype(torch.float32)
model = LSTM_Day(*dims, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learn_rate, foreach=False)  # 使用Adam优化器

# 加载数据
data = np.load("slide_cut6ms_train.npz")
train_x = data['day6']
print('训练集大小：',train_x.shape)
#随机打乱顺序
idx = np.arange(train_x.shape[0])
np.random.shuffle(idx)
train_x = train_x[idx]
if normal:
    train_mean = data['mean'].astype('float32')
    train_mean = train_mean[idx]
    train_std = data['std'].astype('float32')
    train_std = train_std[idx]
    train_x -= np.expand_dims(train_mean,-1) #分两步，避免内存不够
    train_x /= np.expand_dims(train_std,-1)
    
data = np.load("slide_cut6ms_test.npz")
test_x = data['day6']
print('测试集大小：',test_x.shape)
if normal:
    test_mean = data['mean'].astype('float32')
    test_std = data['std'].astype('float32')
    test_x = (test_x-np.expand_dims(test_mean,-1))/np.expand_dims(test_std,-1)
print('数据集加载完毕')

loss_file = open(loss_file_name,'w')
loss_file.write(open(os.path.split(__file__)[-1],encoding='utf-8').read()+'\n')

# 训练模型
min_loss = float('inf')
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0.

    # 训练过程
    for i in range(train_x.shape[0]//batch_size):
        x = torch.tensor(train_x[i*batch_size:(i+1)*batch_size]).to(device)
        if normal:
            std_batch = torch.tensor(train_std[i*batch_size:(i+1)*batch_size]).to(device)
            outputs = model(x)
            outputs = outputs * std_batch.unsqueeze(-1)
        else:
            x = (x - x[:,0].unsqueeze(-1))[:,1:]
            outputs = model(x)
        loss = criterion(outputs, x[:, -1439:]) # 最后 1439 个时间步作为标签
        total_loss += sqrt(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print(f"batch={i}, Train Loss: {total_loss/(i+1):.4f}")

    avg_loss = total_loss / (i+1)
    info = f"Epoch=[{epoch + 1}/{num_epochs}], Train loss={avg_loss:.4f}, "
    print(info, end='')
    
    #测试
    model.eval()
    total_loss = 0.
    for i in range(test_x.shape[0]//batch_size):
        x = torch.tensor(test_x[i*batch_size:(i+1)*batch_size]).to(device)  
        if normal:
            std_batch = torch.tensor(test_std[i*batch_size:(i+1)*batch_size]).to(device)
            outputs = model(x)
            outputs = outputs * std_batch.unsqueeze(-1)
        else:
            x = (x - x[:,0].unsqueeze(-1))[:,:-1]
            outputs = model(x)
        loss = criterion(outputs, x[:, -1439:]) # 最后 1439 个时间步作为标签
        total_loss += sqrt(loss.item())   
    avg_loss = total_loss / (i+1)
    info2 = f"Test loss={avg_loss:.4f}"
    print(info2)
    loss_file.write(info+info2+'\n')
    loss_file.flush()
    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model.state_dict(), loss_file_name[:-4]+".pth") # 保存模型权重