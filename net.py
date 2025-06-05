import torch
import torch.nn as nn
import math
#
# """
# LSTM6网络 最后的训练和测试误差
# Epoch [47/100], Train Loss: 0.0087,Train MAE Loss: 0.0061
# Epoch [47/100], Test Loss: 0.0082, Test MAE Loss: 0.0055
# """

class LSTM_Day(nn.Module):
    """根据过去数天数据预测1天"""
    def __init__(self, dim1=10, dim2=6, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, dim1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, 1)

    def forward(self, x, hx=None):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out_seq, (h_n, c_n) = self.lstm(x, hx)
        # 只取最后一个时间步
        last = out_seq[:, -1, :]                  # (batch, dim1)
        y = torch.tanh(self.fc1(last))            # (batch, dim2)
        y = y.unsqueeze(1)
        y = self.fc2(y).squeeze(-1)               # (batch,)
        return y, (h_n, c_n)


class LSTM_Day_step(nn.Module):
    """根据过去数天数据预测1天，支持隐藏状态复用"""
    def __init__(self, dim1=10, dim2=6, num_layers=2):
        super().__init__()
        # 1→hidden_size, 多层，batch_first
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=dim1,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(dim1, dim2)
        self.fc2 = nn.Linear(dim2, 1)

    def forward(self, x, hx=None):
        # 扩维到 (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        # LSTM 前向，若 hx=None，内部自动初始化为零
        out_seq, (h_n, c_n) = self.lstm(x, hx)

        # 只取最后一个时间步
        last = out_seq[:, -1, :]                  # (batch, dim1)
        y = torch.tanh(self.fc1(last))            # (batch, dim2)
        y = self.fc2(y).squeeze(-1)               # (batch,)
        return y, (h_n, c_n)
        
"""Attention-LSTM6"""
# Attention机制实现

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):
        attn_weights = torch.tanh(self.attn(hidden_states))
        attn_weights = self.context(attn_weights).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(2) * hidden_states, dim=1)
        return context_vector, attn_weights

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1, n_layers=2):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        context_vector, attn_weights = self.attention(out)
        out = self.fc(context_vector)
        return out, attn_weights


class LSTM6_hyb(nn.Module):
    """定义 LSTM 模型"""
    def __init__(self, hidden_size=25, num_layers=2):
        super(LSTM6_hyb, self).__init__()
        self.lstm = nn.LSTM(6, hidden_size, num_layers, batch_first=True)
        self.lstm_ = nn.LSTM(1, 10, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size+10, 15)
        self.fc2 = nn.Linear(15,1)
    def forward(self, x):
        x = x.unsqueeze(-1)
        out_, c = self.lstm_(x)
        out_ = out_[:, -1440:-1, :]
        t = x.view(-1, 6, 1440).transpose(1, 2)
        out, _ = self.lstm(t)
        out = out[:, -1440:-1, :]
        combined_feature = torch.cat([out_, out], dim=2)
        res = self.fc1(combined_feature)
        res = torch.tanh(res)
        res = self.fc2(res)
        res = res.squeeze(-1)
        return res

class LSTM11_hybrid(nn.Module):
    """定义 LSTM 模型"""
    def __init__(self, hidden_size=32, num_layers=2):
        super(LSTM11_hybrid, self).__init__()
        self.lstm = nn.LSTM(11, hidden_size, num_layers, batch_first=True)
        self.lstm_ = nn.LSTM(1, 10 , num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size+10, 20)
        self.fc2 = nn.Linear(20,1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out_, c = self.lstm_(x)
        out_ = out_[:, -1440:-1, :]
        t = x.view(-1,11,1440).transpose(1,2)
        out, _ = self.lstm(t)
        out = out[:, -1440:-1, :]
        combined_feature = torch.cat([out_,out],dim=2)
        print(combined_feature.shape)
        res = self.fc1(combined_feature)
        res = torch.tanh(res)
        res = self.fc2(res)
        res = res.squeeze(-1)
        return res

class LSTM11(nn.Module):
    """定义 LSTM 模型"""
    def __init__(self, hidden_size=32, num_layers=2):
        super(LSTM11, self).__init__()
        self.lstm = nn.LSTM(11, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16,1)

    def forward(self, x):
        x = x.view(-1,11,1440).transpose(1,2)
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1440:-1, :])
        out = torch.tanh(out)
        out = self.fc2(out)
        out = out.squeeze(-1)
        return out

class LSTM11_(nn.Module):
    """定义 LSTM 模型"""
    def __init__(self, hidden_size=10, num_layers=2):
        super(LSTM11_, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 6)
        self.fc2 = nn.Linear(6,1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1440:-1, :])
        out = torch.tanh(out)
        out = self.fc2(out)
        out = out.squeeze(-1)
        return out


# """""Transformer 模型"""""
class TransformerModel(nn.Module):

    def __init__(self, hidden_size = 24, num_heads=2, num_layers=2, dropout=0.1, embed_dim=32):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        # 输入嵌入层，将输入维度映射到隐藏维度
        self.input_embedding = nn.Linear(1 + embed_dim, hidden_size)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 6)
        self.fc2 = nn.Linear(6, 1)

    def get_temporal_embedding(self, seq_length, device):
        """
        生成时间嵌入（正弦和余弦函数）
        """
        position = (torch.arange(seq_length, device=device) % 1400).unsqueeze(1)  # (seq_length, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device) *
                             (-torch.log(torch.tensor(10000.0, device=device)) / self.embed_dim))
        pe = torch.zeros(seq_length, self.embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_length, embed_dim)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)

        # 生成时间嵌入
        temporal_embedding = self.get_temporal_embedding(seq_length, x.device)

        # 将时间嵌入与输入特征拼接
        x = x.unsqueeze(-1) # (batch_size, seq_length, 1)
        temporal_embedding = temporal_embedding.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, seq_length, embed_dim)
        x = torch.cat((x, temporal_embedding), dim=-1)  # (batch_size, seq_length, 1 + embed_dim)

        # 输入嵌入层
        x = self.input_embedding(x)  # (batch_size, seq_length, hidden_size)

        # Transformer 编码器要求输入的形状为 (seq_len, batch_size, hidden_size)
        x = x.permute(1, 0, 2)

        # Transformer 编码器
        out = self.transformer_encoder(x)

        # 取最后一时刻的输出
        out = out.permute(1, 0, 2)  # 转换回 (batch_size, seq_len, hidden_size)
        out = out[:, -1440:, :]  # 取最后 1440 个时间步的输出

        # 输出层
        out = self.fc1(out)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = out.squeeze(-1)  # 去掉最后一维
        return out

class CNN2_1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 15), 1, (0, 7))
        self.pool1 = nn.AvgPool2d((1, 2), (1, 2))
        self.f1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(16, 24, (3, 7), 1, (0, 3))
        self.pool2 = nn.AvgPool2d((1, 2), (1, 2))
        self.f2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(24, 36, (3, 7), 1, (0, 3))
        self.pool3 = nn.AvgPool2d((1, 2), (1, 2))
        self.f3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(36, 48, (3, 7), 1, (0, 3))
        self.pool4 = nn.AvgPool2d((1, 2), (1, 2))
        self.f4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(48, 64, (3, 7), 1, (0, 3))
        self.pool5 = nn.AvgPool2d((1, 2), (1, 2))
        self.f5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(64, 80, (3, 7), 1, 0)
        self.pool6 = nn.AvgPool2d((1, 2), (1, 2))
        self.f6 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(80, 100, (3, 7), 1, 0)
        self.pool7 = nn.AvgPool2d((1, 2), (1, 2))
        self.f7 = nn.LeakyReLU()
        self.conv8 = nn.Conv2d(100, 120, (3, 7), 1, 0)
        self.pool8 = nn.AvgPool2d((1, 2), (2, 2))
        self.f8 = nn.LeakyReLU()
        self.fc1 = nn.Linear(2400,240)
        self.fc2 = nn.Linear(240,24)
        self.fc3 = nn.Linear(24,1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.unfold(2, 3 * 1440, 1440)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.f1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.f2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.f3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.f4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.f5(x)
        x = x[:,:,:,45:90]   #选取中间一部分
        x = self.conv6(x)
        x = self.pool6(x)
        x = self.f6(x)
        x = self.conv7(x)
        x = self.pool7(x)
        x = self.f7(x)
        # x = self.conv8(x)
        # x = self.pool8(x)
        # x = self.f8(x)
        # print("f8.shape", x.shape)
        # x = torch.squeeze(-2)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    x = torch.ones(2,8640)
    module = LSTM_Day()
    out,c = module(x)
    print(out.shape)