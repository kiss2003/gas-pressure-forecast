import torch
import torch.nn as nn
from net import LSTM_Day  # 确保 net.py 中定义了 LSTM_Day

# 1. 设备选择
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2. 加载带 “_diff” 后缀的新模型
model = LSTM_Day(dim1=50, dim2=10, num_layers=2).to(device)
checkpoint = torch.load('LSTM_Day64_10_slide_cut6ms_diff.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 3. 定义一个显式返回隐状态的包装器，直接用 (B, T, 1) 作为 lstm 输入
class ExportWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, x, h0, c0):
        # x: (B, T, I)  这里 I 应该等于 1
        # h0, c0: (num_layers, B, hidden_size)
        out_seq, (hn, cn) = self.model.lstm(x, (h0, c0))
        # 取最后一个时间步的隐藏状态
        last = out_seq[:, -1, :]                 # (B, dim1)
        y = torch.tanh(self.model.fc1(last))     # (B, dim2)
        y = y.unsqueeze(1)                       # (B, 1, dim2)
        y = self.model.fc2(y).squeeze(-1)        # (B,)
        return y, hn, cn

# 4. 包装并移动到 device
wrapped = ExportWrapper(model).to(device)

# 5. 构造示例输入及初始状态
B, T, I = 1, 8641, 1    # batch=1, seq_len=8641, input_dim=1
L, H = 2, 50            # num_layers=2, hidden_size=64
dummy_x = torch.randn(B, T, I, device=device)
h0 = torch.zeros(L, B, H, device=device)
c0 = torch.zeros(L, B, H, device=device)

# 6. 导出为 ONNX
torch.onnx.export(
    wrapped,
    (dummy_x, h0, c0),
    "lstm_day_diff.onnx",
    input_names=['input', 'h0', 'c0'],
    output_names=['output', 'hn', 'cn'],
    opset_version=11,
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'seq_len'},
        'h0':    {1: 'batch_size'},
        'c0':    {1: 'batch_size'},
        'output': {0: 'batch_size'},
    }
)

print("ONNX 模型导出成功：lstm_day_diff.onnx")
