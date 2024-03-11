import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# 定义一个数据集类，用于加载和处理数据
class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# 定义模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)  # 简单的线性层作为嵌入层
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, num_classes)

    def forward(self, src, src_key_padding_mask):
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.mean(dim=0)  # 使用平均池化来聚合序列信息
        output = self.output_layer(output)
        return output


# 准备数据
def prepare_data(X, Y, batch_size):
    # 计算最大序列长度
    max_length = max([len(x) for x in X])
    # Padding X
    X_padded = [np.pad(x, (0, max_length - len(x)), 'constant', constant_values=0) for x in X]
    # 转换为Tensor
    X_tensor = torch.tensor(X_padded, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    # 创建数据集和数据加载器
    dataset = SequenceDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 训练函数
def train(model, dataloader, epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # 创建注意力掩码
            src_key_padding_mask = X_batch == 0
            optimizer.zero_grad()
            outputs = model(X_batch, src_key_padding_mask)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# 假设X和Y是你的数据
def generate_mock_data(num_sequences, sequence_length=26):
    """
    生成模拟数据。

    参数:
    num_sequences (int): 生成序列的数量。
    sequence_length (int): X中每个序列的长度。

    返回:
    X (list of numpy.ndarray): 输入序列的列表。
    Y (numpy.ndarray): 目标分类标签。
    """
    X = []
    Y = np.random.randint(2, size=(num_sequences,))

    for _ in range(num_sequences):
        # 生成一个长度为sequence_length的序列，每个元素是0到1之间的随机浮点数
        sequence = np.random.rand(sequence_length)
        X.append(sequence)

    return X, Y


# 生成模拟数据
num_sequences = 1000  # 假设我们想生成1000个序列
X, Y = generate_mock_data(num_sequences)
print(len(X), len(Y))
# 参数设置
input_dim = 26  # 假设每个时间步的特征维度是26
num_heads = 2
num_layers = 2
num_classes = 2
batch_size = 32
epochs = 10

# 数据准备和模型初始化
dataloader = prepare_data(X, Y, batch_size)
model = TransformerClassifier(input_dim, num_heads, num_layers, num_classes)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练模型
train(model, dataloader, epochs, device)
