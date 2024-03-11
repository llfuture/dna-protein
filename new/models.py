import torch
import torch.nn as nn
import math

from torch.autograd import Variable
from new.data import DataReader
import torch.nn.functional as F
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一些超参数
input_size = 10  # 输入数据的维度
model_dim = 26  # 模型的维度
num_heads = 8    # 多头注意力机制的头数
num_encoder_layers = 6  # 编码器层的数量
num_decoder_layers = 6  # 解码器层的数量
dropout_rate = 0.1  # Dropout比率
num_epochs = 100 # 训练的轮数
def pad_sequences(sequences, padding_value=None):

    """
    对序列进行padding，使它们的长度相同。

    参数:
    sequences: 一个序列的列表，每个序列是一个列表。
    padding_value: 用于填充的值。

    返回:
    一个tensor，其中包含了padding后的序列。
    """
    # 找到最长序列的长度
    if padding_value is None:
        return torch.tensor(sequences, dtype=torch.float)
    else:
        max_len = max(len(seq) for seq in sequences)

        # 初始化一个列表，用于存储padding后的序列
        padded_sequences = []

        # 对每个序列进行padding
        for seq in sequences:
            padded_seq = seq + [padding_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)

        return torch.tensor(padded_sequences, dtype=torch.float)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=26, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return self.dropout(x)
class TransformerModel(nn.Module):
    def __init__(self, d_model, head, num_encoder_layers, num_decoder_layers, dropout_rate):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        # self.emb = nn.Linear(26, 128)
        self.pos_encoder = PositionalEncoding(model_dim, dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=head, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 26)
        self.norm2 = nn.LayerNorm(26)
        self.decoder = nn.Linear(26, 2)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask):
        # x = self.emb(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_mask)
        output = self.norm1(output)
        output = self.linear(output)
        output = F.relu(output)
        output = self.norm2(output)
        output = self.decoder(output)

        return output

# if __name__ == '__main__':
    # # 假设X_train和Y_train是训练数据和标签
    # # 首先，对X_train和Y_train进行padding
    # feature_path = '../data/train_add_norm.dat'
    # label_path = '../data/train_label.dat'
    #
    # # 创建DataReader实例
    # data_reader = DataReader(feature_path, label_path)
    #
    # # 读取数据
    # X_train, Y_train = data_reader.read_data()
    # X_train_padded = pad_sequences(X_train)
    # Y_train_padded = pad_sequences(Y_train)  # 对于标签，使用一个特殊值进行padding
    #
    # # 创建模型实例
    # model = TransformerModel(d_model=model_dim, head=num_heads,
    #                          num_encoder_layers=num_encoder_layers,
    #                          num_decoder_layers=num_decoder_layers,
    #                          dropout_rate=dropout_rate).to(device)
    #
    # # 定义损失函数和优化器
    # loss_fn = nn.CrossEntropyLoss()  # 忽略padding值的损失
    # optimizer = torch.optim.Adam(model.parameters())
    #
    # # 简化的训练循环
    # for epoch in range(num_epochs):
    #     model.train()
    #     optimizer.zero_grad()
    #
    #     # 计算mask
    #     src_mask = model.generate_square_subsequent_mask(X_train_padded.size(0)).to(device)
    #
    #     output = model(X_train_padded.to(device), None)
    #     loss = loss_fn(output.view(-1, output.size(-1)), Y_train_padded.view(-1))
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
