import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import data
import models as m
from torch.nn import functional as F
import tqdm
import time
import tools

# 定义一些超参数
input_size = 10  # 输入数据的维度
d_model = 26  # 模型的维度
num_heads = 2    # 多头注意力机制的头数
num_encoder_layers = 2  # 编码器层的数量
num_decoder_layers = 2  # 解码器层的数量
dropout_rate = 0.1  # Dropout比率
num_epochs = 5000 # 训练的轮数
lr = 0.00001 #学习率
focal_weight = 0.90
model_dir = "models"
train_x_path = '../data/train_add_norm.dat'
train_y_path = '../data/train_label.dat'
test_x_path = '../data/test_add_norm.dat'
test_y_path = '../data/test_label.dat'
X, Y = data.DataReader(train_x_path, train_y_path).read_data()
X_test, Y_test = data.DataReader(test_x_path, test_y_path).read_data()
# X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, test_size=0.0, random_state=42)
X_train, Y_train = X, Y
# 转换为Tensor并进行padding
# X_train_tensor = model.pad_sequences(X_train)
# Y_train_tensor = model.pad_sequences(Y_train, padding_value=-100)
# X_test_tensor = model.pad_sequences(X_test)
# Y_test_tensor = model.pad_sequences(Y_test, padding_value=-100)

# X_train_tensor = [torch.tensor(x).float() for x in X_train]
# Y_train_tensor = [torch.tensor(x).long() for x in Y_train]

# X_train_tensor = (X_train_tensor * 5).int()
# X_test_tensor = (X_test_tensor * 5).int()

# 创建DataLoader
# train_data = TensorDataset(X_train_tensor, Y_train_tensor)
# # test_data = TensorDataset(X_test_tensor, Y_test_tensor)
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def make_name(dir, lr, epoch, loss_name="ce_loss", weight=None):
    if weight is None:
        weight = 0
    name = f"{dir}/{time.strftime('%Y%m%d%H%M%S')}-{lr}-{epoch}-{loss_name}-{weight}.bin"
    return name
def train(x, y, x_test=None, y_test=None, model=None, fname = "..\models\save_model1.bin"):
    if model is None:
        model = m.TransformerModel(d_model=d_model,
                                   head=num_heads,
                                   num_encoder_layers=num_encoder_layers,
                                   num_decoder_layers=num_decoder_layers,
                                   dropout_rate=dropout_rate).to(device)
        model.init_weights()
    # print(model)
    weights = torch.tensor([1.0, 20.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)#ignore_index=-100)
    loss_fn = tools.GHMC_Loss(30, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        count = len(x)
        qbar = tqdm.tqdm(range(count), leave=True, position=0)
        for i in qbar:
            x_batch, y_batch = torch.tensor(x[i]).float(), torch.tensor(y[i]).long()
            x_batch = x_batch.unsqueeze(0)
            x_batch = torch.where(torch.isnan(x_batch), torch.tensor(0).float(), x_batch)
            # Y_batch = Y_batch.unsqueeze(0)
            # print(X_batch, Y_batch)
            # print(X_batch.shape, Y_batch.shape)
            optimizer.zero_grad()
            # 假设我们的模型可以直接处理batched data
            # src_mask = model.generate_square_subsequent_mask(X_batch.size(0)).to(device)
            output = model(x_batch.to(device), src_mask=None)
            output = output.squeeze(0)
            # print(i, output, Y_batch)
            # loss = loss_fn(output, y_batch.to(device))
            loss = tools.focal_loss(output, y_batch.to(device), gamma=1.5, alpha=focal_weight)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 1 == 0:
                qbar.set_description(f"epoch:{epoch},data_shape:{x_batch.shape}, loss:{loss.item()}")
        if epoch % 10 == 0:
            if x_test is not None and y_test is not None:
                eval(x_test, y_test, model=model)
    torch.save(model, fname)

def eval(x, y, model = None, fname = None):
    if model is None:
        model = torch.load(fname)
    model.eval()
    count = len(x)
    total_acc, total_sp, total_sn, total_mcc = 0, 0, 0, 0
    qbar = tqdm.tqdm(range(count), leave=True, position=0)
    for i in qbar:
        x_batch, y_batch = torch.tensor(x[i]).float(), torch.tensor(y[i]).long()
        x_batch = x_batch.unsqueeze(0)
        x_batch = torch.where(torch.isnan(x_batch), torch.tensor(0).float(), x_batch)
        output = model(x_batch.to(device), src_mask=None)
        output = output.squeeze(0)
        output = torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        # prob = torch.sum(torch.eq(output, y_batch.to(device))).cpu().item() / output.size(0)
        acc, sn, sp, mcc = tools.ROC(output.cpu().tolist(), y_batch.tolist())
        total_acc += acc
        total_sp += sp
        total_sn += sn
        total_mcc += mcc
        if i % 1 == 0:
            qbar.set_description(f"prob:{acc}, sn:{sn}, sp:{sp}, mcc:{mcc}")
    print(f"\nfinal acc:{total_acc/count}, final sp:{total_sp/count}, final sn:{total_sn/count}, final mcc:{total_mcc/count}")

if __name__ == '__main__':
    fname = make_name(model_dir, lr, num_epochs, "focal_loss", weight=focal_weight)
    # model = torch.load("models/20240309100932-1e-05-5000-focal_loss-0.05.bin")
    model = None
    train(X_train, Y_train, X_test, Y_test, model, fname)
    # fname = model_dir + "/0.001-100-20240301165035.bin"
    # eval(X_eval, Y_eval, fname)
    eval(X_test, Y_test, fname = fname)
    eval(X_train, Y_train, fname = fname)



