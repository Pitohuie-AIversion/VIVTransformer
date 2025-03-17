import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
# 设置环境变量以解决多线程问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保结果文件夹存在
result_folder = "visualization_results"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


def plot_comparison_figure(input_pressure, true_pressure, predicted_pressure, time_step, epoch, idx, mode='train'):
    plt.figure(figsize=(24, 8))

    # 输入压力场
    plt.subplot(1, 3, 1)
    plt.imshow(input_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Input  Pressure Matrix at t={time_step:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')

    # 真实压力场
    plt.subplot(1, 3, 2)
    plt.imshow(true_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"True  Pressure Matrix at t={time_step:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')

    # 预测压力场
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Predicted  Pressure Matrix at t={time_step:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')

    # 保存并关闭图形
    save_path = os.path.join(result_folder, f"{mode}_epoch_{epoch}_sample_{idx + 1}.png")
    plt.savefig(save_path)
    plt.close()


# 数据集类，读取预处理好的 .pt 文件
class PressureDataset(Dataset):
    def __init__(self, merged_file_path):
        # 加载合并后的 .pt 文件
        self.data = torch.load(merged_file_path)

        # 使用预处理好的in_pressure和pressure数据
        self.in_pressures = self.data['in_pressure']  # 形状 [reynolds, time_steps, 20, 20]
        self.pressures = self.data['pressure']  # 形状 [reynolds, time_steps, 200, 200]
        # 将 time_steps 转换为 NumPy 数组
        self.time_steps = np.array(self.data['time_steps'])  # 所有时间步

        # 确保 time_steps 是二维数组
        if len(self.time_steps.shape) == 1:
            # 如果是单个时间步数组，则复制到所有 Reynolds 数下
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

            # 添加调试信息
        print(f"time_steps shape: {self.time_steps.shape}")
        print(f"time_steps type: {type(self.time_steps)}")

        # 计算总样本数
        self.num_samples = len(self.in_pressures) * len(self.in_pressures[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 计算Reynolds索引和时间步索引
        reynolds_idx = idx // len(self.in_pressures[0])
        time_step_idx = idx % len(self.in_pressures[0])

        # 确保索引有效
        if reynolds_idx >= len(self.in_pressures) or time_step_idx >= len(self.in_pressures[0]):
            raise IndexError("Index out of bounds")

        # 获取数据
        in_pressures = self.in_pressures[reynolds_idx, time_step_idx]  # [20, 20]
        pressure = self.pressures[reynolds_idx, time_step_idx]  # [200, 200]

        # 确保 time_steps 的访问是有效的
        if isinstance(self.time_steps[reynolds_idx], (list, np.ndarray)):
            time_step = self.time_steps[reynolds_idx][time_step_idx]

        else:
            raise TypeError("time_steps[reynolds_idx] is not subscriptable")

        # 将in_pressures展平为400维向量
        in_pressures_flat = in_pressures.view(-1)  # [400]

        # 将pressure展平为输出
        pressure_flat = pressure.view(-1)  # [40000]

        return in_pressures_flat, pressure_flat, time_step

    # 处理 DataLoader 以跳过 None 值


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


# Transformer 模型
class TransformerFlowReconstructionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6,
                 d_model=512, max_time_steps=100):
        super().__init__()

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)

        # 时间步嵌入层
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)

        # 位置编码（可选）
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

        # 自定义编码器和解码器
        encoder_layer = CustomEncoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dim_feedforward=2048)
        decoder_layer = CustomDecoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dim_feedforward=2048)

        self.encoder = CustomEncoder(encoder_layer=encoder_layer,
                                     num_layers=num_layers)

        self.decoder = CustomDecoder(decoder_layer=decoder_layer,
                                     num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        x_time_steps = x_time_steps.long()
        # 输入嵌入和时间步嵌入
        x_embedded = self.embedding(x_in_pressures_flat) + \
                     self.time_step_embedding(x_time_steps) + \
                     self.positional_encoding

        # 编码器处理
        encoder_output = self.encoder(x_embedded.unsqueeze(1))  # 保持维度 [batch, seq_len=1, d_model]

        # 解码器处理 (不做squeeze)
        decoder_output = self.decoder(encoder_output, encoder_output)

        # 最终输出
        out_pressure_flat_pred = self.fc_out(decoder_output.mean(dim=1))

        return out_pressure_flat_pred


def load_data():
    case_folder = "case_Re_500"
    case_path = r"F:\Zhaoyang"
    merged_file_path = os.path.join(case_path, f"merged_all_pressures_separated_normalized.pt")

    dataset = PressureDataset(merged_file_path)

    # 数据集划分 7:2:1
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


# 训练过程
def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cpu',
                early_stop_patience=10):
    model.to(device)
    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    best_valid_loss = float('inf')
    patience_counter = 0

    plt.ion()  # 开启交互模式

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, (in_press, out_pressure, time_steps) in enumerate(train_loader):
            in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)

            optimizer.zero_grad()
            model_out = model(in_press, time_steps)
            loss_value = criterion(model_out, out_pressure)
            loss_value.backward()
            optimizer.step()

            total_train_loss += loss_value.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in valid_loader:
                in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)
                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure)
                total_valid_loss += loss_value.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        total_test_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in test_loader:
                in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)
                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure)
                total_test_loss += loss_value.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # 早期停止逻辑
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

            # 每10个epoch保存预测图像
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    sample_input, sample_output, sample_time_steps = next(iter(valid_loader))
                    sample_input = sample_input.to(device)
                    sample_output = sample_output.to(device)
                    sample_time_steps = sample_time_steps.to(device)
                    predictions = model(sample_input, sample_time_steps)

                    for idx in range(len(predictions)):
                        # 将输入、真实输出、预测输出展平并转回二维
                        input_pressure = sample_input[idx].view(20, 20).cpu().numpy()
                        true_pressure = sample_output[idx].view(200, 200).cpu().numpy()
                        predicted_pressure = predictions[idx].view(200, 200).cpu().numpy()

                        plot_comparison_figure(input_pressure, true_pressure, predicted_pressure,
                                               sample_time_steps[idx].item(), epoch + 1, idx,
                                               mode='validation')

                except StopIteration:
                    continue

    plt.ioff()  # 关闭交互模式
    return model, train_loss_history, valid_loss_history, test_loss_history


# 测试过程
def test_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for in_press, out_pressure, time_steps in test_loader:
            in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)
            model_out = model(in_press, time_steps)
            loss_value = criterion(model_out, out_pressure)
            total_test_loss += loss_value.item()

            # 保存预测图像（每隔一定数量的批次）
            for idx in range(len(model_out)):
                input_pressure = in_press[idx].view(20, 20).cpu().numpy()
                true_pressure = out_pressure[idx].view(200, 200).cpu().numpy()
                predicted_pressure = model_out[idx].view(200, 200).cpu().numpy()

                plot_comparison_figure(input_pressure, true_pressure, predicted_pressure,
                                       time_steps[idx].item(), 0, idx,
                                       mode='test')

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")


# 绘制损失图
def plot_losses(train_loss_history, valid_loss_history, test_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.plot(test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training  and Validation Losses')
    plt.legend()
    plt.show()


class RelativePositionSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=500):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 正确初始化相对位置嵌入
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(num_heads, 2 * max_len - 1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 计算正确的相对位置索引
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        relative_position = position_ids.unsqueeze(-1) - position_ids.unsqueeze(0) + seq_len - 1  # 偏移保证索引>=0

        relative_position_embeddings = self.relative_position_embeddings[:, relative_position]
        scores += relative_position_embeddings.unsqueeze(0)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output


class SparseSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.block_size = block_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 创建稀疏掩码
        mask = torch.ones_like(scores)
        for i in range(seq_len):
            start = (i // self.block_size) * self.block_size
            end = (i // self.block_size + 1) * self.block_size
            mask[:, :, i, start:end] = 0

        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output


class LSHSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_hash_functions=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_hash_functions = num_hash_functions

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def hash_function(self, x):
        return torch.randint(0, 2, (x.size(0), x.size(1))).to(x.device)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 哈希掩码
        hash_codes_q = self.hash_function(q)
        hash_codes_k = self.hash_function(k)

        mask = (hash_codes_q.unsqueeze(-1) != hash_codes_k.unsqueeze(-2)).float()

        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output


class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 替换为任意一种新的自注意力机制
        # 示例：使用相对位置注意力
        self.self_attn = RelativePositionSelfAttention(d_model, num_heads)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 自注意力层（例如相对位置注意力）
        self.self_attn = RelativePositionSelfAttention(d_model, num_heads)

        # 交叉注意力层（同样可以替换为其他类型）
        self.multihead_attn = RelativePositionSelfAttention(d_model, num_heads)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt)  # 假设memory也经过相同处理
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class CustomEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class CustomDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


def clones(module, N):
    """生成N个相同的层"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def main():
    learning_rate = 0.0001
    train_loader, valid_loader, test_loader = load_data()

    input_dim = 400
    output_dim = 40000
    model = TransformerFlowReconstructionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=8,
        num_layers=6,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trained_model, train_loss, valid_loss, test_loss = train_model(
        model,
        train_loader,
        valid_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=100,
        device=device
    )

    plot_losses(train_loss, valid_loss, test_loss)

    test_model(trained_model, test_loader, criterion, device=device)


if __name__ == "__main__":
    main()