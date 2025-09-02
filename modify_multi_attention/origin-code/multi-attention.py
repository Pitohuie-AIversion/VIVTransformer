import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

# 解决多线程问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 结果目录
result_folder = "visualization_results"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


def plot_comparison_figure(input_pressure, true_pressure, predicted_pressure, time_step, epoch, idx, mode='train'):
    plt.figure(figsize=(24, 8))
    # 输入
    plt.subplot(1, 3, 1)
    plt.imshow(input_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Input  Pressure Matrix at t={time_step:.4f}")
    plt.xlabel('x'); plt.ylabel('y')
    # 真实
    plt.subplot(1, 3, 2)
    plt.imshow(true_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"True  Pressure Matrix at t={time_step:.4f}")
    plt.xlabel('x'); plt.ylabel('y')
    # 预测
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Predicted  Pressure Matrix at t={time_step:.4f}")
    plt.xlabel('x'); plt.ylabel('y')

    save_path = os.path.join(result_folder, f"{mode}_epoch_{epoch}_sample_{idx + 1}.png")
    plt.savefig(save_path)
    plt.close()


# =========================
# 数据集
# =========================
class PressureDataset(Dataset):
    def __init__(self, merged_file_path):
        self.data = torch.load(merged_file_path)

        # 形状 [R, T, 20, 20] 与 [R, T, 200, 200]
        self.in_pressures = self.data['in_pressure']
        self.pressures = self.data['pressure']
        self.time_steps = np.array(self.data['time_steps'])  # 可能是一维或二维的 numpy

        # 统一 time_steps 形状为 [R, T]
        if len(self.time_steps.shape) == 1:
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

        # 归一化时间（0~1），供连续时间嵌入使用
        self.t_min = float(np.min(self.time_steps))
        self.t_max = float(np.max(self.time_steps))
        self.t_den = (self.t_max - self.t_min) + 1e-8

        print(f"time_steps shape: {self.time_steps.shape}")
        print(f"time_steps type: {type(self.time_steps)}")
        print(f"time min/max: {self.t_min:.6f}/{self.t_max:.6f}")

        self.R = len(self.in_pressures)
        self.T = len(self.in_pressures[0])
        self.num_samples = self.R * self.T

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        reynolds_idx = idx // self.T
        time_step_idx = idx % self.T

        if reynolds_idx >= self.R or time_step_idx >= self.T:
            raise IndexError("Index out of bounds")

        in_pressures = self.in_pressures[reynolds_idx, time_step_idx]  # [20,20], torch.Tensor
        pressure = self.pressures[reynolds_idx, time_step_idx]         # [200,200], torch.Tensor

        # 原始时间与归一化时间
        tt = float(self.time_steps[reynolds_idx][time_step_idx])
        tt_norm = (tt - self.t_min) / self.t_den  # 0~1

        in_pressures_flat = in_pressures.view(-1)  # [400]
        pressure_flat = pressure.view(-1)          # [40000]

        # 返回连续时间（float），模型里用 MLP 嵌入
        return in_pressures_flat, pressure_flat, np.float32(tt_norm)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


# =========================
# 注意力模块
# =========================
class RelativePositionSelfAttention(nn.Module):
    """相对位置自注意力（用于自注意力场景）"""
    def __init__(self, d_model, num_heads, max_len=500):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.relative_position_embeddings = nn.Parameter(
            torch.randn(num_heads, 2 * max_len - 1)
        )

    def forward(self, q, k=None, v=None):
        # 本模块仅用于自注意力，若传入 k/v 则忽略，按 q 自身处理
        x = q
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        relative_position = position_ids.unsqueeze(-1) - position_ids.unsqueeze(0) + seq_len - 1
        relative_position_embeddings = self.relative_position_embeddings[:, relative_position]  # [H, L, L]
        scores = scores + relative_position_embeddings.unsqueeze(0)  # [B,H,L,L]

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,L,d_v]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [B,L,d_model]
        return out


class SparseSelfAttention(nn.Module):
    """块稀疏自注意力（用于自注意力场景）"""
    def __init__(self, d_model, num_heads, block_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.block_size = block_size

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, q, k=None, v=None):
        x = q
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 稀疏掩码：仅允许每个 token 关注自身所在块
        mask = torch.ones_like(scores, dtype=torch.bool)
        for i in range(seq_len):
            start = (i // self.block_size) * self.block_size
            end = min((i // self.block_size + 1) * self.block_size, seq_len)
            mask[:, :, i, start:end] = False  # 允许的区域设为 False（不屏蔽）

        scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return out


class LSHSelfAttention(nn.Module):
    """简化版 LSH 自注意力（用于自注意力场景，演示用）"""
    def __init__(self, d_model, num_heads, num_hash_functions=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_hash_functions = num_hash_functions

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def hash_function(self, x):
        # 这里用随机 hash（演示），实际可替换为可重复的投影-符号函数
        return torch.randint(0, 2, (x.size(0), x.size(1)), device=x.device)

    def forward(self, q, k=None, v=None):
        x = q
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # LSH 掩码：仅允许相同桶交互（演示）
        hash_codes_q = self.hash_function(q[..., 0])  # [B,H,L] -> 用一个通道近似
        hash_codes_k = self.hash_function(k[..., 0])
        mask = (hash_codes_q.unsqueeze(-1) != hash_codes_k.unsqueeze(-2)).unsqueeze(2)  # [B,1,L,L] 广播到 H
        scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return out


class StdQKV(nn.Module):
    """标准多头注意力（可用于自注意力或交叉注意力）"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, q, k=None, v=None):
        if k is None: k = q
        if v is None: v = k
        out, _ = self.attn(q, k, v)
        return out


def get_attention_module(attention_type, d_model, num_heads):
    if attention_type == "relative":
        return RelativePositionSelfAttention(d_model, num_heads)
    if attention_type == "sparse":
        return SparseSelfAttention(d_model, num_heads)
    if attention_type == "lsh":
        return LSHSelfAttention(d_model, num_heads)
    if attention_type == "self":  # 标准 QKV
        return StdQKV(d_model, num_heads)
    raise ValueError(f"Unknown attention type: {attention_type}")


# =========================
# 编解码器层
# =========================
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative"):
        super().__init__()
        self.self_attn = get_attention_module(attention_type, d_model, num_heads)  # 自注意力
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative"):
        super().__init__()
        self.self_attn  = get_attention_module(attention_type, d_model, num_heads)  # 自注意力
        self.cross_attn = StdQKV(d_model, num_heads)  # 交叉注意力统一用标准 MHA，稳定可靠
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # 自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        # 交叉注意力：Q=tgt, K=V=memory
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        # FFN
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class CustomEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class CustomDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


# =========================
# 主模型
# =========================
class TransformerFlowReconstructionModel(nn.Module):
    """
    关键变化：
    1) embedding: input_dim -> (seq_len * d_model)，再 reshape 为 [B, seq_len, d_model]
    2) 连续时间嵌入 MLP（不再 .long()）
    3) 交叉注意力修正为 (Q=tgt, K=V=memory)
    """
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6,
                 d_model=512, attention_type="relative", seq_len=49,
                 input_hw=None, pe_type='learnable_1d', output_head_type='global', out_channels_per_token=None):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # 兼容新增参数但在旧版实现中不使用
        self._compat_input_hw = input_hw
        self._compat_pe_type = pe_type
        self._compat_output_head_type = output_head_type
        self._compat_out_channels_per_token = out_channels_per_token

        # 把 400 维映射到 seq_len * d_model，并 reshape 成 [B, seq_len, d_model]
        self.embedding = nn.Linear(input_dim, seq_len * d_model)

        # 连续时间嵌入（输入归一化到 0~1 的实数）
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 可学习二维位置编码（这里简化为 [1, seq_len, d_model]）
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = CustomEncoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dim_feedforward=4*d_model,
                                           dropout=0.1,
                                           attention_type=attention_type)

        decoder_layer = CustomDecoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dim_feedforward=4*d_model,
                                           dropout=0.1,
                                           attention_type=attention_type)

        self.encoder = CustomEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.decoder = CustomDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # 输出层：pool 后线性映射到 200x200=40000
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        """
        x_in_pressures_flat: [B, 400]
        x_time_steps:       [B] 或 [B,1]，为已归一化到 0~1 的 float
        """
        B = x_in_pressures_flat.size(0)

        # 输入 token 化
        x = self.embedding(x_in_pressures_flat)                # [B, seq_len*d_model]
        x = x.view(B, self.seq_len, self.d_model)              # [B, seq_len, d_model]

        # 连续时间嵌入
        if x_time_steps.dim() == 1:
            x_time_steps = x_time_steps.unsqueeze(-1)
        time_embed = self.time_mlp(x_time_steps.float())       # [B, d_model]
        time_embed = time_embed.unsqueeze(1)                   # [B,1,d_model] -> 广播到序列

        # 加时间与位置
        x = x + time_embed + self.positional_encoding          # [B, seq_len, d_model]

        # 编码
        memory = self.encoder(x)                               # [B, seq_len, d_model]

        # 解码：以 memory 作为初始 tgt（也可用可学习 query，这里取最简）
        tgt = memory
        dec = self.decoder(tgt, memory)                        # [B, seq_len, d_model]

        pooled = dec.mean(dim=1)                               # [B, d_model]
        out_pressure_flat_pred = self.fc_out(pooled)           # [B, 40000]
        return out_pressure_flat_pred


# =========================
# 数据加载
# =========================
def load_data():
    case_folder = "case_Re_500"
    case_path = r"F:\Zhaoyang"
    merged_file_path = os.path.join(case_path, f"merged_all_pressures_separated_normalized.pt")

    dataset = PressureDataset(merged_file_path)

    # 7:2:1 随机切分（如需避免泄漏，建议按 Reynolds/序列做分组切分）
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


# =========================
# 训练 / 评估 / 可视化
# =========================
def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cpu',
                early_stop_patience=100):
    model.to(device)
    train_loss_history, valid_loss_history, test_loss_history = [], [], []

    best_valid_loss = float('inf')
    patience_counter = 0
    plt.ion()

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        for i, (in_press, out_pressure, time_steps) in enumerate(train_loader):
            in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)

            optimizer.zero_grad()
            preds = model(in_press, time_steps)
            loss = criterion(preds, out_pressure)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_loss_history.append(avg_train_loss)

        # ---- Valid ----
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in valid_loader:
                in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)
                preds = model(in_press, time_steps)
                loss = criterion(preds, out_pressure)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / max(1, len(valid_loader))
        valid_loss_history.append(avg_valid_loss)

        # （可选）每轮计算 test，或只在最终评估
        total_test_loss = 0.0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in test_loader:
                in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)
                preds = model(in_press, time_steps)
                loss = criterion(preds, out_pressure)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / max(1, len(test_loader))
        test_loss_history.append(avg_test_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train: {avg_train_loss:.6f}  Valid: {avg_valid_loss:.6f}  Test: {avg_test_loss:.6f}")

        # Early Stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model.pth")
        else:
            patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

        # 每 10 轮保存一批可视化
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
                        input_pressure = sample_input[idx].view(20, 20).cpu().numpy()
                        true_pressure  = sample_output[idx].view(200, 200).cpu().numpy()
                        predicted_pressure = predictions[idx].view(200, 200).cpu().numpy()

                        plot_comparison_figure(input_pressure, true_pressure, predicted_pressure,
                                               float(sample_time_steps[idx].item()), epoch + 1, idx,
                                               mode='validation')
                except StopIteration:
                    pass

    plt.ioff()
    return model, train_loss_history, valid_loss_history, test_loss_history


def test_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for in_press, out_pressure, time_steps in test_loader:
            in_press, out_pressure, time_steps = in_press.to(device), out_pressure.to(device), time_steps.to(device)
            preds = model(in_press, time_steps)
            loss = criterion(preds, out_pressure)
            total_test_loss += loss.item()

            # 可视化
            for idx in range(len(preds)):
                input_pressure = in_press[idx].view(20, 20).cpu().numpy()
                true_pressure  = out_pressure[idx].view(200, 200).cpu().numpy()
                predicted_pressure = preds[idx].view(200, 200).cpu().numpy()

                plot_comparison_figure(input_pressure, true_pressure, predicted_pressure,
                                       float(time_steps[idx].item()), 0, idx, mode='test')

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    print(f"Test Loss: {avg_test_loss:.6f}")


def plot_losses(train_loss_history, valid_loss_history, test_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    if len(test_loss_history) > 0:
        plt.plot(test_loss_history, label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training / Validation / Test Losses')
    plt.legend(); plt.show()


# =========================
# 主入口
# =========================
def main():
    learning_rate = 1e-4
    attention_type = "relative"   # 可选: "relative" / "sparse" / "lsh" / "self"
    seq_len = 49                  # 可改为 400（更重），或其它分块策略

    train_loader, valid_loader, test_loader = load_data()

    input_dim = 400
    output_dim = 40000
    model = TransformerFlowReconstructionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=8,
        num_layers=6,
        d_model=512,
        attention_type=attention_type,
        seq_len=seq_len,
        input_hw=(int(seq_len**0.5), int(seq_len**0.5)) if int(seq_len**0.5)**2 == seq_len else None,
        pe_type='learnable_1d',
        output_head_type='global',
        out_channels_per_token=None
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
