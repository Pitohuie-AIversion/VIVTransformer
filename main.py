
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, input_dim, output_dim, num_heads, num_layers, d_model=512, max_time_steps=100):
        super(TransformerFlowReconstructionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x, time_steps):
        time_steps = time_steps.long()
        time_step_encoding = self.time_step_embedding(time_steps)
        x = self.embedding(x)
        x = x + self.positional_encoding + time_step_encoding
        x = x.unsqueeze(1)
        x = self.transformer(x, x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out

    # 加载数据


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


# 主函数
if __name__ == "__main__":
    learning_rate = 0.0001
    train_loader, valid_loader, test_loader = load_data()

    input_dim = 400  # 展平后的in_pressure维度（20x20）
    output_dim = 40000  # 展平后的整个压力场维度（200x200）
    model = TransformerFlowReconstructionModel(input_dim=input_dim, output_dim=output_dim, num_heads=8, num_layers=6)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trained_model, train_loss_history, valid_loss_history, test_loss_history = train_model(model, train_loader,
                                                                                           valid_loader,
                                                                                           test_loader,
                                                                                           criterion,
                                                                                           optimizer,
                                                                                           num_epochs=100,
                                                                                           device=device)

    plot_losses(train_loss_history, valid_loss_history, test_loss_history)

    test_model(trained_model, test_loader, criterion, device=device)