import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置环境变量以解决多线程问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保结果文件夹存在
result_folder = "visualization_results"
loss_folder = "loss_plots"
model_folder = "saved_models"
for folder in [result_folder, loss_folder, model_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)


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


class PressureDataset(Dataset):
    def __init__(self, merged_file_path):
        self.data = torch.load(merged_file_path)
        self.in_pressures = self.data['in_pressure']
        self.pressures = self.data['pressure']
        self.time_steps = np.array(self.data['time_steps'])

        if len(self.time_steps.shape) == 1:
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

        print(f"time_steps shape: {self.time_steps.shape}")
        print(f"time_steps type: {type(self.time_steps)}")

        self.num_samples = len(self.in_pressures) * len(self.in_pressures[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        reynolds_idx = idx // len(self.in_pressures[0])
        time_step_idx = idx % len(self.in_pressures[0])

        assert reynolds_idx < len(self.in_pressures), "Reynolds index out of bounds"
        assert time_step_idx < len(self.in_pressures[0]), "Time step index out of bounds"

        in_pressures = self.in_pressures[reynolds_idx, time_step_idx]
        pressure = self.pressures[reynolds_idx, time_step_idx]

        if isinstance(self.time_steps[reynolds_idx], (list, np.ndarray)):
            time_step = self.time_steps[reynolds_idx][time_step_idx]
        else:
            raise TypeError("time_steps[reynolds_idx] is not subscriptable")

        in_pressures_flat = in_pressures.view(-1)
        pressure_flat = pressure.view(-1)

        return in_pressures_flat, pressure_flat, time_step


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class TransformerFlowReconstructionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, d_model=256, max_time_steps=100):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=1024,
            dropout=0.1
        )

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, time_steps):
        time_steps = time_steps.long()
        time_step_encoding = self.time_step_embedding(time_steps)
        x = self.embedding(x)
        x = x + time_step_encoding.unsqueeze(1)
        x = self.pos_encoder(x)
        x = self.transformer(x, x)
        out = self.fc_out(x.mean(dim=1))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:x.size(1)]


def load_data(data_path="F:/Zhaoyang/merged_all_pressures_separated_normalized.pt"):
    dataset = PressureDataset(data_path)

    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader


def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler=None,
                num_epochs=100, device='cuda', early_stop_patience=10):
    model.to(device)
    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for in_press, out_pressure, time_steps in train_loader:
            in_press, out_pressure, time_steps = (
                in_press.to(device),
                out_pressure.to(device),
                time_steps.to(device)
            )

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
                in_press, out_pressure, time_steps = (
                    in_press.to(device),
                    out_pressure.to(device),
                    time_steps.to(device)
                )

                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure)
                total_valid_loss += loss_value.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        if scheduler is not None:
            scheduler.step(avg_valid_loss)

            # Early stopping logic
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_folder, "best_model.pth"))
        else:
            patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

                # Save intermediate results every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    sample_batch = next(iter(valid_loader))
                    in_press, out_pressure, time_steps = [
                        x.to(device) for x in sample_batch
                    ]

                    model_out = model(in_press, time_steps)

                    for idx in range(min(len(model_out), 5)):  # Only save first 5 samples
                        input_pressure = in_press[idx].view(20, 20).cpu().numpy()
                        true_pressure = out_pressure[idx].view(200, 200).cpu().numpy()
                        predicted_pressure = model_out[idx].view(200, 200).cpu().numpy()

                        plot_comparison_figure(
                            input_pressure,
                            true_pressure,
                            predicted_pressure,
                            time_steps[idx].item(),
                            epoch + 1,
                            idx,
                            mode='validation'
                        )

                except StopIteration:
                    continue

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Valid Loss: {avg_valid_loss:.4f}")

    # Final validation on test set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for in_press, out_pressure, time_steps in test_loader:
            in_press, out_pressure, time_steps = (
                in_press.to(device),
                out_pressure.to(device),
                time_steps.to(device)
            )

            model_out = model(in_press, time_steps)
            loss_value = criterion(model_out, out_pressure)
            total_test_loss += loss_value.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_loss_history.append(avg_test_loss)

    print(f"Final Test Loss: {avg_test_loss:.4f}")

    # Plot and save losses
    plot_losses(train_loss_history, valid_loss_history, os.path.join(loss_folder, "training_curves.png"))

    return model


def plot_losses(train_loss, valid_loss, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss', alpha=0.7)
    plt.plot(valid_loss, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training  and Validation Loss Curves')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved at {save_path}")

    plt.close()


def main():
    data_path = r"F:\Zhaoyang\merged_all_pressures_separated_normalized.pt"
    train_loader, valid_loader, test_loader = load_data(data_path)

    input_dim = 400
    output_dim = 40000
    model = TransformerFlowReconstructionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=8,
        num_layers=6,
        d_model=256
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trained_model = train_model(
        model,
        train_loader,
        valid_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        early_stop_patience=15
    )


if __name__ == "__main__":
    main()