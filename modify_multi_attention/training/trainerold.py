import os
import threading
import torch
import time
from modify_multi_attention.utils.visualization import plot_comparison_figure

# 训练过程
import torch
import matplotlib.pyplot as plt
import os

def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cuda',
                early_stop_patience=10, attention_type='default'):
    model.to(device)

    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for i, (in_press, out_pressure, time_steps) in enumerate(train_loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device)
            )

            optimizer.zero_grad()

            model_out = model(in_press, time_steps)

            loss_value = criterion(model_out, out_pressure)
            loss_value.backward()
            optimizer.step()

            total_train_loss += loss_value.item()

            if (i + 1) % 50 == 0 or i == 0:
                print(f"    🔄 Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss_value.item():.6f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in valid_loader:
                in_press, out_pressure, time_steps = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device)
                )
                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure)
                total_valid_loss += loss_value.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        total_test_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in test_loader:
                in_press, out_pressure, time_steps = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device)
                )
                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure)
                total_test_loss += loss_value.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(
            f"🎯 Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, "
            f"Valid Loss: {avg_valid_loss:.6f}, Test Loss: {avg_test_loss:.6f}"
        )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0

            save_dir = f"attention_results/{attention_type}"
            os.makedirs(save_dir, exist_ok=True)  # 新增的目录创建代码

            torch.save(model.state_dict(), f"{save_dir}/best_model_{attention_type}.pth")
            print("✅ 模型已保存 (Best Model Updated)")

        else:
            patience_counter += 1
            print(f"⚠️ 早停计数: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print("⏹️ 触发 Early Stopping!")
            break

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    sample_input, sample_output, sample_time_steps = next(iter(valid_loader))
                    sample_input, sample_output, sample_time_steps = (
                        sample_input.to(device),
                        sample_output.to(device),
                        sample_time_steps.to(device)
                    )
                    predictions = model(sample_input, sample_time_steps)

                    for idx in range(len(predictions)):
                        input_pressure = sample_input[idx].view(20, 20).cpu().numpy()
                        true_pressure = sample_output[idx].view(200, 200).cpu().numpy()
                        predicted_pressure = predictions[idx].view(200, 200).cpu().numpy()

                        plot_comparison_figure(
                            input_pressure=input_pressure,
                            true_pressure=true_pressure,
                            predicted_pressure=predicted_pressure,
                            time_step=sample_time_steps[idx].item(),
                            epoch=epoch + 1,
                            idx=idx,
                            attention_type=attention_type,  # 只通过关键字传入
                            parent_dir="attention_results",
                            mode='validation'
                        )


                except StopIteration:
                    print("⚠️ 验证集数据不足，无法生成可视化结果。")

    plt.ioff()
    return model, train_loss_history, valid_loss_history, test_loss_history




# 测试过程
def test_model(model, test_loader, criterion, device='cuda', attention_type='default', parent_dir="attention_results"):
    model.eval()
    model.to(device)
    total_test_loss = 0

    with torch.no_grad():
        for idx, (in_press, out_pressure, time_steps) in enumerate(test_loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device)
            )

            model_out = model(in_press, time_steps)
            loss_value = criterion(model_out, out_pressure)
            total_test_loss += loss_value.item()

            # 可视化保存路径明确到注意力机制子目录
            input_pressure = in_press[0].view(20, 20).cpu().numpy()
            true_pressure = out_pressure[0].view(200, 200).cpu().numpy()
            predicted_pressure = model_out[0].view(200, 200).cpu().numpy()

            plot_comparison_figure(
                input_pressure=input_pressure,
                true_pressure=true_pressure,
                predicted_pressure=predicted_pressure,
                time_step=time_steps[0].item(),
                epoch=0,
                idx=idx,
                attention_type=attention_type,  # 明确注意力机制参数
                parent_dir=parent_dir,
                mode='test'
            )

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    return avg_test_loss
