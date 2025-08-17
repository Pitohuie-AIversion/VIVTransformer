#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示如何使用修复后的SVD损失函数进行稳定训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from svd10_loss import TotalLossWithSVD
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
# 统一使用全局 sitecustomize.py 的中文字体和负号设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_size=1024, hidden_size=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_synthetic_data(batch_size=8, size=32, num_batches=100):
    """创建合成数据用于测试"""
    data = []
    for _ in range(num_batches):
        # 创建具有不同特性的数据
        if np.random.random() < 0.1:  # 10%概率创建病态数据
            x = torch.randn(batch_size, size, size) * 1e-6  # 非常小的值
        elif np.random.random() < 0.1:  # 10%概率创建大值数据
            x = torch.randn(batch_size, size, size) * 1e3   # 大值
        else:
            x = torch.randn(batch_size, size, size)         # 正常数据
        
        # 目标是输入的简单变换
        y = x + 0.1 * torch.randn_like(x)
        data.append((x.view(batch_size, -1), y.view(batch_size, -1)))
    
    return data

def train_with_robust_svd_loss():
    """使用稳定的SVD损失函数进行训练演示"""
    print("=== 稳定SVD损失函数训练演示 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleModel(input_size=1024).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建稳定的SVD损失函数
    criterion = TotalLossWithSVD(
        base_weight=0.7,     # 较高的基础权重确保稳定性
        topk=5               # 使用较少的模态减少计算复杂度
    )
    
    # 打印权重信息
    print("\n损失函数权重配置:")
    criterion.print_weight_info()
    
    # 创建数据
    train_data = create_synthetic_data(batch_size=4, size=32, num_batches=50)
    
    # 训练循环
    losses = []
    svd_failures = 0
    
    print("\n开始训练...")
    for epoch in range(10):
        epoch_losses = []
        
        for batch_idx, (x, y) in enumerate(train_data):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(x)
            
            # 计算损失
            loss = criterion(output.view(-1, 32, 32), y.view(-1, 32, 32))
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  警告: Epoch {epoch+1}, Batch {batch_idx+1} 损失无效")
                svd_failures += 1
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（推荐）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1:2d}: 平均损失 = {avg_loss:.6f}, 批次数 = {len(epoch_losses)}")
        else:
            print(f"Epoch {epoch+1:2d}: 所有批次都失败")
    
    print(f"\n训练完成!")
    print(f"SVD失败次数: {svd_failures}")
    print(f"成功训练批次: {len(train_data) * 10 - svd_failures}")
    
    return losses, model

def compare_loss_functions():
    """比较标准MSE损失和稳定SVD损失"""
    print("\n=== 损失函数比较 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    x = torch.randn(2, 32, 32, device=device)
    y = torch.randn(2, 32, 32, device=device)
    
    # 标准MSE损失
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(x, y)
    
    # 稳定SVD损失
    svd_loss = TotalLossWithSVD(base_weight=0.5, topk=5)
    svd_value = svd_loss(x, y)
    
    print(f"MSE损失: {mse_value.item():.6f}")
    print(f"SVD损失: {svd_value.item():.6f}")
    
    # 测试病态数据
    print("\n测试病态数据:")
    x_bad = torch.ones(2, 32, 32, device=device) * 1e6  # 大值
    y_bad = torch.ones(2, 32, 32, device=device) * 1e6
    
    try:
        mse_bad = mse_loss(x_bad, y_bad)
        print(f"MSE损失(病态): {mse_bad.item():.6f}")
    except Exception as e:
        print(f"MSE损失(病态): 失败 - {str(e)}")
    
    try:
        svd_bad = svd_loss(x_bad, y_bad)
        print(f"SVD损失(病态): {svd_bad.item():.6f}")
    except Exception as e:
        print(f"SVD损失(病态): 失败 - {str(e)}")

def plot_training_curve(losses):
    """绘制训练曲线"""
    if not losses:
        print("没有损失数据可绘制")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('稳定SVD损失函数训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度
    
    # 保存图片
    plt.savefig('svd_training_curve.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存为 svd_training_curve.png")
    plt.show()

def main():
    """主函数"""
    print("稳定SVD损失函数演示")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 比较损失函数
    compare_loss_functions()
    
    # 训练演示
    losses, model = train_with_robust_svd_loss()
    
    # 绘制训练曲线
    if losses:
        plot_training_curve(losses)
    
    print("\n=== 使用建议 ===")
    print("1. 在实际训练中，建议使用较高的基础权重(0.6-0.8)")
    print("2. 从较少的SVD模态开始(topk=3-5)，根据效果调整")
    print("3. 添加梯度裁剪以提高训练稳定性")
    print("4. 监控警告信息，了解SVD计算状态")
    print("5. 如果频繁出现SVD失败，考虑数据预处理")

if __name__ == "__main__":
    main()