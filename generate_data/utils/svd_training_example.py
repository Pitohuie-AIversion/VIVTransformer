# SVD 损失函数训练示例代码
import torch
import torch.nn as nn
import torch.optim as optim
from modify_multi_attention.utils.enhanced_svd_loss import create_enhanced_svd_loss

def setup_svd_loss():
    """设置 SVD 损失函数"""
    criterion = create_enhanced_svd_loss(
        base_weight=0.500,
        svd_weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        topk=10,
        mixed_precision=True,
        adaptive_weights=True,  # 启用自适应权重调整
        monitoring=True,        # 启用性能监控
        fallback_level=2        # 设置回退级别
    )
    return criterion

def train_with_svd_loss(model, dataloader, num_epochs=10):
    """使用 SVD 损失进行训练"""
    criterion = setup_svd_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算 SVD 损失
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} 平均损失: {avg_loss:.6f}')
        
        # 打印当前权重信息（如果启用了自适应权重）
        if hasattr(criterion, 'print_weight_info'):
            criterion.print_weight_info()

# 使用示例
if __name__ == "__main__":
    # 创建模型和数据加载器（这里需要替换为你的实际模型和数据）
    # model = YourModel()
    # dataloader = YourDataLoader()
    
    # 开始训练
    # train_with_svd_loss(model, dataloader)
    
    # 或者只是测试损失函数
    criterion = setup_svd_loss()
    print("SVD 损失函数设置完成")
    criterion.print_weight_info()
