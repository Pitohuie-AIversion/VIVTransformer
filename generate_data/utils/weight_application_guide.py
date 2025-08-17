#!/usr/bin/env python3
"""
SVD 损失权重应用指南
====================

此脚本提供了将权重计算器生成的权重配置应用到SVD损失函数的具体示例和代码。
支持基础版和增强版SVD损失函数。

使用方式:
    python weight_application_guide.py --config svd_loss_config_data_adaptive.yaml
    
    或直接修改代码中的权重值并运行
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径以便导入模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "modify_multi_attention" / "utils"))

def load_weight_config(config_path: str) -> dict:
    """加载权重配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 成功加载配置: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_path}")
        return {}
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return {}

def demonstrate_basic_svd_loss(base_weight: float, svd_weights: list, topk: int):
    """演示基础 SVD 损失的使用"""
    print("\n" + "="*60)
    print("基础 SVD 损失函数使用示例")
    print("="*60)
    
    try:
        # 导入基础 SVD 损失
        from loss import TotalLossWithSVD
        
        # 创建损失函数
        criterion = TotalLossWithSVD(
            base_weight=base_weight,
            svd_weights=svd_weights,
            topk=topk
        )
        
        print("✅ 基础 SVD 损失函数创建成功")
        
        # 打印权重信息
        criterion.print_weight_info()
        
        # 测试损失计算
        print("\n📊 测试损失计算...")
        batch_size, channels, height, width = 4, 1, 64, 64
        pred = torch.randn(batch_size, channels, height, width)
        target = torch.randn(batch_size, channels, height, width)
        
        loss = criterion(pred, target)
        print(f"计算得到的损失值: {loss.item():.6f}")
        
        return criterion
        
    except ImportError as e:
        print(f"❌ 导入基础 SVD 损失模块失败: {e}")
        print("请确保 modify_multi_attention/utils/loss.py 存在")
        return None
    except Exception as e:
        print(f"❌ 创建基础 SVD 损失函数失败: {e}")
        return None

def demonstrate_enhanced_svd_loss(base_weight: float, svd_weights: list, topk: int, 
                                adaptive_weights: bool = False):
    """演示增强版 SVD 损失的使用"""
    print("\n" + "="*60)
    print("增强版 SVD 损失函数使用示例")
    print("="*60)
    
    try:
        # 导入增强版 SVD 损失
        from enhanced_svd_loss import create_enhanced_svd_loss
        
        # 创建增强版损失函数
        criterion = create_enhanced_svd_loss(
            base_weight=base_weight,
            svd_weights=svd_weights,
            topk=topk,
            mixed_precision=True,
            adaptive_weights=adaptive_weights,
            monitoring=True,
            fallback_level=2
        )
        
        print("✅ 增强版 SVD 损失函数创建成功")
        
        # 打印权重信息
        criterion.print_weight_info()
        
        # 测试损失计算
        print("\n📊 测试损失计算...")
        batch_size, channels, height, width = 4, 1, 64, 64
        pred = torch.randn(batch_size, channels, height, width)
        target = torch.randn(batch_size, channels, height, width)
        
        loss = criterion(pred, target)
        print(f"计算得到的损失值: {loss.item():.6f}")
        
        return criterion
        
    except ImportError as e:
        print(f"❌ 导入增强版 SVD 损失模块失败: {e}")
        print("请确保 modify_multi_attention/utils/enhanced_svd_loss.py 存在")
        return None
    except Exception as e:
        print(f"❌ 创建增强版 SVD 损失函数失败: {e}")
        return None

def generate_training_code_example(config: dict, output_path: str = None):
    """生成训练代码示例"""
    print("\n" + "="*60)
    print("生成训练代码示例")
    print("="*60)
    
    base_weight = config.get('base_weight', 0.5)
    svd_weights = config.get('svd_weights', [0.05] * 10)
    topk = config.get('topk', 10)
    
    code_template = f'''# SVD 损失函数训练示例代码
import torch
import torch.nn as nn
import torch.optim as optim
from modify_multi_attention.utils.enhanced_svd_loss import create_enhanced_svd_loss

def setup_svd_loss():
    """设置 SVD 损失函数"""
    criterion = create_enhanced_svd_loss(
        base_weight={base_weight:.3f},
        svd_weights={svd_weights},
        topk={topk},
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
                print(f'Epoch {{epoch+1}}/{{num_epochs}}, Batch {{batch_idx}}, Loss: {{loss.item():.6f}}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {{epoch+1}} 平均损失: {{avg_loss:.6f}}')
        
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
'''
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code_template)
            print(f"✅ 训练代码示例已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存训练代码示例失败: {e}")
    else:
        print("🔍 训练代码示例:")
        print(code_template)

def main():
    parser = argparse.ArgumentParser(description='SVD 损失权重应用指南')
    parser.add_argument('--config', type=str, 
                       default='svd_loss_config_data_adaptive.yaml',
                       help='权重配置文件路径')
    parser.add_argument('--demo-basic', action='store_true',
                       help='演示基础 SVD 损失')
    parser.add_argument('--demo-enhanced', action='store_true',
                       help='演示增强版 SVD 损失')
    parser.add_argument('--generate-code', action='store_true',
                       help='生成训练代码示例')
    parser.add_argument('--output-code', type=str,
                       help='训练代码输出路径')
    
    args = parser.parse_args()
    
    print("🎯 SVD 损失权重应用指南")
    print("="*60)
    
    # 如果没有指定具体操作，则执行所有演示
    if not any([args.demo_basic, args.demo_enhanced, args.generate_code]):
        args.demo_basic = True
        args.demo_enhanced = True
        args.generate_code = True
    
    # 加载配置
    config = load_weight_config(args.config)
    if not config:
        print("❌ 无法加载配置，使用默认权重演示")
        config = {
            'base_weight': 0.5,
            'svd_weights': [0.05] * 10,
            'topk': 10
        }
    
    base_weight = config.get('base_weight', 0.5)
    svd_weights = config.get('svd_weights', [0.05] * 10)
    topk = config.get('topk', 10)
    
    print(f"\n📋 当前配置:")
    print(f"  - 基础权重: {base_weight:.3f}")
    print(f"  - SVD权重: {[f'{w:.4f}' for w in svd_weights[:5]]}..." if len(svd_weights) > 5 else f"  - SVD权重: {[f'{w:.4f}' for w in svd_weights]}")
    print(f"  - TopK模态: {topk}")
    
    # 演示基础 SVD 损失
    if args.demo_basic:
        demonstrate_basic_svd_loss(base_weight, svd_weights, topk)
    
    # 演示增强版 SVD 损失
    if args.demo_enhanced:
        demonstrate_enhanced_svd_loss(base_weight, svd_weights, topk, adaptive_weights=True)
    
    # 生成训练代码示例
    if args.generate_code:
        output_path = args.output_code or "svd_training_example.py"
        generate_training_code_example(config, output_path)
    
    print("\n" + "="*60)
    print("🎉 权重应用指南演示完成!")
    print("="*60)
    print("\n📝 使用建议:")
    print("1. 从较保守的权重开始（基础权重较高）")
    print("2. 根据训练效果逐步调整权重比例")
    print("3. 启用自适应权重功能以获得更好的训练稳定性")
    print("4. 监控训练过程中的权重变化")
    print("5. 如果遇到不稳定，可以减少 topk 或增加基础权重")

if __name__ == "__main__":
    main()