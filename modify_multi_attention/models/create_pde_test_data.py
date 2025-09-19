#!/usr/bin/env python3
"""
PDE测试数据生成器
生成符合各模型输入输出格式规范的标准化测试数据
确保数据处理流程、转换规则和输出结果严格遵循预设规范
"""

import numpy as np
import torch
from pathlib import Path

def create_standardized_pde_data():
    """
    创建标准化的PDE测试数据
    
    数据格式规范:
    - 输入数据: [batch, height, width] - 原始2D场数据
    - 输出数据: [batch, height, width] - 目标2D场数据
    - 数值范围: [0, 1] - 归一化到单位区间
    - 数据类型: float32 - 标准浮点精度
    - 物理意义: 模拟PDE解的时空演化
    """
    print("创建标准化PDE测试数据...")
    
    # 数据参数
    batch_size = 100
    height = 128
    width = 128
    
    # 生成物理意义的PDE数据
    # 模拟热传导方程、波动方程等的解
    x = np.linspace(0, 2*np.pi, width)
    y = np.linspace(0, 2*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    input_data = []
    output_data = []
    
    for i in range(batch_size):
        # 生成不同的初始条件和边界条件
        # 模拟不同的PDE参数和解
        
        # 输入: 初始状态或当前时刻的场
        freq_x = np.random.uniform(0.5, 3.0)
        freq_y = np.random.uniform(0.5, 3.0)
        phase_x = np.random.uniform(0, 2*np.pi)
        phase_y = np.random.uniform(0, 2*np.pi)
        
        # 生成具有物理意义的场分布
        input_field = (
            np.sin(freq_x * X + phase_x) * np.cos(freq_y * Y + phase_y) +
            0.3 * np.sin(2 * freq_x * X) * np.sin(2 * freq_y * Y) +
            0.1 * np.random.normal(0, 0.1, (height, width))  # 添加噪声
        )
        
        # 输出: 演化后的状态或目标场
        # 模拟时间演化、扩散、对流等物理过程
        diffusion_coeff = np.random.uniform(0.1, 0.5)
        time_step = np.random.uniform(0.1, 1.0)
        
        # 简化的扩散演化
        output_field = input_field * np.exp(-diffusion_coeff * time_step)
        
        # 添加非线性项（模拟反应扩散等）
        nonlinear_strength = np.random.uniform(0.1, 0.3)
        output_field += nonlinear_strength * np.sin(input_field * np.pi)
        
        # 归一化到[0, 1]区间
        input_field = (input_field - input_field.min()) / (input_field.max() - input_field.min())
        output_field = (output_field - output_field.min()) / (output_field.max() - output_field.min())
        
        input_data.append(input_field)
        output_data.append(output_field)
    
    # 转换为numpy数组
    input_data = np.array(input_data, dtype=np.float32)
    output_data = np.array(output_data, dtype=np.float32)
    
    print(f"生成数据形状:")
    print(f"  输入数据: {input_data.shape}")
    print(f"  输出数据: {output_data.shape}")
    print(f"  数据类型: {input_data.dtype}")
    print(f"  数值范围: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    return input_data, output_data

def validate_data_format(input_data, output_data):
    """
    验证数据格式是否符合规范
    
    Args:
        input_data: 输入数据
        output_data: 输出数据
        
    Returns:
        bool: 验证是否通过
    """
    print("\n验证数据格式规范...")
    
    checks = []
    
    # 检查数据形状
    if len(input_data.shape) == 3 and len(output_data.shape) == 3:
        checks.append(("✓", "数据维度正确 (3D)"))
    else:
        checks.append(("✗", f"数据维度错误: 输入{input_data.shape}, 输出{output_data.shape}"))
    
    # 检查批次大小一致性
    if input_data.shape[0] == output_data.shape[0]:
        checks.append(("✓", f"批次大小一致: {input_data.shape[0]}"))
    else:
        checks.append(("✗", f"批次大小不一致: 输入{input_data.shape[0]}, 输出{output_data.shape[0]}"))
    
    # 检查数据类型
    if input_data.dtype == np.float32 and output_data.dtype == np.float32:
        checks.append(("✓", "数据类型正确 (float32)"))
    else:
        checks.append(("✗", f"数据类型错误: 输入{input_data.dtype}, 输出{output_data.dtype}"))
    
    # 检查数值范围
    input_range = (input_data.min(), input_data.max())
    output_range = (output_data.min(), output_data.max())
    
    if 0 <= input_range[0] and input_range[1] <= 1 and 0 <= output_range[0] and output_range[1] <= 1:
        checks.append(("✓", f"数值范围正确: 输入[{input_range[0]:.3f}, {input_range[1]:.3f}], 输出[{output_range[0]:.3f}, {output_range[1]:.3f}]"))
    else:
        checks.append(("✗", f"数值范围错误: 输入[{input_range[0]:.3f}, {input_range[1]:.3f}], 输出[{output_range[0]:.3f}, {output_range[1]:.3f}]"))
    
    # 检查数据完整性
    if not np.any(np.isnan(input_data)) and not np.any(np.isnan(output_data)):
        checks.append(("✓", "数据完整性检查通过 (无NaN值)"))
    else:
        checks.append(("✗", "数据包含NaN值"))
    
    if not np.any(np.isinf(input_data)) and not np.any(np.isinf(output_data)):
        checks.append(("✓", "数据有界性检查通过 (无无穷值)"))
    else:
        checks.append(("✗", "数据包含无穷值"))
    
    # 输出检查结果
    all_passed = True
    for status, message in checks:
        print(f"  {status} {message}")
        if status == "✗":
            all_passed = False
    
    return all_passed

def save_pde_data(input_data, output_data, output_path):
    """
    保存PDE数据到标准格式文件
    
    Args:
        input_data: 输入数据
        output_data: 输出数据
        output_path: 输出文件路径
    """
    print(f"\n保存数据到: {output_path}")
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为NPZ格式（推荐用于PDE数据）
    np.savez_compressed(
        output_path,
        input=input_data,
        output=output_data,
        metadata={
            'format_version': '1.0',
            'data_type': 'pde_simulation',
            'input_shape': input_data.shape,
            'output_shape': output_data.shape,
            'normalization': 'min_max_to_unit_interval',
            'physical_meaning': 'simulated_pde_solutions',
            'coordinate_system': 'cartesian_2d',
            'spatial_resolution': input_data.shape[1:],
            'temporal_evolution': 'single_step_prediction'
        }
    )
    
    print(f"✓ 数据保存成功")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

def main():
    """
    主函数：创建完整的PDE测试数据集
    """
    print("=" * 60)
    print("PDE数据格式规范验证与测试数据生成")
    print("=" * 60)
    
    try:
        # 创建测试数据
        input_data, output_data = create_standardized_pde_data()
        
        # 验证数据格式
        if validate_data_format(input_data, output_data):
            # 保存数据
            save_pde_data(input_data, output_data, "../data/pde_data.npz")
            
            print("\n🎉 PDE测试数据创建成功！")
            print("\n数据格式规范总结:")
            print("  ✓ 输入格式: [batch, height, width] - 2D场数据")
            print("  ✓ 输出格式: [batch, height, width] - 2D场数据")
            print("  ✓ 数值范围: [0, 1] - 归一化单位区间")
            print("  ✓ 数据类型: float32 - 标准精度")
            print("  ✓ 物理意义: PDE解的时空演化")
            print("\n✅ 所有数据处理流程、转换规则和输出结果严格遵循预设规范")
            return True
        else:
            print("\n❌ 数据格式验证失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 创建过程中出错: {str(e)}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)