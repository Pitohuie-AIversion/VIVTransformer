#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU修复验证脚本

用于验证多GPU训练设备错误是否已修复
"""

import torch
import sys
from pathlib import Path

def check_gpu_availability():
    """检查GPU可用性"""
    print("=== GPU环境检查 ===")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"[OK] 检测到 {gpu_count} 张GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return gpu_count > 1

def check_trainer_fix():
    """检查trainer.py是否已修复"""
    print("\n=== 代码修复检查 ===")
    
    trainer_path = Path("modify_multi_attention/training/trainer.py")
    
    if not trainer_path.exists():
        print(f"[ERROR] 找不到文件: {trainer_path}")
        return False
    
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "if not isinstance(model, torch.nn.DataParallel):" in content:
        print("[OK] trainer.py 已正确修复")
        return True
    else:
        print("[ERROR] trainer.py 尚未修复")
        print("   请运行: python fix_multi_gpu_error.py")
        return False

def test_dataparallel():
    """测试DataParallel功能"""
    print("\n=== DataParallel功能测试 ===")
    
    try:
        # 创建简单的测试模型
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # 测试单GPU模式
        device = torch.device('cuda:0')
        model = TestModel().to(device)
        print("[OK] 单GPU模式正常")
        
        # 测试多GPU模式
        if torch.cuda.device_count() > 1:
            model_parallel = torch.nn.DataParallel(model)
            test_input = torch.randn(4, 10).to(device)
            output = model_parallel(test_input)
            print("[OK] DataParallel模式正常")
            return True
        else:
            print("[WARN] 只有1张GPU，跳过DataParallel测试")
            return True
            
    except Exception as e:
        print(f"[ERROR] DataParallel测试失败: {e}")
        return False

def check_config():
    """检查配置文件"""
    print("\n=== 配置文件检查 ===")
    
    config_path = Path("generate_data/dynamic_config.yaml")
    
    if not config_path.exists():
        print(f"[ERROR] 找不到配置文件: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "use_dataparallel: true" in content:
        print("[OK] 配置文件已启用多GPU训练")
        return True
    elif "use_dataparallel: false" in content:
        print("[WARN] 配置文件中多GPU训练未启用")
        print("   建议设置: use_dataparallel: true")
        return False
    else:
        print("[ERROR] 配置文件中未找到use_dataparallel设置")
        return False

def main():
    print("多GPU修复验证脚本")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("modify_multi_attention").exists():
        print("[ERROR] 请在项目根目录下运行此脚本")
        sys.exit(1)
    
    all_checks_passed = True
    
    # 1. 检查GPU环境
    gpu_available = check_gpu_availability()
    if not gpu_available:
        print("\n[WARN] 多GPU环境不可用，但单GPU训练应该正常工作")
    
    # 2. 检查代码修复
    trainer_fixed = check_trainer_fix()
    if not trainer_fixed:
        all_checks_passed = False
    
    # 3. 测试DataParallel
    if gpu_available:
        dataparallel_ok = test_dataparallel()
        if not dataparallel_ok:
            all_checks_passed = False
    
    # 4. 检查配置
    config_ok = check_config()
    if not config_ok:
        all_checks_passed = False
    
    # 总结
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("[OK] 所有检查通过！多GPU训练应该可以正常工作")
        print("\n[INFO] 现在可以运行多GPU训练:")
        print("   python generate_data/dynamic_resolution_trainer.py --use_dataparallel --config generate_data/dynamic_config.yaml")
    else:
        print("[ERROR] 部分检查未通过，请根据上述提示进行修复")
        print("\n[INFO] 修复步骤:")
        if not trainer_fixed:
            print("   1. 运行: python fix_multi_gpu_error.py")
        if not config_ok:
            print("   2. 在dynamic_config.yaml中设置: use_dataparallel: true")

if __name__ == "__main__":
    main()