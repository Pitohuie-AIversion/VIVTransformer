#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU训练设备错误修复脚本

这个脚本用于修复多GPU训练中的设备不匹配错误：
RuntimeError: module must have its parameters and buffers on device cuda:0 
(device_ids[0]) but found one of them on device: cuda:1

使用方法:
python fix_multi_gpu_error.py
"""

import os
import sys
from pathlib import Path

def fix_trainer_file():
    """修复trainer.py文件中的设备设置问题"""
    
    # 获取trainer.py文件路径
    trainer_path = Path("modify_multi_attention/training/trainer.py")
    
    if not trainer_path.exists():
        print(f"[ERROR] 找不到文件: {trainer_path}")
        return False
    
    # 读取原文件内容
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修复
    if "if not isinstance(model, torch.nn.DataParallel):" in content:
        print("[OK] trainer.py 已经修复，无需重复操作")
        return True
    
    # 查找需要替换的代码
    old_code = "    model.to(device)"
    new_code = """    # 只有在模型不是DataParallel时才移动到device
    # DataParallel模型已经在主程序中正确设置了设备
    if not isinstance(model, torch.nn.DataParallel):
        model.to(device)"""
    
    if old_code not in content:
        print("[ERROR] 未找到需要修复的代码段")
        return False
    
    # 执行替换
    fixed_content = content.replace(old_code, new_code)
    
    # 备份原文件
    backup_path = trainer_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[INFO] 原文件已备份到: {backup_path}")
    
    # 写入修复后的内容
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"[OK] 已修复 {trainer_path}")
    return True

def verify_fix():
    """验证修复是否成功"""
    trainer_path = Path("modify_multi_attention/training/trainer.py")
    
    if not trainer_path.exists():
        return False
    
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查修复后的代码是否存在
    if "if not isinstance(model, torch.nn.DataParallel):" in content:
        print("[OK] 修复验证成功")
        return True
    else:
        print("[ERROR] 修复验证失败")
        return False

def main():
    print("=== 多GPU训练设备错误修复脚本 ===")
    print()
    
    # 检查当前目录
    if not Path("modify_multi_attention").exists():
        print("[ERROR] 请在项目根目录下运行此脚本")
        sys.exit(1)
    
    # 执行修复
    print("🔧 开始修复trainer.py...")
    if fix_trainer_file():
        print()
        print("[INFO] 验证修复结果...")
        if verify_fix():
            print()
            print("[OK] 修复完成！现在可以正常运行多GPU训练了")
            print()
            print("[INFO] 修复说明:")
            print("   - 修复了trainer.py中的设备设置逻辑")
            print("   - DataParallel模型不会被重新移动到单个设备")
            print("   - 保持了多GPU的正确设备分布")
            print()
            print("[INFO] 现在可以运行多GPU训练:")
            print("   python dynamic_resolution_trainer.py --use_dataparallel --config dynamic_config.yaml")
        else:
            print("[ERROR] 修复验证失败，请检查文件内容")
            sys.exit(1)
    else:
        print("[ERROR] 修复失败")
        sys.exit(1)

if __name__ == "__main__":
    main()