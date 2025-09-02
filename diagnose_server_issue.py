#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器多GPU问题诊断脚本

用于诊断服务器上的多GPU训练问题
"""

import os
import sys
import torch
from pathlib import Path

def check_file_content():
    """检查关键文件内容"""
    print("=== 文件内容检查 ===")
    
    trainer_path = Path("modify_multi_attention/training/trainer.py")
    
    if not trainer_path.exists():
        print(f"[ERROR] 找不到文件: {trainer_path}")
        return False
    
    print(f"[INFO] 检查文件: {trainer_path}")
    
    with open(trainer_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找关键代码行
    found_fix = False
    found_old_code = False
    
    for i, line in enumerate(lines, 1):
        if "if not isinstance(model, torch.nn.DataParallel):" in line:
            found_fix = True
            print(f"[OK] 第{i}行: 找到修复代码")
            # 显示上下文
            start = max(0, i-3)
            end = min(len(lines), i+3)
            print("   上下文:")
            for j in range(start, end):
                marker = ">>> " if j == i-1 else "    "
                print(f"   {marker}{j+1:3d}: {lines[j].rstrip()}")
            break
        elif line.strip() == "model.to(device)":
            found_old_code = True
            print(f"[ERROR] 第{i}行: 发现未修复的代码")
            print(f"   {i:3d}: {line.rstrip()}")
    
    if found_fix:
        print("[OK] 代码已正确修复")
        return True
    elif found_old_code:
        print("[ERROR] 代码尚未修复")
        return False
    else:
        print("[WARN] 未找到相关代码行")
        return False

def show_fix_instructions():
    """显示修复指令"""
    print("\n=== 手动修复指令 ===")
    print("在服务器上执行以下步骤:")
    print()
    print("1. 编辑文件:")
    print("   vim modify_multi_attention/training/trainer.py")
    print()
    print("2. 找到这一行 (大约第32行):")
    print("   model.to(device)")
    print()
    print("3. 替换为:")
    print("   # 只有在模型不是DataParallel时才移动到device")
    print("   # DataParallel模型已经在主程序中正确设置了设备")
    print("   if not isinstance(model, torch.nn.DataParallel):")
    print("       model.to(device)")
    print()
    print("4. 保存文件并退出")
    print()
    print("5. 验证修复:")
    print("   python verify_multi_gpu_fix.py")

def create_patch_file():
    """创建补丁文件"""
    print("\n=== 创建补丁文件 ===")
    
    patch_content = '''--- a/modify_multi_attention/training/trainer.py
+++ b/modify_multi_attention/training/trainer.py
@@ -29,7 +29,10 @@ def train_model(model, train_loader, valid_loader, test_loader, criterion, opti
 
     print(f"[INFO] 早停配置: 启用={enable_early_stopping}, 监控={monitor}, 模式={mode}, 耐心值={patience}")
 
-    model.to(device)
+    # 只有在模型不是DataParallel时才移动到device
+    # DataParallel模型已经在主程序中正确设置了设备
+    if not isinstance(model, torch.nn.DataParallel):
+        model.to(device)
 
     train_loss_history = []
     valid_loss_history = []
'''
    
    patch_file = Path("multi_gpu_fix.patch")
    with open(patch_file, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print(f"[OK] 补丁文件已创建: {patch_file}")
    print("\n在服务器上应用补丁:")
    print(f"   patch -p1 < {patch_file}")
    print("或者:")
    print(f"   git apply {patch_file}")

def check_environment():
    """检查环境信息"""
    print("\n=== 环境信息 ===")
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"当前工作目录: {os.getcwd()}")

def create_server_fix_script():
    """创建服务器修复脚本"""
    print("\n=== 创建服务器修复脚本 ===")
    
    script_content = '''#!/bin/bash
# 服务器多GPU修复脚本

echo "=== 多GPU训练设备错误修复 ==="
echo

# 检查文件是否存在
if [ ! -f "modify_multi_attention/training/trainer.py" ]; then
    echo "[ERROR] 找不到trainer.py文件"
    exit 1
fi

# 备份原文件
cp modify_multi_attention/training/trainer.py modify_multi_attention/training/trainer.py.backup
echo "[INFO] 已备份原文件"

# 执行替换
sed -i 's/^    model\.to(device)$/    # 只有在模型不是DataParallel时才移动到device\n    # DataParallel模型已经在主程序中正确设置了设备\n    if not isinstance(model, torch.nn.DataParallel):\n        model.to(device)/' modify_multi_attention/training/trainer.py

echo "[OK] 修复完成"
echo
echo "[INFO] 验证修复结果:"
if grep -q "if not isinstance(model, torch.nn.DataParallel):" modify_multi_attention/training/trainer.py; then
    echo "[OK] 修复验证成功"
else
    echo "[ERROR] 修复验证失败"
    echo "请手动修复或使用备份文件恢复"
fi

echo
echo "[INFO] 现在可以运行多GPU训练:"
echo "python generate_data/dynamic_resolution_trainer.py --use_dataparallel --config generate_data/dynamic_config.yaml"
'''
    
    script_file = Path("fix_server_multi_gpu.sh")
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_file, 0o755)
    
    print(f"[OK] 服务器修复脚本已创建: {script_file}")
    print("\n在服务器上运行:")
    print(f"   chmod +x {script_file}")
    print(f"   ./{script_file}")

def main():
    print("服务器多GPU问题诊断脚本")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("modify_multi_attention").exists():
        print("[ERROR] 请在项目根目录下运行此脚本")
        sys.exit(1)
    
    # 检查环境
    check_environment()
    
    # 检查文件内容
    is_fixed = check_file_content()
    
    if not is_fixed:
        # 显示修复指令
        show_fix_instructions()
        
        # 创建补丁文件
        create_patch_file()
        
        # 创建服务器修复脚本
        create_server_fix_script()
        
        print("\n" + "=" * 50)
        print("[INFO] 修复选项:")
        print("1. 手动编辑 (推荐)")
        print("2. 使用补丁文件")
        print("3. 使用自动脚本")
        print("\n请选择适合的方法在服务器上修复代码")
    else:
        print("\n" + "=" * 50)
        print("[OK] 本地代码已正确修复")
        print("\n如果服务器上仍有问题，请确保:")
        print("1. 代码已正确同步到服务器")
        print("2. 服务器上的Python环境正确")
        print("3. 没有缓存的.pyc文件干扰")
        print("\n建议在服务器上运行:")
        print("   find . -name '*.pyc' -delete")
        print("   python -c \"import modify_multi_attention.training.trainer; print('导入成功')\"")

if __name__ == "__main__":
    main()