#!/usr/bin/env python3
"""
服务器多GPU错误修复脚本
用于在服务器上直接修复trainer.py文件中的设备冲突问题

使用方法:
1. 将此脚本上传到服务器
2. 在服务器上运行: python server_fix_multi_gpu.py
3. 重新运行训练脚本
"""

import os
import sys
import shutil
from datetime import datetime

def backup_file(file_path):
    """备份原文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"[OK] 已备份原文件: {backup_path}")
    return backup_path

def fix_trainer_file(trainer_path):
    """修复trainer.py文件"""
    print(f"🔧 开始修复文件: {trainer_path}")
    
    if not os.path.exists(trainer_path):
        print(f"[ERROR] 文件不存在: {trainer_path}")
        return False
    
    # 备份原文件
    backup_path = backup_file(trainer_path)
    
    try:
        # 读取文件内容
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经修复
        if "if not isinstance(model, torch.nn.DataParallel):" in content:
            print("[OK] 文件已经包含修复代码，无需重复修复")
            return True
        
        # 查找需要修复的位置
        target_line = "model.to(device)"
        if target_line not in content:
            print(f"[ERROR] 未找到目标代码行: {target_line}")
            return False
        
        # 应用修复
        fixed_content = content.replace(
            "    model.to(device)",
            "    # 只有在模型不是DataParallel时才移动到device\n    # DataParallel模型已经在主程序中正确设置了设备\n    if not isinstance(model, torch.nn.DataParallel):\n        model.to(device)"
        )
        
        # 写入修复后的内容
        with open(trainer_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("[OK] 文件修复完成")
        return True
        
    except Exception as e:
        print(f"[ERROR] 修复过程中出错: {e}")
        # 恢复备份
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, trainer_path)
            print(f"[INFO] 已恢复备份文件")
        return False

def verify_fix(trainer_path):
    """验证修复是否成功"""
    try:
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "if not isinstance(model, torch.nn.DataParallel):" in content:
            print("[OK] 修复验证成功: 已包含DataParallel检查代码")
            return True
        else:
            print("[ERROR] 修复验证失败: 未找到DataParallel检查代码")
            return False
    except Exception as e:
        print(f"[ERROR] 验证过程中出错: {e}")
        return False

def clean_python_cache():
    """清理Python缓存文件"""
    print("🧹 清理Python缓存文件...")
    
    # 查找并删除__pycache__目录
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"🗑️ 已删除: {cache_dir}")
            except Exception as e:
                print(f"[WARN] 无法删除 {cache_dir}: {e}")
    
    # 删除.pyc文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    print(f"🗑️ 已删除: {pyc_file}")
                except Exception as e:
                    print(f"[WARN] 无法删除 {pyc_file}: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 服务器多GPU错误修复脚本")
    print("=" * 60)
    
    # 查找trainer.py文件
    possible_paths = [
        "./modify_multi_attention/training/trainer.py",
        "./training/trainer.py",
        "../modify_multi_attention/training/trainer.py",
        "modify_multi_attention/training/trainer.py"
    ]
    
    trainer_path = None
    for path in possible_paths:
        if os.path.exists(path):
            trainer_path = path
            break
    
    if not trainer_path:
        print("[ERROR] 未找到trainer.py文件")
        print("请确保在正确的项目目录中运行此脚本")
        return False
    
    print(f"[INFO] 找到trainer.py文件: {trainer_path}")
    
    # 清理Python缓存
    clean_python_cache()
    
    # 修复文件
    if fix_trainer_file(trainer_path):
        # 验证修复
        if verify_fix(trainer_path):
            print("\n" + "=" * 60)
            print("[OK] 修复完成！")
            print("=" * 60)
            print("\n[INFO] 接下来的步骤:")
            print("1. 重新运行训练脚本")
            print("2. 确认多GPU训练正常工作")
            print("\n[INFO] 预期输出:")
            print("- [INFO] 启用多GPU训练: 检测到 X 张GPU")
            print("- 训练正常进行，无设备错误")
            return True
        else:
            print("[ERROR] 修复验证失败")
            return False
    else:
        print("[ERROR] 修复失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)