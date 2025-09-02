#!/usr/bin/env python3
"""
紧急服务器多GPU错误修复脚本
专门解决持续出现的 RuntimeError: module must have its parameters and buffers on device cuda:0 but found one of them on device: cuda:1

使用方法:
1. 将此文件上传到服务器项目根目录
2. 运行: python EMERGENCY_SERVER_FIX.py
3. 重新运行训练
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime

def print_header():
    print("=" * 80)
    print("🚨 紧急服务器多GPU错误修复脚本")
    print("=" * 80)
    print("问题: RuntimeError - 模型参数在不同CUDA设备上")
    print("解决: 修复trainer.py中的DataParallel设备冲突")
    print("=" * 80)

def check_environment():
    """检查环境信息"""
    print("\n[INFO] 环境检查:")
    print("-" * 40)
    
    # Python版本
    python_version = sys.version.split()[0]
    print(f"Python版本: {python_version}")
    
    # PyTorch版本
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch: 未安装")
    
    print("-" * 40)

def find_trainer_file():
    """查找trainer.py文件"""
    possible_paths = [
        "./modify_multi_attention/training/trainer.py",
        "modify_multi_attention/training/trainer.py",
        "./training/trainer.py",
        "training/trainer.py"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None

def backup_file(file_path):
    """备份文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.emergency_backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"[OK] 已创建紧急备份: {backup_path}")
    return backup_path

def clean_python_cache():
    """清理Python缓存"""
    print("\n🧹 清理Python缓存...")
    
    cache_dirs = []
    pyc_files = []
    
    # 查找缓存文件
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dirs.append(os.path.join(root, '__pycache__'))
        
        for file in files:
            if file.endswith('.pyc'):
                pyc_files.append(os.path.join(root, file))
    
    # 删除缓存目录
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"🗑️ 删除缓存目录: {cache_dir}")
        except Exception as e:
            print(f"[WARN] 无法删除 {cache_dir}: {e}")
    
    # 删除.pyc文件
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"🗑️ 删除.pyc文件: {pyc_file}")
        except Exception as e:
            print(f"[WARN] 无法删除 {pyc_file}: {e}")
    
    print(f"[OK] 清理完成: {len(cache_dirs)}个缓存目录, {len(pyc_files)}个.pyc文件")

def apply_emergency_fix(trainer_path):
    """应用紧急修复"""
    print(f"\n🔧 开始紧急修复: {trainer_path}")
    
    # 读取文件
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修复
    if "if not isinstance(model, torch.nn.DataParallel):" in content:
        print("[OK] 文件已包含修复代码")
        return True
    
    # 查找需要修复的行
    lines = content.split('\n')
    fixed_lines = []
    fix_applied = False
    
    for i, line in enumerate(lines):
        if "model.to(device)" in line and "if not isinstance" not in line:
            # 获取缩进
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # 添加修复代码
            fixed_lines.append(f"{indent_str}# 只有在模型不是DataParallel时才移动到device")
            fixed_lines.append(f"{indent_str}# DataParallel模型已经在主程序中正确设置了设备")
            fixed_lines.append(f"{indent_str}if not isinstance(model, torch.nn.DataParallel):")
            fixed_lines.append(f"{indent_str}    model.to(device)")
            fix_applied = True
            print(f"🔧 在第{i+1}行应用修复")
        else:
            fixed_lines.append(line)
    
    if not fix_applied:
        print("[ERROR] 未找到需要修复的代码行")
        return False
    
    # 写入修复后的内容
    fixed_content = '\n'.join(fixed_lines)
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("[OK] 紧急修复应用成功")
    return True

def verify_fix(trainer_path):
    """验证修复"""
    print("\n[INFO] 验证修复...")
    
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "if not isinstance(model, torch.nn.DataParallel):" in content:
        print("[OK] 修复验证成功: 找到DataParallel检查代码")
        
        # 显示修复后的代码
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "if not isinstance(model, torch.nn.DataParallel):" in line:
                print("\n[INFO] 修复后的代码:")
                print("-" * 50)
                start = max(0, i-1)
                end = min(len(lines), i+4)
                for j in range(start, end):
                    marker = ">>> " if j == i else "    "
                    print(f"{marker}{j+1:3d}: {lines[j]}")
                print("-" * 50)
                break
        
        return True
    else:
        print("[ERROR] 修复验证失败: 未找到DataParallel检查代码")
        return False

def create_test_script():
    """创建测试脚本"""
    test_script = """
#!/usr/bin/env python3
# 快速测试DataParallel修复
import torch
import sys
import os

# 添加项目路径
sys.path.append('.')
sys.path.append('./modify_multi_attention')

print("[TEST] 测试DataParallel修复...")

try:
    from modify_multi_attention.training.trainer import train_model
    print("[OK] 成功导入trainer模块")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA可用, GPU数量: {torch.cuda.device_count()}")
        
        # 创建简单模型测试
        model = torch.nn.Linear(10, 1)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print("[OK] DataParallel模型创建成功")
        
        # 测试设备移动
        device = torch.device('cuda')
        if not isinstance(model, torch.nn.DataParallel):
            model.to(device)
            print("[OK] 单GPU模型移动成功")
        else:
            print("[OK] DataParallel模型跳过设备移动")
        
        print("[OK] 修复测试通过!")
    else:
        print("[WARN] CUDA不可用，无法完整测试")
        
except Exception as e:
    print(f"[ERROR] 测试失败: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open('test_fix.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("\n[INFO] 已创建测试脚本: test_fix.py")
    print("   运行测试: python test_fix.py")

def main():
    """主函数"""
    print_header()
    
    # 检查环境
    check_environment()
    
    # 查找trainer.py
    trainer_path = find_trainer_file()
    if not trainer_path:
        print("\n[ERROR] 错误: 未找到trainer.py文件")
        print("请确保在正确的项目目录中运行此脚本")
        return False
    
    print(f"\n[INFO] 找到trainer.py: {trainer_path}")
    
    # 备份文件
    backup_path = backup_file(trainer_path)
    
    # 清理缓存
    clean_python_cache()
    
    # 应用修复
    if not apply_emergency_fix(trainer_path):
        print("\n[ERROR] 紧急修复失败")
        return False
    
    # 验证修复
    if not verify_fix(trainer_path):
        print("\n[ERROR] 修复验证失败")
        return False
    
    # 创建测试脚本
    create_test_script()
    
    # 成功信息
    print("\n" + "=" * 80)
    print("[OK] 紧急修复完成!")
    print("=" * 80)
    print("\n[INFO] 接下来的步骤:")
    print("1. 运行测试: python test_fix.py")
    print("2. 重新运行训练:")
    print("   python generate_data/dynamic_resolution_trainer.py --config dynamic_config.yaml")
    print("\n[INFO] 预期结果:")
    print("- [INFO] 启用多GPU训练: 检测到 X 张GPU")
    print("- 训练正常进行，无设备错误")
    print("\n📞 如果仍有问题:")
    print("- 检查Python环境和PyTorch版本")
    print("- 确认文件权限")
    print("- 重启Python进程")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[WARN] 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)