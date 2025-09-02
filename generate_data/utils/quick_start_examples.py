#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态分辨率训练器快速开始示例

这个脚本展示了如何使用dynamic_resolution_trainer.py进行不同分辨率的训练
支持命令行参数和YAML配置文件两种方式
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"[INFO] {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] 执行成功!")
            # 只显示最后几行输出
            lines = result.stdout.strip().split('\n')
            if len(lines) > 5:
                print("...")
                for line in lines[-5:]:
                    print(line)
            else:
                print(result.stdout)
        else:
            print("[ERROR] 执行失败!")
            print(f"错误: {result.stderr}")
    except Exception as e:
        print(f"[ERROR] 执行异常: {e}")

def main():
    """主函数 - 运行各种示例"""
    print("[INFO] 动态分辨率训练器使用示例")
    print("这些示例展示了如何使用不同的配置进行训练")
    
    # 确保在正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    examples = [
        {
            "cmd": "python dynamic_resolution_trainer.py --input_resolution 16 16 --output_resolution 64 64 --epochs 1 --num_samples 20 --batch_size 4",
            "desc": "示例1: 16x16 → 64x64 快速训练"
        },
        {
            "cmd": "python dynamic_resolution_trainer.py --input_resolution 32 32 --output_resolution 128 128 --epochs 1 --num_samples 30 --attention_type cbam",
            "desc": "示例2: 32x32 → 128x128 使用CBAM注意力"
        },
        {
            "cmd": "python dynamic_resolution_trainer.py --config dynamic_config.yaml --epochs 1 --num_samples 25",
            "desc": "示例3: 使用YAML配置文件"
        },
        {
            "cmd": "python dynamic_resolution_trainer.py --input_resolution 24 24 --output_resolution 96 96 --epochs 1 --num_samples 15 --crop_mode random",
            "desc": "示例4: 非标准分辨率 + 随机裁剪"
        }
    ]
    
    print("\n选择要运行的示例:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['desc']}")
    print("5. 运行所有示例")
    print("0. 退出")
    
    try:
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == '0':
            print("[INFO] 退出程序")
            return
        elif choice == '5':
            print("[INFO] 运行所有示例...")
            for example in examples:
                run_command(example['cmd'], example['desc'])
        elif choice in ['1', '2', '3', '4']:
            idx = int(choice) - 1
            run_command(examples[idx]['cmd'], examples[idx]['desc'])
        else:
            print("[ERROR] 无效选择")
            
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断程序")
    except Exception as e:
        print(f"[ERROR] 程序异常: {e}")

if __name__ == "__main__":
    main()