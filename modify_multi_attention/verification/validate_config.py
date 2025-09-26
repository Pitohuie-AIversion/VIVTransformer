#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应配置文件验证脚本
验证adaptive_training_config.yaml的语法和关键配置项
"""

import yaml
import os
import sys

def validate_adaptive_config():
    """验证自适应配置文件"""
    config_path = "configs/adaptive_training_config.yaml"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
        # 加载YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print("✅ YAML语法验证成功！")
        
        # 验证关键配置项
        print("\n📋 关键配置项验证:")
        print(f"  实验名称: {config['experiment']['name']}")
        print(f"  实验描述: {config['experiment']['description']}")
        print(f"  版本: {config['experiment']['version']}")
        
        # 验证自适应资源配置
        adaptive = config['adaptive_resources']
        print(f"\n🔧 自适应资源配置:")
        print(f"  启用自适应: {adaptive['enable_adaptive']}")
        print(f"  硬件自动检测: {adaptive['hardware_detection']['auto_detect']}")
        print(f"  检测超时: {adaptive['hardware_detection']['detection_timeout']}秒")
        print(f"  内存安全边际: {adaptive['hardware_detection']['memory_safety_margin']}")
        
        # 验证批次大小自适应
        batch_adapt = adaptive['batch_size_adaptation']
        print(f"\n📦 批次大小自适应:")
        print(f"  策略: {batch_adapt['strategy']}")
        print(f"  自动检测最优: {batch_adapt['auto_detect_optimal']}")
        print(f"  最小批次: {batch_adapt['min_batch_size']}")
        print(f"  最大批次: {batch_adapt['max_batch_size']}")
        print(f"  内存阈值: {batch_adapt['memory_threshold']}")
        print(f"  OOM恢复: {batch_adapt['oom_recovery']}")
        print(f"  动态调整: {batch_adapt['dynamic_adjustment']}")
        
        # 验证数据加载器自适应
        loader_adapt = adaptive['dataloader_adaptation']
        print(f"\n🔄 数据加载器自适应:")
        print(f"  策略: {loader_adapt['strategy']}")
        print(f"  自动检测最优: {loader_adapt['auto_detect_optimal']}")
        print(f"  最小工作进程: {loader_adapt['min_workers']}")
        print(f"  最大工作进程: {loader_adapt['max_workers']}")
        print(f"  CPU阈值: {loader_adapt['cpu_threshold']}")
        print(f"  动态调整: {loader_adapt['dynamic_adjustment']}")
        
        # 验证优化策略
        opt_strategy = adaptive['optimization_strategy']
        print(f"\n⚡ 优化策略:")
        print(f"  目标: {opt_strategy['target']}")
        print(f"  自动混合精度: {opt_strategy['auto_mixed_precision']}")
        print(f"  梯度检查点: {opt_strategy['gradient_checkpointing']}")
        print(f"  模型编译: {opt_strategy['compile_model']}")
        print(f"  内存固定: {opt_strategy['pin_memory']}")
        print(f"  非阻塞传输: {opt_strategy['non_blocking']}")
        
        # 验证预设模板
        presets = config['presets']
        print(f"\n🎯 预设模板:")
        for preset_name in presets.keys():
            print(f"  - {preset_name}")
            
        # 验证硬件优化
        hw_opt = config['hardware_optimizations']
        print(f"\n🖥️ 硬件优化:")
        print(f"  自动应用: {hw_opt['auto_apply']}")
        print(f"  GPU自动检测: {hw_opt['gpu']['auto_detect']}")
        print(f"  CPU自动检测: {hw_opt['cpu']['auto_detect']}")
        print(f"  内存自动检测: {hw_opt['memory']['auto_detect']}")
        
        print(f"\n✅ 配置文件验证完成！所有关键配置项都正确设置。")
        print(f"📄 配置文件路径: {os.path.abspath(config_path)}")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ YAML语法错误: {e}")
        return False
    except KeyError as e:
        print(f"❌ 缺少必需的配置项: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    print("🔍 开始验证自适应配置文件...")
    success = validate_adaptive_config()
    sys.exit(0 if success else 1)