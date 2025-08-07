#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速配置工具

提供简单的命令行界面来快速管理和生成配置文件
支持常用的配置操作，无需复杂的命令行参数

作者: AI Assistant
日期: 2025
"""

import os
import sys
import json
from pathlib import Path
from config_manager import ConfigManager

def print_banner():
    """
    打印工具横幅
    """
    print("\n" + "="*70)
    print("🚀 快速配置工具 - VIVTransformer 配置管理")
    print("="*70)
    print("功能: 快速生成、修改和管理训练配置文件")
    print("支持: 参数复用、模板生成、批量实验配置")
    print("="*70)

def show_menu():
    """
    显示主菜单
    """
    print("\n📋 请选择操作:")
    print("1. 📊 查看配置摘要")
    print("2. 🔧 快速修改配置")
    print("3. 🎯 使用分辨率预设")
    print("4. ⚡ 硬件优化配置")
    print("5. 🧪 生成实验配置")
    print("6. 🔍 比较配置文件")
    print("7. ✅ 验证配置文件")
    print("8. 📖 运行使用示例")
    print("9. ❓ 帮助信息")
    print("0. 🚪 退出")
    print("-" * 50)

def get_config_file():
    """
    获取配置文件路径
    """
    default_configs = [
        "dynamic_config_server_downsampling_optimized.yaml",
        "dynamic_config_server_downsampling.yaml"
    ]
    
    print("\n📁 选择配置文件:")
    available_configs = []
    
    for i, config_file in enumerate(default_configs):
        if Path(config_file).exists():
            available_configs.append(config_file)
            print(f"{i+1}. {config_file} ✅")
        else:
            print(f"{i+1}. {config_file} ❌ (不存在)")
    
    # 查找其他YAML文件
    yaml_files = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.yml'))
    other_configs = [f for f in yaml_files if f.name not in default_configs]
    
    if other_configs:
        print("\n其他YAML文件:")
        for i, config_file in enumerate(other_configs):
            available_configs.append(str(config_file))
            print(f"{len(default_configs) + i + 1}. {config_file.name}")
    
    print(f"{len(available_configs) + 1}. 手动输入路径")
    
    while True:
        try:
            choice = input("\n请选择配置文件 (输入数字): ").strip()
            
            if not choice:
                continue
            
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(available_configs):
                return available_configs[choice_num - 1]
            elif choice_num == len(available_configs) + 1:
                custom_path = input("请输入配置文件路径: ").strip()
                if Path(custom_path).exists():
                    return custom_path
                else:
                    print("❌ 文件不存在，请重新选择")
            else:
                print("❌ 无效选择，请重新输入")
        
        except ValueError:
            print("❌ 请输入有效数字")
        except KeyboardInterrupt:
            print("\n👋 操作取消")
            return None

def action_view_summary():
    """
    查看配置摘要
    """
    print("\n📊 查看配置摘要")
    print("-" * 30)
    
    config_file = get_config_file()
    if not config_file:
        return
    
    try:
        config_manager = ConfigManager(config_file)
        config_manager.print_config_summary()
        
        # 验证配置
        print("\n🔍 配置验证:")
        is_valid = config_manager.validate_config()
        
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_quick_modify():
    """
    快速修改配置
    """
    print("\n🔧 快速修改配置")
    print("-" * 30)
    
    config_file = get_config_file()
    if not config_file:
        return
    
    try:
        config_manager = ConfigManager(config_file)
        
        print("\n常用修改选项:")
        print("1. 批次大小 (batch_size)")
        print("2. 学习率 (learning_rate)")
        print("3. 训练轮数 (epochs)")
        print("4. 样本数量 (num_samples)")
        print("5. 模型层数 (num_layers)")
        print("6. 模型维度 (d_model)")
        print("7. 自定义修改")
        
        choice = input("\n请选择要修改的参数 (1-7): ").strip()
        
        overrides = {}
        
        if choice == '1':
            batch_size = int(input("请输入新的批次大小: "))
            overrides['data.batch_size'] = batch_size
        
        elif choice == '2':
            learning_rate = float(input("请输入新的学习率: "))
            overrides['training.learning_rate'] = learning_rate
        
        elif choice == '3':
            epochs = int(input("请输入新的训练轮数: "))
            overrides['training.epochs'] = epochs
        
        elif choice == '4':
            num_samples = int(input("请输入新的样本数量: "))
            overrides['data.num_samples'] = num_samples
        
        elif choice == '5':
            num_layers = int(input("请输入新的模型层数: "))
            overrides['model.num_layers'] = num_layers
        
        elif choice == '6':
            d_model = int(input("请输入新的模型维度: "))
            overrides['model.d_model'] = d_model
        
        elif choice == '7':
            print("\n自定义修改格式: 参数路径=值")
            print("例如: data.batch_size=32")
            print("多个参数用逗号分隔: data.batch_size=32,training.learning_rate=0.001")
            
            custom_input = input("\n请输入修改参数: ").strip()
            
            for param in custom_input.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试转换数据类型
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # 保持字符串类型
                        pass
                    
                    overrides[key] = value
        
        else:
            print("❌ 无效选择")
            return
        
        if overrides:
            print("\n🔧 应用修改:")
            for key, value in overrides.items():
                print(f"  {key}: {value}")
            
            new_config = config_manager.override_params(overrides)
            
            # 验证新配置
            print("\n🔍 验证修改后的配置...")
            is_valid = config_manager.validate_config(new_config)
            
            if is_valid:
                output_file = input("\n请输入输出文件名 (默认: config_modified.yaml): ").strip()
                if not output_file:
                    output_file = "config_modified.yaml"
                
                config_manager.save_config(output_file, new_config)
                print(f"✅ 修改后的配置已保存到: {output_file}")
            else:
                print("❌ 配置验证失败，请检查参数")
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_resolution_presets():
    """
    使用分辨率预设
    """
    print("\n🎯 使用分辨率预设")
    print("-" * 30)
    
    config_file = get_config_file()
    if not config_file:
        return
    
    try:
        config_manager = ConfigManager(config_file)
        
        if not config_manager.presets:
            print("❌ 当前配置文件中没有分辨率预设")
            return
        
        print("\n📋 可用的分辨率预设:")
        preset_list = list(config_manager.presets.keys())
        
        for i, preset_name in enumerate(preset_list):
            preset = config_manager.presets[preset_name]
            input_res = preset.get('input_resolution', 'N/A')
            output_res = preset.get('output_resolution', 'N/A')
            batch_size = preset.get('recommended_batch_size', 'N/A')
            print(f"{i+1}. {preset_name}: {input_res} -> {output_res} (批次: {batch_size})")
        
        choice = input("\n请选择分辨率预设 (输入数字): ").strip()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(preset_list):
                preset_name = preset_list[choice_num - 1]
                preset = config_manager.presets[preset_name]
                
                overrides = {
                    'data.input_resolution': preset['input_resolution'],
                    'data.output_resolution': preset['output_resolution']
                }
                
                if 'recommended_batch_size' in preset:
                    overrides['data.batch_size'] = preset['recommended_batch_size']
                
                print(f"\n🎯 应用 {preset_name.upper()} 分辨率预设:")
                for key, value in overrides.items():
                    print(f"  {key}: {value}")
                
                new_config = config_manager.override_params(overrides)
                
                # 验证配置
                is_valid = config_manager.validate_config(new_config)
                
                if is_valid:
                    output_file = f"config_{preset_name}_resolution.yaml"
                    config_manager.save_config(output_file, new_config)
                    print(f"✅ {preset_name.upper()} 分辨率配置已保存到: {output_file}")
                else:
                    print("❌ 配置验证失败")
            
            else:
                print("❌ 无效选择")
        
        except ValueError:
            print("❌ 请输入有效数字")
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_hardware_optimization():
    """
    硬件优化配置
    """
    print("\n⚡ 硬件优化配置")
    print("-" * 30)
    
    config_file = get_config_file()
    if not config_file:
        return
    
    try:
        config_manager = ConfigManager(config_file)
        
        print("\n💻 选择硬件类型:")
        print("1. 💻 笔记本电脑 (低配置)")
        print("2. 🖥️  工作站 (中等配置)")
        print("3. 🖥️  服务器 (高配置)")
        print("4. 🔧 自定义配置")
        
        choice = input("\n请选择硬件类型 (1-4): ").strip()
        
        hardware_configs = {
            '1': {
                'name': 'laptop',
                'display_name': '笔记本电脑',
                'overrides': {
                    'data.batch_size': 8,
                    'dataloader.num_workers': 2,
                    'dataloader.prefetch_factor': 2,
                    'model.num_layers': 4,
                    'model.d_model': 256,
                    'training.epochs': 50
                }
            },
            '2': {
                'name': 'workstation',
                'display_name': '工作站',
                'overrides': {
                    'data.batch_size': 32,
                    'dataloader.num_workers': 8,
                    'dataloader.prefetch_factor': 4,
                    'model.num_layers': 6,
                    'model.d_model': 512,
                    'training.epochs': 100
                }
            },
            '3': {
                'name': 'server',
                'display_name': '服务器',
                'overrides': {
                    'data.batch_size': 64,
                    'dataloader.num_workers': 16,
                    'dataloader.prefetch_factor': 8,
                    'model.num_layers': 8,
                    'model.d_model': 512,
                    'training.epochs': 200
                }
            }
        }
        
        if choice in hardware_configs:
            config = hardware_configs[choice]
            
            print(f"\n⚡ 应用 {config['display_name']} 优化配置:")
            for key, value in config['overrides'].items():
                print(f"  {key}: {value}")
            
            new_config = config_manager.override_params(config['overrides'])
            
            # 验证配置
            is_valid = config_manager.validate_config(new_config)
            
            if is_valid:
                output_file = f"config_optimized_for_{config['name']}.yaml"
                config_manager.save_config(output_file, new_config)
                print(f"✅ {config['display_name']} 优化配置已保存到: {output_file}")
            else:
                print("❌ 配置验证失败")
        
        elif choice == '4':
            print("\n🔧 自定义硬件配置")
            print("请依次输入以下参数 (直接回车使用默认值):")
            
            overrides = {}
            
            # 批次大小
            batch_size = input("批次大小 (默认: 16): ").strip()
            if batch_size:
                overrides['data.batch_size'] = int(batch_size)
            
            # 工作进程数
            num_workers = input("数据加载工作进程数 (默认: 4): ").strip()
            if num_workers:
                overrides['dataloader.num_workers'] = int(num_workers)
            
            # 模型层数
            num_layers = input("模型层数 (默认: 6): ").strip()
            if num_layers:
                overrides['model.num_layers'] = int(num_layers)
            
            # 模型维度
            d_model = input("模型维度 (默认: 512): ").strip()
            if d_model:
                overrides['model.d_model'] = int(d_model)
            
            # 训练轮数
            epochs = input("训练轮数 (默认: 100): ").strip()
            if epochs:
                overrides['training.epochs'] = int(epochs)
            
            if overrides:
                print("\n🔧 应用自定义配置:")
                for key, value in overrides.items():
                    print(f"  {key}: {value}")
                
                new_config = config_manager.override_params(overrides)
                
                # 验证配置
                is_valid = config_manager.validate_config(new_config)
                
                if is_valid:
                    output_file = "config_custom_optimized.yaml"
                    config_manager.save_config(output_file, new_config)
                    print(f"✅ 自定义优化配置已保存到: {output_file}")
                else:
                    print("❌ 配置验证失败")
            else:
                print("❌ 没有输入任何参数")
        
        else:
            print("❌ 无效选择")
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_generate_experiments():
    """
    生成实验配置
    """
    print("\n🧪 生成实验配置")
    print("-" * 30)
    
    config_file = get_config_file()
    if not config_file:
        return
    
    try:
        config_manager = ConfigManager(config_file)
        
        print("\n🔬 选择实验类型:")
        print("1. 🎯 批次大小实验")
        print("2. 📈 学习率实验")
        print("3. 🧠 模型结构实验")
        print("4. 🔧 自定义实验")
        
        choice = input("\n请选择实验类型 (1-4): ").strip()
        
        param_grids = {
            '1': {
                'name': '批次大小实验',
                'grid': {
                    'data.batch_size': [8, 16, 32, 64]
                }
            },
            '2': {
                'name': '学习率实验',
                'grid': {
                    'training.learning_rate': [0.0001, 0.0005, 0.001, 0.005]
                }
            },
            '3': {
                'name': '模型结构实验',
                'grid': {
                    'model.num_layers': [4, 6, 8],
                    'model.d_model': [256, 512, 768]
                }
            }
        }
        
        if choice in param_grids:
            experiment = param_grids[choice]
            param_grid = experiment['grid']
            
            print(f"\n🔬 {experiment['name']} 参数网格:")
            total_combinations = 1
            for param, values in param_grid.items():
                print(f"  {param}: {values}")
                total_combinations *= len(values)
            
            print(f"\n📊 总共将生成 {total_combinations} 个实验配置")
            
            confirm = input("\n是否继续生成? (y/n): ").strip().lower()
            if confirm == 'y':
                output_dir = f"experiment_{choice}"
                config_paths = config_manager.generate_experiment_configs(
                    config_manager.config, param_grid, output_dir
                )
                
                print(f"\n✅ 成功生成 {len(config_paths)} 个实验配置到目录: {output_dir}")
        
        elif choice == '4':
            print("\n🔧 自定义实验配置")
            print("格式: 参数路径:[值1,值2,值3]")
            print("例如: data.batch_size:[16,32,64]")
            print("多个参数用分号分隔")
            
            custom_input = input("\n请输入实验参数: ").strip()
            
            try:
                param_grid = {}
                
                for param_def in custom_input.split(';'):
                    if ':' in param_def:
                        param, values_str = param_def.split(':', 1)
                        param = param.strip()
                        
                        # 解析值列表
                        values_str = values_str.strip()
                        if values_str.startswith('[') and values_str.endswith(']'):
                            values_str = values_str[1:-1]
                        
                        values = []
                        for value in values_str.split(','):
                            value = value.strip()
                            try:
                                if '.' in value:
                                    values.append(float(value))
                                else:
                                    values.append(int(value))
                            except ValueError:
                                values.append(value)
                        
                        param_grid[param] = values
                
                if param_grid:
                    print("\n🔬 自定义实验参数网格:")
                    total_combinations = 1
                    for param, values in param_grid.items():
                        print(f"  {param}: {values}")
                        total_combinations *= len(values)
                    
                    print(f"\n📊 总共将生成 {total_combinations} 个实验配置")
                    
                    confirm = input("\n是否继续生成? (y/n): ").strip().lower()
                    if confirm == 'y':
                        output_dir = "experiment_custom"
                        config_paths = config_manager.generate_experiment_configs(
                            config_manager.config, param_grid, output_dir
                        )
                        
                        print(f"\n✅ 成功生成 {len(config_paths)} 个实验配置到目录: {output_dir}")
                else:
                    print("❌ 没有解析到有效的参数")
            
            except Exception as e:
                print(f"❌ 解析参数失败: {e}")
        
        else:
            print("❌ 无效选择")
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_compare_configs():
    """
    比较配置文件
    """
    print("\n🔍 比较配置文件")
    print("-" * 30)
    
    print("\n选择第一个配置文件:")
    config1 = get_config_file()
    if not config1:
        return
    
    print("\n选择第二个配置文件:")
    config2 = get_config_file()
    if not config2:
        return
    
    try:
        config_manager1 = ConfigManager(config1)
        config_manager2 = ConfigManager(config2)
        
        differences = config_manager1.compare_configs(
            config_manager1.config, config_manager2.config
        )
        
        print(f"\n📊 配置比较结果:")
        print(f"  仅在配置1中: {len(differences['only_in_config1'])} 项")
        print(f"  仅在配置2中: {len(differences['only_in_config2'])} 项")
        print(f"  值不同: {len(differences['different_values'])} 项")
        print(f"  值相同: {len(differences['same_values'])} 项")
        
        # 显示主要差异
        if differences['different_values']:
            print("\n🔍 主要差异 (前10项):")
            for i, (path, values) in enumerate(list(differences['different_values'].items())[:10]):
                print(f"  {i+1}. {path}:")
                print(f"     配置1: {values['config1']}")
                print(f"     配置2: {values['config2']}")
        
        # 保存比较结果
        save_result = input("\n是否保存比较结果? (y/n): ").strip().lower()
        if save_result == 'y':
            output_file = "config_comparison_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(differences, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 比较结果已保存到: {output_file}")
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_validate_config():
    """
    验证配置文件
    """
    print("\n✅ 验证配置文件")
    print("-" * 30)
    
    config_file = get_config_file()
    if not config_file:
        return
    
    try:
        config_manager = ConfigManager(config_file)
        
        print(f"\n🔍 验证配置文件: {config_file}")
        is_valid = config_manager.validate_config()
        
        if is_valid:
            print("\n✅ 配置验证通过！")
            print("配置文件可以正常使用")
        else:
            print("\n❌ 配置验证失败！")
            print("请检查上述错误信息并修正配置")
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")

def action_run_examples():
    """
    运行使用示例
    """
    print("\n📖 运行使用示例")
    print("-" * 30)
    
    try:
        from config_usage_examples import main as run_examples
        
        print("\n🚀 开始运行配置管理示例...")
        run_examples()
        
    except ImportError:
        print("❌ 找不到 config_usage_examples.py 文件")
        print("请确保该文件在当前目录中")
    except Exception as e:
        print(f"❌ 运行示例失败: {e}")

def action_show_help():
    """
    显示帮助信息
    """
    print("\n❓ 帮助信息")
    print("-" * 30)
    
    help_text = """
🚀 快速配置工具使用指南

📋 主要功能:
1. 📊 查看配置摘要 - 快速了解配置文件的主要参数
2. 🔧 快速修改配置 - 修改常用参数如批次大小、学习率等
3. 🎯 使用分辨率预设 - 应用预定义的分辨率配置
4. ⚡ 硬件优化配置 - 根据硬件类型生成优化配置
5. 🧪 生成实验配置 - 批量生成参数网格搜索配置
6. 🔍 比较配置文件 - 分析两个配置文件的差异
7. ✅ 验证配置文件 - 检查配置文件的合理性
8. 📖 运行使用示例 - 运行完整的使用示例

🎯 使用建议:
- 首先使用"查看配置摘要"了解当前配置
- 使用"硬件优化配置"根据你的硬件生成合适的配置
- 使用"快速修改配置"进行小幅调整
- 使用"生成实验配置"进行批量实验
- 使用"验证配置文件"确保配置正确

📁 输出文件:
- config_*.yaml - 生成的配置文件
- experiment_*/ - 实验配置目录
- config_comparison_result.json - 配置比较结果

💡 提示:
- 所有生成的配置都会自动验证
- 支持YAML锚点和引用的参数复用
- 可以处理嵌套参数路径 (如 data.batch_size)
- 支持多种数据类型的自动转换
"""
    
    print(help_text)

def main():
    """
    主函数
    """
    print_banner()
    
    while True:
        try:
            show_menu()
            choice = input("请选择操作 (0-9): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用快速配置工具！")
                break
            
            elif choice == '1':
                action_view_summary()
            
            elif choice == '2':
                action_quick_modify()
            
            elif choice == '3':
                action_resolution_presets()
            
            elif choice == '4':
                action_hardware_optimization()
            
            elif choice == '5':
                action_generate_experiments()
            
            elif choice == '6':
                action_compare_configs()
            
            elif choice == '7':
                action_validate_config()
            
            elif choice == '8':
                action_run_examples()
            
            elif choice == '9':
                action_show_help()
            
            else:
                print("❌ 无效选择，请输入 0-9 之间的数字")
            
            # 等待用户按键继续
            if choice != '0':
                input("\n按回车键继续...")
        
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 程序错误: {e}")
            input("\n按回车键继续...")

if __name__ == '__main__':
    main()