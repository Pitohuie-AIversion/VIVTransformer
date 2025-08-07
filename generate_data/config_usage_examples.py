#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理工具使用示例

展示如何使用 config_manager.py 进行配置参数复用和管理

作者: AI Assistant
日期: 2025
"""

import os
import sys
import json
from pathlib import Path
from config_manager import ConfigManager

def example_1_basic_usage():
    """
    示例1: 基础使用 - 加载和验证配置
    """
    print("\n" + "="*60)
    print("📖 示例1: 基础使用")
    print("="*60)
    
    # 加载配置文件
    config_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 创建配置管理器
    config_manager = ConfigManager(config_path)
    
    # 验证配置
    print("\n🔍 验证配置...")
    is_valid = config_manager.validate_config()
    
    # 打印配置摘要
    print("\n📊 配置摘要:")
    config_manager.print_config_summary()
    
    return config_manager

def example_2_parameter_override():
    """
    示例2: 参数覆盖 - 快速修改配置
    """
    print("\n" + "="*60)
    print("🔧 示例2: 参数覆盖")
    print("="*60)
    
    config_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    config_manager = ConfigManager(config_path)
    
    # 定义要覆盖的参数
    overrides = {
        'data.batch_size': 32,  # 修改批次大小
        'data.num_samples': 5000,  # 修改样本数量
        'training.learning_rate': 0.0005,  # 修改学习率
        'training.epochs': 50,  # 修改训练轮数
        'model.num_layers': 4,  # 修改模型层数
        'model.d_model': 256,  # 修改模型维度
    }
    
    print("\n🔧 应用参数覆盖:")
    for key, value in overrides.items():
        print(f"  {key}: {value}")
    
    # 应用覆盖
    new_config = config_manager.override_params(overrides)
    
    # 验证新配置
    print("\n🔍 验证覆盖后的配置...")
    is_valid = config_manager.validate_config(new_config)
    
    # 保存新配置
    output_path = "config_override_example.yaml"
    config_manager.save_config(output_path, new_config)
    print(f"\n💾 覆盖后的配置已保存到: {output_path}")
    
    return new_config

def example_3_resolution_presets():
    """
    示例3: 分辨率预设 - 使用预定义的分辨率配置
    """
    print("\n" + "="*60)
    print("🎯 示例3: 分辨率预设")
    print("="*60)
    
    config_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    config_manager = ConfigManager(config_path)
    
    # 显示可用的分辨率预设
    print("\n📋 可用的分辨率预设:")
    for preset_name, preset_config in config_manager.presets.items():
        input_res = preset_config.get('input_resolution', 'N/A')
        output_res = preset_config.get('output_resolution', 'N/A')
        print(f"  {preset_name}: {input_res} -> {output_res}")
    
    # 使用不同的分辨率预设创建配置
    preset_configs = {}
    
    for preset_name in ['low', 'medium', 'high']:
        if preset_name in config_manager.presets:
            preset = config_manager.presets[preset_name]
            
            # 创建覆盖参数
            overrides = {
                'data.input_resolution': preset['input_resolution'],
                'data.output_resolution': preset['output_resolution'],
                'data.batch_size': preset.get('recommended_batch_size', 16)
            }
            
            # 应用覆盖
            new_config = config_manager.override_params(overrides)
            preset_configs[preset_name] = new_config
            
            # 保存配置
            output_path = f"config_{preset_name}_resolution.yaml"
            config_manager.save_config(output_path, new_config)
            print(f"\n💾 {preset_name.upper()} 分辨率配置已保存到: {output_path}")
    
    return preset_configs

def example_4_batch_experiment_generation():
    """
    示例4: 批量实验配置生成 - 参数网格搜索
    """
    print("\n" + "="*60)
    print("🧪 示例4: 批量实验配置生成")
    print("="*60)
    
    config_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    config_manager = ConfigManager(config_path)
    
    # 定义参数网格
    param_grid = {
        'data.batch_size': [16, 32, 64],
        'training.learning_rate': [0.0001, 0.0005, 0.001],
        'model.num_layers': [4, 6, 8],
        'model.d_model': [256, 512]
    }
    
    print("\n🔬 参数网格:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"\n📊 总共将生成 {total_combinations} 个实验配置")
    
    # 生成实验配置
    output_dir = "experiment_configs"
    config_paths = config_manager.generate_experiment_configs(
        config_manager.config, param_grid, output_dir
    )
    
    print(f"\n✅ 成功生成 {len(config_paths)} 个实验配置到目录: {output_dir}")
    
    # 显示前几个配置文件
    print("\n📄 生成的配置文件（前5个）:")
    for i, path in enumerate(config_paths[:5]):
        print(f"  {i+1}. {Path(path).name}")
    
    if len(config_paths) > 5:
        print(f"  ... 还有 {len(config_paths) - 5} 个配置文件")
    
    return config_paths

def example_5_config_comparison():
    """
    示例5: 配置比较 - 分析配置差异
    """
    print("\n" + "="*60)
    print("🔍 示例5: 配置比较")
    print("="*60)
    
    # 比较原始配置和优化配置
    config1_path = "dynamic_config_server_downsampling.yaml"
    config2_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config1_path).exists() or not Path(config2_path).exists():
        print(f"❌ 配置文件不存在")
        return
    
    config_manager1 = ConfigManager(config1_path)
    config_manager2 = ConfigManager(config2_path)
    
    # 比较配置
    differences = config_manager1.compare_configs(
        config_manager1.config, config_manager2.config
    )
    
    print("\n📊 配置比较结果:")
    print(f"  仅在原始配置中: {len(differences['only_in_config1'])} 项")
    print(f"  仅在优化配置中: {len(differences['only_in_config2'])} 项")
    print(f"  值不同: {len(differences['different_values'])} 项")
    print(f"  值相同: {len(differences['same_values'])} 项")
    
    # 显示主要差异
    if differences['different_values']:
        print("\n🔍 主要差异:")
        for path, values in list(differences['different_values'].items())[:10]:
            print(f"  {path}:")
            print(f"    原始: {values['config1']}")
            print(f"    优化: {values['config2']}")
    
    # 保存比较结果
    comparison_file = "config_comparison_result.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(differences, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 比较结果已保存到: {comparison_file}")
    
    return differences

def example_6_custom_templates():
    """
    示例6: 自定义模板 - 创建和使用配置模板
    """
    print("\n" + "="*60)
    print("📋 示例6: 自定义模板")
    print("="*60)
    
    config_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    config_manager = ConfigManager(config_path)
    
    # 显示可用模板
    print("\n📋 可用的配置模板:")
    for template_name in config_manager.templates.keys():
        print(f"  - {template_name}")
    
    # 使用模板创建配置
    template_configs = {}
    
    for template_name in config_manager.templates.keys():
        try:
            # 从模板创建配置
            new_config = config_manager.create_from_template(template_name)
            template_configs[template_name] = new_config
            
            # 保存配置
            output_path = f"config_from_{template_name}_template.yaml"
            config_manager.save_config(output_path, new_config)
            print(f"\n💾 从模板 '{template_name}' 创建的配置已保存到: {output_path}")
            
        except Exception as e:
            print(f"\n❌ 使用模板 '{template_name}' 失败: {e}")
    
    return template_configs

def example_7_performance_optimization():
    """
    示例7: 性能优化配置 - 针对不同硬件环境优化
    """
    print("\n" + "="*60)
    print("⚡ 示例7: 性能优化配置")
    print("="*60)
    
    config_path = "dynamic_config_server_downsampling_optimized.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    config_manager = ConfigManager(config_path)
    
    # 定义不同硬件环境的优化配置
    hardware_configs = {
        'laptop': {
            'data.batch_size': 8,
            'dataloader.num_workers': 2,
            'dataloader.prefetch_factor': 2,
            'model.num_layers': 4,
            'model.d_model': 256,
            'training.epochs': 50
        },
        'workstation': {
            'data.batch_size': 32,
            'dataloader.num_workers': 8,
            'dataloader.prefetch_factor': 4,
            'model.num_layers': 6,
            'model.d_model': 512,
            'training.epochs': 100
        },
        'server': {
            'data.batch_size': 64,
            'dataloader.num_workers': 16,
            'dataloader.prefetch_factor': 8,
            'model.num_layers': 8,
            'model.d_model': 512,
            'training.epochs': 200
        }
    }
    
    # 为每种硬件环境生成优化配置
    optimized_configs = {}
    
    for hardware_type, overrides in hardware_configs.items():
        print(f"\n🖥️  生成 {hardware_type.upper()} 优化配置:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
        
        # 应用优化
        new_config = config_manager.override_params(overrides)
        optimized_configs[hardware_type] = new_config
        
        # 验证配置
        is_valid = config_manager.validate_config(new_config)
        
        # 保存配置
        output_path = f"config_optimized_for_{hardware_type}.yaml"
        config_manager.save_config(output_path, new_config)
        print(f"  💾 已保存到: {output_path}")
    
    return optimized_configs

def main():
    """
    运行所有示例
    """
    print("🚀 配置管理工具使用示例")
    print("="*80)
    
    try:
        # 示例1: 基础使用
        config_manager = example_1_basic_usage()
        
        # 示例2: 参数覆盖
        override_config = example_2_parameter_override()
        
        # 示例3: 分辨率预设
        preset_configs = example_3_resolution_presets()
        
        # 示例4: 批量实验配置生成
        experiment_configs = example_4_batch_experiment_generation()
        
        # 示例5: 配置比较
        comparison_result = example_5_config_comparison()
        
        # 示例6: 自定义模板
        template_configs = example_6_custom_templates()
        
        # 示例7: 性能优化配置
        optimized_configs = example_7_performance_optimization()
        
        print("\n" + "="*80)
        print("✅ 所有示例运行完成！")
        print("="*80)
        
        print("\n📁 生成的文件:")
        generated_files = [
            "config_override_example.yaml",
            "config_low_resolution.yaml",
            "config_medium_resolution.yaml",
            "config_high_resolution.yaml",
            "config_comparison_result.json",
            "config_optimized_for_laptop.yaml",
            "config_optimized_for_workstation.yaml",
            "config_optimized_for_server.yaml"
        ]
        
        for file_path in generated_files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} (未生成)")
        
        # 检查实验配置目录
        experiment_dir = Path("experiment_configs")
        if experiment_dir.exists():
            experiment_files = list(experiment_dir.glob("*.yaml"))
            print(f"  📁 experiment_configs/ ({len(experiment_files)} 个实验配置)")
        
        print("\n🎯 使用建议:")
        print("1. 使用 config_override_example.yaml 作为快速修改的起点")
        print("2. 根据硬件选择对应的优化配置文件")
        print("3. 使用 experiment_configs/ 中的配置进行批量实验")
        print("4. 查看 config_comparison_result.json 了解配置差异")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()