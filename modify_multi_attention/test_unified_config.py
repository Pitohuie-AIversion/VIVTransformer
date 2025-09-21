#!/usr/bin/env python3
"""
测试unified_training_config.yaml配置文件
验证多模型训练配置是否正确
"""

import yaml
import sys
from pathlib import Path

def test_unified_config():
    """测试统一配置文件"""
    config_path = Path("configs/unified_training_config.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查关键配置项
        checks = []
        
        # 检查实验模式
        if config.get('experiment', {}).get('mode') == 'multi_model_comparison':
            checks.append("✅ 实验模式: multi_model_comparison")
        else:
            checks.append("⚠️  实验模式未设置为multi_model_comparison")
        
        # 检查模型配置
        models = config.get('models', {})
        required_models = ['transformer', 'mlp', 'unet', 'fno']
        
        for model in required_models:
            if model in models:
                model_config = models[model]
                if 'model_type' in model_config and 'input_dim' in model_config:
                    checks.append(f"✅ {model}模型配置完整")
                else:
                    checks.append(f"❌ {model}模型配置不完整")
            else:
                checks.append(f"❌ 缺少{model}模型配置")
        
        # 检查多模型对比配置
        comparison = config.get('comparison', {})
        multi_model = comparison.get('multi_model', {})
        
        if multi_model.get('enabled'):
            checks.append("✅ 多模型对比已启用")
            
            config_models = multi_model.get('models', [])
            if all(model in config_models for model in required_models):
                checks.append("✅ 多模型对比包含所有必需模型")
            else:
                checks.append("⚠️  多模型对比模型列表不完整")
        else:
            checks.append("❌ 多模型对比未启用")
        
        # 检查数据配置
        data_config = config.get('data', {})
        if 'input_dim' in data_config and 'output_dim' in data_config:
            checks.append("✅ 数据维度配置正确")
        else:
            checks.append("❌ 数据维度配置缺失")
        
        # 检查训练配置
        training_config = config.get('training', {})
        if 'epochs' in training_config and 'learning_rate' in training_config:
            checks.append("✅ 训练参数配置正确")
        else:
            checks.append("❌ 训练参数配置缺失")
        
        # 输出检查结果
        print("\n📋 配置检查结果:")
        for check in checks:
            print(f"  {check}")
        
        # 统计结果
        success_count = sum(1 for check in checks if check.startswith("✅"))
        total_count = len(checks)
        
        print(f"\n📊 检查通过率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count == total_count:
            print("\n🎉 配置文件完全正确，可以用于多模型训练！")
            return True
        elif success_count >= total_count * 0.8:
            print("\n⚠️  配置文件基本正确，但有一些小问题需要注意")
            return True
        else:
            print("\n❌ 配置文件存在重要问题，需要修复")
            return False
            
    except yaml.YAMLError as e:
        print(f"❌ YAML格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

def show_usage_example():
    """显示使用示例"""
    print("\n📖 使用示例:")
    print("1. 快速测试 (2个epoch):")
    print("   python run_crop_model_test.py --config configs/unified_training_config.yaml --models \"MLP,FNO\" --epochs 2")
    
    print("\n2. 完整多模型对比:")
    print("   python run_crop_model_test.py --config configs/unified_training_config.yaml --models \"MLP,UNet,Transformer,FNO\"")
    
    print("\n3. 单模型训练:")
    print("   python run_crop_model_test.py --config configs/unified_training_config.yaml --models \"Transformer\"")

if __name__ == "__main__":
    print("🔍 测试unified_training_config.yaml配置文件...")
    
    success = test_unified_config()
    
    if success:
        show_usage_example()
        sys.exit(0)
    else:
        print("\n❌ 配置测试失败，请检查配置文件")
        sys.exit(1)