#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件与测试脚本模型兼容性验证

功能:
1. 验证unified_training_config.yaml中定义的所有模型
2. 检查这些模型在run_crop_model_test.py中的可用性
3. 测试模型创建和前向传播
4. 生成兼容性报告

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import traceback

# 设置路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'models'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigModelCompatibilityTester:
    """配置文件与测试脚本模型兼容性测试器"""
    
    def __init__(self, config_path: str, test_script_path: str):
        self.config_path = Path(config_path)
        self.test_script_path = Path(test_script_path)
        self.config = self.load_config()
        self.test_results = {}
        
        # 导入测试脚本中的模型创建函数
        self.import_test_script_functions()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            raise
    
    def import_test_script_functions(self):
        """导入测试脚本中的模型创建函数"""
        try:
            # 动态导入run_crop_model_test.py中的函数
            spec = __import__('importlib.util').util.spec_from_file_location(
                "run_crop_model_test", self.test_script_path
            )
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 获取模型创建函数
            self.create_enhanced_model = getattr(module, 'create_enhanced_model', None)
            self.create_simple_model = getattr(module, 'create_simple_model', None)
            self.create_model = getattr(module, 'create_model', None)
            
            # 获取模型类
            self.SimpleMLP = getattr(module, 'SimpleMLP', None)
            self.SimpleTransformer = getattr(module, 'SimpleTransformer', None)
            self.CustomTransformerWrapper = getattr(module, 'CustomTransformerWrapper', None)
            self.EnhancedFNOWrapper = getattr(module, 'EnhancedFNOWrapper', None)
            self.EnhancedUNetWrapper = getattr(module, 'EnhancedUNetWrapper', None)
            
            logger.info("✅ 成功导入测试脚本中的模型创建函数")
            
        except Exception as e:
            logger.error(f"❌ 导入测试脚本函数失败: {e}")
            raise
    
    def get_config_models(self) -> Dict[str, Dict[str, Any]]:
        """获取配置文件中定义的所有模型"""
        models = {}
        
        if 'models' in self.config:
            for model_name, model_config in self.config['models'].items():
                if model_name != 'active_model':  # 跳过active_model配置项
                    models[model_name] = model_config
        
        logger.info(f"📋 配置文件中定义的模型: {list(models.keys())}")
        return models
    
    def test_model_creation(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试模型创建"""
        result = {
            'model_name': model_name,
            'config_available': True,
            'enhanced_model_creation': False,
            'simple_model_creation': False,
            'wrapper_model_creation': False,
            'forward_pass': False,
            'parameter_count': 0,
            'input_shape': None,
            'output_shape': None,
            'error_messages': [],
            'success_method': None
        }
        
        # 从配置获取输入输出维度
        input_dim = self.config.get('data', {}).get('input_dim', 1024)
        output_dim = self.config.get('data', {}).get('output_dim', 16384)
        
        result['input_shape'] = (1, input_dim)  # batch_size=1
        
        try:
            # 方法1: 尝试使用增强模型创建函数
            if self.create_enhanced_model:
                try:
                    model = self.create_enhanced_model(model_config, input_dim, output_dim)
                    result['enhanced_model_creation'] = True
                    result['success_method'] = 'enhanced_model'
                    result['parameter_count'] = sum(p.numel() for p in model.parameters())
                    
                    # 测试前向传播
                    test_input = torch.randn(1, input_dim)
                    with torch.no_grad():
                        output = model(test_input)
                        result['output_shape'] = tuple(output.shape)
                        result['forward_pass'] = True
                    
                    logger.info(f"✅ {model_name}: 增强模型创建成功")
                    return result
                    
                except Exception as e:
                    result['error_messages'].append(f"增强模型创建失败: {str(e)}")
                    logger.warning(f"⚠️ {model_name}: 增强模型创建失败: {e}")
            
            # 方法2: 尝试使用简单模型创建函数
            if self.create_simple_model:
                try:
                    model = self.create_simple_model(model_config.get('model_type', model_name), 
                                                   input_dim, output_dim, model_config)
                    result['simple_model_creation'] = True
                    result['success_method'] = 'simple_model'
                    result['parameter_count'] = sum(p.numel() for p in model.parameters())
                    
                    # 测试前向传播
                    test_input = torch.randn(1, input_dim)
                    with torch.no_grad():
                        output = model(test_input)
                        result['output_shape'] = tuple(output.shape)
                        result['forward_pass'] = True
                    
                    logger.info(f"✅ {model_name}: 简单模型创建成功")
                    return result
                    
                except Exception as e:
                    result['error_messages'].append(f"简单模型创建失败: {str(e)}")
                    logger.warning(f"⚠️ {model_name}: 简单模型创建失败: {e}")
            
            # 方法3: 尝试使用通用模型创建函数
            if self.create_model:
                try:
                    model_type = model_config.get('model_type', model_name)
                    model = self.create_model(model_type, input_dim, output_dim, **model_config)
                    result['wrapper_model_creation'] = True
                    result['success_method'] = 'wrapper_model'
                    result['parameter_count'] = sum(p.numel() for p in model.parameters())
                    
                    # 测试前向传播
                    test_input = torch.randn(1, input_dim)
                    with torch.no_grad():
                        output = model(test_input)
                        result['output_shape'] = tuple(output.shape)
                        result['forward_pass'] = True
                    
                    logger.info(f"✅ {model_name}: 包装器模型创建成功")
                    return result
                    
                except Exception as e:
                    result['error_messages'].append(f"包装器模型创建失败: {str(e)}")
                    logger.warning(f"⚠️ {model_name}: 包装器模型创建失败: {e}")
            
            # 方法4: 尝试直接使用模型类
            model_class_map = {
                'mlp': self.SimpleMLP,
                'transformer': self.SimpleTransformer,
                'custom_transformer': self.CustomTransformerWrapper,
                'fno': self.EnhancedFNOWrapper,
                'unet': self.EnhancedUNetWrapper
            }
            
            model_type = model_config.get('model_type', model_name)
            if model_type in model_class_map and model_class_map[model_type]:
                try:
                    model_class = model_class_map[model_type]
                    model = model_class(input_dim, output_dim, **model_config)
                    result['wrapper_model_creation'] = True
                    result['success_method'] = 'direct_class'
                    result['parameter_count'] = sum(p.numel() for p in model.parameters())
                    
                    # 测试前向传播
                    test_input = torch.randn(1, input_dim)
                    with torch.no_grad():
                        output = model(test_input)
                        result['output_shape'] = tuple(output.shape)
                        result['forward_pass'] = True
                    
                    logger.info(f"✅ {model_name}: 直接类创建成功")
                    return result
                    
                except Exception as e:
                    result['error_messages'].append(f"直接类创建失败: {str(e)}")
                    logger.warning(f"⚠️ {model_name}: 直接类创建失败: {e}")
            
        except Exception as e:
            result['error_messages'].append(f"测试过程异常: {str(e)}")
            logger.error(f"❌ {model_name}: 测试过程异常: {e}")
        
        return result
    
    def run_compatibility_test(self) -> Dict[str, Any]:
        """运行兼容性测试"""
        logger.info("🚀 开始配置文件与测试脚本模型兼容性测试")
        
        config_models = self.get_config_models()
        
        # 测试每个模型
        for model_name, model_config in config_models.items():
            logger.info(f"🔍 测试模型: {model_name}")
            result = self.test_model_creation(model_name, model_config)
            self.test_results[model_name] = result
        
        # 生成统计信息
        total_models = len(self.test_results)
        successful_models = sum(1 for r in self.test_results.values() 
                              if r['forward_pass'])
        success_rate = (successful_models / total_models * 100) if total_models > 0 else 0
        
        summary = {
            'total_models': total_models,
            'successful_models': successful_models,
            'failed_models': total_models - successful_models,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'config_path': str(self.config_path),
            'test_script_path': str(self.test_script_path),
            'test_time': datetime.now().isoformat()
        }
        
        logger.info(f"📊 测试完成: {successful_models}/{total_models} 模型成功 ({success_rate:.1f}%)")
        
        return summary
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """生成兼容性测试报告"""
        if output_path is None:
            output_path = f"config_model_compatibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report_lines = [
            "# 配置文件与测试脚本模型兼容性报告",
            "",
            f"**生成时间**: {results['test_time']}",
            f"**配置文件**: `{results['config_path']}`",
            f"**测试脚本**: `{results['test_script_path']}`",
            "",
            "## 测试概览",
            "",
            f"- **总模型数**: {results['total_models']}",
            f"- **成功模型数**: {results['successful_models']}",
            f"- **失败模型数**: {results['failed_models']}",
            f"- **成功率**: {results['success_rate']:.1f}%",
            "",
            "## 详细测试结果",
            ""
        ]
        
        # 成功的模型
        successful_models = {name: result for name, result in results['test_results'].items() 
                           if result['forward_pass']}
        
        if successful_models:
            report_lines.extend([
                "### ✅ 成功运行的模型",
                "",
                "| 模型名称 | 创建方法 | 参数数量 | 输入形状 | 输出形状 |",
                "|---------|---------|---------|---------|---------|"
            ])
            
            for name, result in successful_models.items():
                report_lines.append(
                    f"| {name} | {result['success_method']} | {result['parameter_count']:,} | "
                    f"{result['input_shape']} | {result['output_shape']} |"
                )
            
            report_lines.append("")
        
        # 失败的模型
        failed_models = {name: result for name, result in results['test_results'].items() 
                        if not result['forward_pass']}
        
        if failed_models:
            report_lines.extend([
                "### ❌ 运行失败的模型",
                "",
                "| 模型名称 | 错误信息 |",
                "|---------|---------|"
            ])
            
            for name, result in failed_models.items():
                error_msg = "; ".join(result['error_messages']) if result['error_messages'] else "未知错误"
                report_lines.append(f"| {name} | {error_msg} |")
            
            report_lines.append("")
        
        # 兼容性分析
        report_lines.extend([
            "## 兼容性分析",
            "",
            "### 模型创建方法统计",
            ""
        ])
        
        method_stats = {}
        for result in results['test_results'].values():
            if result['forward_pass']:
                method = result['success_method']
                method_stats[method] = method_stats.get(method, 0) + 1
        
        for method, count in method_stats.items():
            report_lines.append(f"- **{method}**: {count} 个模型")
        
        report_lines.extend([
            "",
            "### 建议",
            ""
        ])
        
        if results['success_rate'] >= 80:
            report_lines.append("✅ **兼容性良好**: 大部分模型都可以正常运行。")
        elif results['success_rate'] >= 60:
            report_lines.append("⚠️ **兼容性一般**: 部分模型存在问题，建议检查失败的模型配置。")
        else:
            report_lines.append("❌ **兼容性较差**: 多数模型无法运行，需要修复模型实现或配置。")
        
        if failed_models:
            report_lines.extend([
                "",
                "**针对失败模型的建议**:",
                ""
            ])
            
            for name, result in failed_models.items():
                report_lines.append(f"- **{name}**: 检查模型参数配置和依赖项")
        
        # 写入报告文件
        report_content = "\n".join(report_lines)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"📄 兼容性报告已保存到: {output_path}")
        except Exception as e:
            logger.error(f"❌ 保存报告失败: {e}")
        
        return output_path

def main():
    """主函数"""
    # 配置文件和测试脚本路径
    config_path = "configs/unified_training_config.yaml"
    test_script_path = "run_crop_model_test.py"
    
    try:
        # 创建测试器
        tester = ConfigModelCompatibilityTester(config_path, test_script_path)
        
        # 运行兼容性测试
        results = tester.run_compatibility_test()
        
        # 生成报告
        report_path = tester.generate_report(results)
        
        print(f"\n🎉 兼容性测试完成!")
        print(f"📊 成功率: {results['success_rate']:.1f}% ({results['successful_models']}/{results['total_models']})")
        print(f"📄 详细报告: {report_path}")
        
        # 显示简要结果
        print(f"\n📋 测试结果概览:")
        for name, result in results['test_results'].items():
            status = "✅" if result['forward_pass'] else "❌"
            method = result.get('success_method', 'failed')
            print(f"  {status} {name}: {method}")
        
    except Exception as e:
        logger.error(f"❌ 测试过程发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()