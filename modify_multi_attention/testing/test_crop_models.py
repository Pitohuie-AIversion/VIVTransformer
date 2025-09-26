#!/usr/bin/env python3
"""
测试run_crop_model_test.py中所有模型的运行状态
检查增强模型、简单模型和包装器模型的可用性
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import traceback
from datetime import datetime

# 设置路径
current_dir = Path(__file__).parent
models_dir = current_dir / 'models'
sys.path.insert(0, str(models_dir))
sys.path.insert(0, str(current_dir))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CropModelTester:
    """测试run_crop_model_test.py中所有模型的运行状态"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        self.enhanced_models_available = False
        self.simple_models_available = False
        
        # 测试输入维度
        self.input_dim = 400  # 20x20
        self.output_dim = 40000  # 200x200
        self.batch_size = 2
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"测试输入维度: {self.input_dim}")
        logger.info(f"测试输出维度: {self.output_dim}")
    
    def test_enhanced_models_import(self):
        """测试增强模型导入"""
        logger.info("🔍 测试增强模型导入...")
        
        try:
            # 尝试导入增强模型
            try:
                from enhanced_transformer import EnhancedTransformer1d, EnhancedTransformer2d
                from enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
                from enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
                from enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
            except ImportError:
                from models.enhanced_transformer import EnhancedTransformer1d, EnhancedTransformer2d
                from models.enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
                from models.enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
                from models.enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
            
            self.enhanced_models_available = True
            self.enhanced_classes = {
                'EnhancedTransformer1d': EnhancedTransformer1d,
                'EnhancedTransformer2d': EnhancedTransformer2d,
                'EnhancedFNO1d': EnhancedFNO1d,
                'EnhancedFNO2d': EnhancedFNO2d,
                'EnhancedMLP1d': EnhancedMLP1d,
                'EnhancedMLP2d': EnhancedMLP2d,
                'EnhancedUNet1d': EnhancedUNet1d,
                'EnhancedUNet2d': EnhancedUNet2d
            }
            
            logger.info("✅ 增强模型导入成功")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ 增强模型导入失败: {e}")
            self.enhanced_models_available = False
            return False
    
    def test_simple_models_import(self):
        """测试简单模型导入"""
        logger.info("🔍 测试简单模型导入...")
        
        try:
            # 导入run_crop_model_test.py中的简单模型
            from run_crop_model_test import (
                SimpleMLP, SimpleTransformer, CustomTransformerWrapper,
                EnhancedFNOWrapper, EnhancedUNetWrapper
            )
            
            self.simple_models_available = True
            self.simple_classes = {
                'SimpleMLP': SimpleMLP,
                'SimpleTransformer': SimpleTransformer,
                'CustomTransformerWrapper': CustomTransformerWrapper,
                'EnhancedFNOWrapper': EnhancedFNOWrapper,
                'EnhancedUNetWrapper': EnhancedUNetWrapper
            }
            
            logger.info("✅ 简单模型导入成功")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ 简单模型导入失败: {e}")
            self.simple_models_available = False
            return False
    
    def test_model_creation_and_forward(self, model_class, model_name: str, **kwargs):
        """测试模型创建和前向传播"""
        try:
            # 创建模型
            if 'Enhanced' in model_name and '1d' in model_name:
                # 增强1D模型
                model = model_class(
                    input_channels=1,
                    output_channels=1,
                    input_resolution=int(np.sqrt(self.input_dim)),
                    output_resolution=int(np.sqrt(self.output_dim)),
                    **kwargs
                )
            elif 'Enhanced' in model_name and '2d' in model_name:
                # 增强2D模型
                model = model_class(
                    in_channels=1,
                    out_channels=1,
                    input_resolution=(int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim))),
                    output_resolution=(int(np.sqrt(self.output_dim)), int(np.sqrt(self.output_dim))),
                    **kwargs
                )
            else:
                # 简单模型和包装器
                model = model_class(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    **kwargs
                )
            
            model = model.to(self.device)
            model.eval()
            
            # 创建测试输入
            if 'Enhanced' in model_name and ('1d' in model_name or '2d' in model_name):
                # 增强模型需要特定形状的输入
                if '1d' in model_name:
                    test_input = torch.randn(self.batch_size, 1, int(np.sqrt(self.input_dim))).to(self.device)
                else:  # 2d
                    test_input = torch.randn(self.batch_size, 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim))).to(self.device)
            else:
                # 简单模型使用扁平输入
                test_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                output = model(test_input)
            
            # 检查输出形状
            expected_shape = (self.batch_size, self.output_dim)
            if output.shape != expected_shape:
                # 对于增强模型，输出可能需要reshape
                if len(output.shape) > 2:
                    output = output.view(self.batch_size, -1)
            
            # 计算参数数量
            param_count = sum(p.numel() for p in model.parameters())
            
            return {
                'status': 'success',
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'param_count': param_count,
                'device': str(next(model.parameters()).device)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_enhanced_models(self):
        """测试所有增强模型"""
        if not self.enhanced_models_available:
            logger.warning("⚠️ 增强模型不可用，跳过测试")
            return
        
        logger.info("🧪 测试增强模型...")
        
        # 测试增强模型的默认配置
        enhanced_configs = {
            'EnhancedTransformer1d': {
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'attention_type': 'simplified_self_attention',
                'pe_type': 'learnable_1d',
                'dropout': 0.1
            },
            'EnhancedMLP1d': {
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout': 0.1
            },
            'EnhancedUNet1d': {
                'init_features': 32
            },
            'EnhancedFNO1d': {
                'modes': 12,
                'width': 64
            }
        }
        
        for model_name, model_class in self.enhanced_classes.items():
            if '1d' in model_name:  # 只测试1D版本
                logger.info(f"  测试 {model_name}...")
                config = enhanced_configs.get(model_name, {})
                result = self.test_model_creation_and_forward(model_class, model_name, **config)
                self.test_results[model_name] = result
                
                if result['status'] == 'success':
                    logger.info(f"    ✅ {model_name} 测试成功")
                else:
                    logger.error(f"    ❌ {model_name} 测试失败: {result['error']}")
    
    def test_simple_models(self):
        """测试所有简单模型"""
        if not self.simple_models_available:
            logger.warning("⚠️ 简单模型不可用，跳过测试")
            return
        
        logger.info("🧪 测试简单模型...")
        
        # 测试简单模型的默认配置
        simple_configs = {
            'SimpleMLP': {
                'hidden_dims': [256, 128],
                'dropout': 0.1
            },
            'SimpleTransformer': {
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.1
            },
            'CustomTransformerWrapper': {
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.1,
                'attention_type': "relative"
            },
            'EnhancedFNOWrapper': {},
            'EnhancedUNetWrapper': {}
        }
        
        for model_name, model_class in self.simple_classes.items():
            logger.info(f"  测试 {model_name}...")
            config = simple_configs.get(model_name, {})
            result = self.test_model_creation_and_forward(model_class, model_name, **config)
            self.test_results[model_name] = result
            
            if result['status'] == 'success':
                logger.info(f"    ✅ {model_name} 测试成功")
            else:
                logger.error(f"    ❌ {model_name} 测试失败: {result['error']}")
    
    def test_model_factory_functions(self):
        """测试模型工厂函数"""
        logger.info("🧪 测试模型工厂函数...")
        
        try:
            from run_crop_model_test import create_enhanced_model, create_simple_model, create_model
            
            # 测试create_enhanced_model
            model_configs = [
                {'model_type': 'transformer', 'd_model': 128, 'num_heads': 4},
                {'model_type': 'mlp', 'hidden_dims': [256]},
                {'model_type': 'unet', 'base_ch': 32},
                {'model_type': 'fno', 'modes': 12, 'width': 64}
            ]
            
            for config in model_configs:
                model_type = config['model_type']
                logger.info(f"  测试 create_enhanced_model({model_type})...")
                
                try:
                    model = create_enhanced_model(config, self.input_dim, self.output_dim)
                    model = model.to(self.device)
                    
                    # 测试前向传播
                    test_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
                    with torch.no_grad():
                        output = model(test_input)
                    
                    param_count = sum(p.numel() for p in model.parameters())
                    
                    self.test_results[f'create_enhanced_model_{model_type}'] = {
                        'status': 'success',
                        'input_shape': list(test_input.shape),
                        'output_shape': list(output.shape),
                        'param_count': param_count
                    }
                    logger.info(f"    ✅ create_enhanced_model({model_type}) 测试成功")
                    
                except Exception as e:
                    self.test_results[f'create_enhanced_model_{model_type}'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    logger.error(f"    ❌ create_enhanced_model({model_type}) 测试失败: {e}")
            
            # 测试create_model
            model_types = ['mlp', 'transformer', 'custom_transformer', 'fno', 'unet']
            
            for model_type in model_types:
                logger.info(f"  测试 create_model({model_type})...")
                
                try:
                    model = create_model(model_type, self.input_dim, self.output_dim)
                    model = model.to(self.device)
                    
                    # 测试前向传播
                    test_input = torch.randn(self.batch_size, self.input_dim).to(self.device)
                    with torch.no_grad():
                        output = model(test_input)
                    
                    param_count = sum(p.numel() for p in model.parameters())
                    
                    self.test_results[f'create_model_{model_type}'] = {
                        'status': 'success',
                        'input_shape': list(test_input.shape),
                        'output_shape': list(output.shape),
                        'param_count': param_count
                    }
                    logger.info(f"    ✅ create_model({model_type}) 测试成功")
                    
                except Exception as e:
                    self.test_results[f'create_model_{model_type}'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    logger.error(f"    ❌ create_model({model_type}) 测试失败: {e}")
                    
        except ImportError as e:
            logger.error(f"❌ 无法导入模型工厂函数: {e}")
    
    def generate_report(self):
        """生成测试报告"""
        logger.info("📊 生成测试报告...")
        
        # 统计结果
        total_models = len(self.test_results)
        successful_models = sum(1 for result in self.test_results.values() if result['status'] == 'success')
        failed_models = total_models - successful_models
        
        # 生成报告
        report_lines = [
            "# run_crop_model_test.py 模型运行状态报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试概览",
            f"- 总模型数: {total_models}",
            f"- 成功运行: {successful_models}",
            f"- 运行失败: {failed_models}",
            f"- 成功率: {successful_models/total_models*100:.1f}%" if total_models > 0 else "- 成功率: 0%",
            "",
            "## 详细结果",
            ""
        ]
        
        # 按类别分组
        enhanced_models = {k: v for k, v in self.test_results.items() if 'Enhanced' in k and 'create_' not in k}
        simple_models = {k: v for k, v in self.test_results.items() if 'Enhanced' not in k and 'create_' not in k}
        factory_models = {k: v for k, v in self.test_results.items() if 'create_' in k}
        
        # 增强模型结果
        if enhanced_models:
            report_lines.extend([
                "### 增强模型 (Enhanced Models)",
                ""
            ])
            
            for model_name, result in enhanced_models.items():
                status_icon = "✅" if result['status'] == 'success' else "❌"
                report_lines.append(f"**{model_name}** {status_icon}")
                
                if result['status'] == 'success':
                    report_lines.extend([
                        f"- 参数数量: {result['param_count']:,}",
                        f"- 输入形状: {result['input_shape']}",
                        f"- 输出形状: {result['output_shape']}",
                        f"- 设备: {result['device']}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"- 错误: {result['error']}",
                        ""
                    ])
        
        # 简单模型结果
        if simple_models:
            report_lines.extend([
                "### 简单模型 (Simple Models)",
                ""
            ])
            
            for model_name, result in simple_models.items():
                status_icon = "✅" if result['status'] == 'success' else "❌"
                report_lines.append(f"**{model_name}** {status_icon}")
                
                if result['status'] == 'success':
                    report_lines.extend([
                        f"- 参数数量: {result['param_count']:,}",
                        f"- 输入形状: {result['input_shape']}",
                        f"- 输出形状: {result['output_shape']}",
                        f"- 设备: {result['device']}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"- 错误: {result['error']}",
                        ""
                    ])
        
        # 工厂函数结果
        if factory_models:
            report_lines.extend([
                "### 模型工厂函数 (Model Factory Functions)",
                ""
            ])
            
            for model_name, result in factory_models.items():
                status_icon = "✅" if result['status'] == 'success' else "❌"
                report_lines.append(f"**{model_name}** {status_icon}")
                
                if result['status'] == 'success':
                    report_lines.extend([
                        f"- 参数数量: {result['param_count']:,}",
                        f"- 输入形状: {result['input_shape']}",
                        f"- 输出形状: {result['output_shape']}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"- 错误: {result['error']}",
                        ""
                    ])
        
        # 保存报告
        report_path = current_dir / "crop_models_test_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"📄 测试报告已保存到: {report_path}")
        
        # 打印总结
        logger.info("=" * 60)
        logger.info("🎯 测试总结")
        logger.info(f"总模型数: {total_models}")
        logger.info(f"成功运行: {successful_models}")
        logger.info(f"运行失败: {failed_models}")
        logger.info(f"成功率: {successful_models/total_models*100:.1f}%" if total_models > 0 else "成功率: 0%")
        logger.info("=" * 60)
        
        return report_path
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始测试run_crop_model_test.py中的所有模型...")
        
        # 测试模型导入
        self.test_enhanced_models_import()
        self.test_simple_models_import()
        
        # 测试模型创建和运行
        self.test_enhanced_models()
        self.test_simple_models()
        self.test_model_factory_functions()
        
        # 生成报告
        report_path = self.generate_report()
        
        return self.test_results, report_path

def main():
    """主函数"""
    tester = CropModelTester()
    results, report_path = tester.run_all_tests()
    
    print(f"\n📄 详细报告已保存到: {report_path}")
    print("🎉 测试完成！")

if __name__ == "__main__":
    main()