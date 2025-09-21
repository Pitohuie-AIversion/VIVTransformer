#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面网络运行状态检查脚本
检查项目中所有模型的可用性和运行状态
"""

import torch
import torch.nn as nn
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

class NetworkStatusChecker:
    """网络运行状态检查器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.test_data_1d = torch.randn(2, 64, 1).to(self.device)  # [batch, length, channels]
        self.test_data_2d = torch.randn(2, 1, 32, 32).to(self.device)  # [batch, channels, H, W]
        
    def check_model_import(self, module_name: str, model_names: List[str]) -> Dict[str, bool]:
        """检查模型导入状态"""
        import_status = {}
        
        try:
            module = __import__(module_name)
            for model_name in model_names:
                try:
                    getattr(module, model_name)
                    import_status[model_name] = True
                    logger.info(f"✅ {model_name} 导入成功")
                except AttributeError:
                    import_status[model_name] = False
                    logger.warning(f"❌ {model_name} 导入失败 - 属性不存在")
        except ImportError as e:
            logger.error(f"❌ {module_name} 模块导入失败: {e}")
            for model_name in model_names:
                import_status[model_name] = False
                
        return import_status
    
    def test_model_creation(self, model_creator, model_name: str, **kwargs) -> Dict[str, Any]:
        """测试模型创建"""
        result = {
            'name': model_name,
            'creation_success': False,
            'forward_success': False,
            'parameter_count': 0,
            'error_message': None,
            'output_shape': None
        }
        
        try:
            # 创建模型
            model = model_creator(**kwargs)
            model = model.to(self.device)
            result['creation_success'] = True
            result['parameter_count'] = sum(p.numel() for p in model.parameters())
            
            # 测试前向传播
            model.eval()
            with torch.no_grad():
                if '1d' in model_name.lower():
                    output = model(self.test_data_1d)
                    result['output_shape'] = list(output.shape)
                elif '2d' in model_name.lower():
                    output = model(self.test_data_2d)
                    result['output_shape'] = list(output.shape)
                else:
                    # 尝试1D数据
                    try:
                        output = model(self.test_data_1d)
                        result['output_shape'] = list(output.shape)
                    except:
                        # 尝试2D数据
                        output = model(self.test_data_2d)
                        result['output_shape'] = list(output.shape)
                        
            result['forward_success'] = True
            logger.info(f"✅ {model_name} 创建和前向传播成功")
            
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"❌ {model_name} 测试失败: {e}")
            
        return result
    
    def check_enhanced_mlp(self) -> Dict[str, Any]:
        """检查增强MLP模型"""
        logger.info("\n🔍 检查增强MLP模型...")
        results = {}
        
        try:
            from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
            
            # 测试MLP1D
            results['mlp_1d'] = self.test_model_creation(
                create_enhanced_mlp1d, 'Enhanced_MLP_1D',
                input_channels=1, output_channels=1, hidden_dim=128, num_layers=4,
                input_resolution=64, output_resolution=128
            )
            
            # 测试MLP2D
            results['mlp_2d'] = self.test_model_creation(
                create_enhanced_mlp2d, 'Enhanced_MLP_2D',
                input_channels=1, output_channels=1, hidden_dim=128, num_layers=4,
                input_resolution=(32, 32), output_resolution=(64, 64)
            )
            
        except ImportError as e:
            logger.error(f"❌ Enhanced MLP 导入失败: {e}")
            results['import_error'] = str(e)
            
        return results
    
    def check_enhanced_unet(self) -> Dict[str, Any]:
        """检查增强UNet模型"""
        logger.info("\n🔍 检查增强UNet模型...")
        results = {}
        
        try:
            from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d
            
            # 测试UNet1D
            results['unet_1d'] = self.test_model_creation(
                create_enhanced_unet1d, 'Enhanced_UNet_1D',
                in_channels=1, out_channels=1, init_features=32,
                input_resolution=64, output_resolution=128
            )
            
            # 测试UNet2D
            results['unet_2d'] = self.test_model_creation(
                create_enhanced_unet2d, 'Enhanced_UNet_2D',
                in_channels=1, out_channels=1, init_features=32,
                input_resolution=(32, 32), output_resolution=(64, 64)
            )
            
        except ImportError as e:
            logger.error(f"❌ Enhanced UNet 导入失败: {e}")
            results['import_error'] = str(e)
            
        return results
    
    def check_enhanced_fno(self) -> Dict[str, Any]:
        """检查增强FNO模型"""
        logger.info("\n🔍 检查增强FNO模型...")
        results = {}
        
        try:
            from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
            
            # 测试FNO1D
            results['fno_1d'] = self.test_model_creation(
                create_enhanced_fno1d, 'Enhanced_FNO_1D',
                num_channels=1, modes=16, width=64,
                input_resolution=64, output_resolution=128
            )
            
            # 测试FNO2D
            results['fno_2d'] = self.test_model_creation(
                create_enhanced_fno2d, 'Enhanced_FNO_2D',
                num_channels=1, modes1=12, modes2=12, width=20,
                input_resolution=(32, 32), output_resolution=(64, 64)
            )
            
        except ImportError as e:
            logger.error(f"❌ Enhanced FNO 导入失败: {e}")
            results['import_error'] = str(e)
            
        return results
    
    def check_enhanced_transformer(self) -> Dict[str, Any]:
        """检查增强Transformer模型"""
        logger.info("\n🔍 检查增强Transformer模型...")
        results = {}
        
        try:
            from enhanced_transformer import create_enhanced_transformer1d, create_enhanced_transformer2d
            
            # 测试Transformer1D
            results['transformer_1d'] = self.test_model_creation(
                create_enhanced_transformer1d, 'Enhanced_Transformer_1D',
                input_channels=1, output_channels=1, d_model=128, num_heads=4, num_layers=2,
                input_resolution=64, output_resolution=128
            )
            
            # 测试Transformer2D
            results['transformer_2d'] = self.test_model_creation(
                create_enhanced_transformer2d, 'Enhanced_Transformer_2D',
                input_channels=1, output_channels=1, d_model=128, num_heads=4, num_layers=2,
                input_resolution=(32, 32), output_resolution=(64, 64)
            )
            
        except ImportError as e:
            logger.error(f"❌ Enhanced Transformer 导入失败: {e}")
            results['import_error'] = str(e)
            
        return results
    
    def check_enhanced_pinn(self) -> Dict[str, Any]:
        """检查增强PINN模型"""
        logger.info("\n🔍 检查增强PINN模型...")
        results = {}
        
        try:
            from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
            
            # 测试PINN1D
            results['pinn_1d'] = self.test_model_creation(
                create_enhanced_pinn1d, 'Enhanced_PINN_1D',
                output_dim=1, hidden_dim=128, num_layers=4,
                input_resolution=64, output_resolution=128
            )
            
            # 测试PINN2D
            results['pinn_2d'] = self.test_model_creation(
                create_enhanced_pinn2d, 'Enhanced_PINN_2D',
                output_dim=1, hidden_dim=128, num_layers=4,
                input_resolution=(32, 32), output_resolution=(64, 64)
            )
            
        except ImportError as e:
            logger.error(f"❌ Enhanced PINN 导入失败: {e}")
            results['import_error'] = str(e)
            
        return results
    
    def check_fixed_models(self) -> Dict[str, Any]:
        """检查修复版模型"""
        logger.info("\n🔍 检查修复版模型...")
        results = {}
        
        # 检查修复版MLP
        try:
            sys.path.append(str(project_root.parent / 'generate_data'))
            from fixed_enhanced_mlp import FixedEnhancedMLP
            
            results['fixed_mlp'] = self.test_model_creation(
                lambda **kwargs: FixedEnhancedMLP(**kwargs), 'Fixed_Enhanced_MLP',
                input_dim=64, output_dim=128, hidden_dim=256, num_layers=6
            )
            
        except ImportError as e:
            logger.warning(f"⚠️ Fixed MLP 导入失败: {e}")
            results['fixed_mlp_import_error'] = str(e)
        
        # 检查修复版UNet
        try:
            from fixed_enhanced_unet import FixedEnhancedUNet
            
            results['fixed_unet'] = self.test_model_creation(
                lambda **kwargs: FixedEnhancedUNet(**kwargs), 'Fixed_Enhanced_UNet',
                input_dim=64, output_dim=128, init_features=64
            )
            
        except ImportError as e:
            logger.warning(f"⚠️ Fixed UNet 导入失败: {e}")
            results['fixed_unet_import_error'] = str(e)
            
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """运行全面检查"""
        logger.info("🚀 开始全面网络运行状态检查...")
        logger.info(f"设备: {self.device}")
        
        all_results = {
            'device': str(self.device),
            'enhanced_mlp': self.check_enhanced_mlp(),
            'enhanced_unet': self.check_enhanced_unet(),
            'enhanced_fno': self.check_enhanced_fno(),
            'enhanced_transformer': self.check_enhanced_transformer(),
            'enhanced_pinn': self.check_enhanced_pinn(),
            'fixed_models': self.check_fixed_models()
        }
        
        return all_results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """生成总结报告"""
        report = []
        report.append("="*80)
        report.append("🔍 网络运行状态检查报告")
        report.append("="*80)
        report.append(f"设备: {results['device']}")
        report.append("")
        
        # 统计成功和失败的模型
        total_models = 0
        successful_models = 0
        failed_models = []
        successful_list = []
        
        for category, category_results in results.items():
            if category == 'device':
                continue
                
            report.append(f"\n📋 {category.upper().replace('_', ' ')} 模型:")
            report.append("-" * 50)
            
            if 'import_error' in category_results:
                report.append(f"❌ 导入失败: {category_results['import_error']}")
                continue
                
            for model_name, model_result in category_results.items():
                if isinstance(model_result, dict) and 'name' in model_result:
                    total_models += 1
                    
                    if model_result['creation_success'] and model_result['forward_success']:
                        successful_models += 1
                        successful_list.append(model_result['name'])
                        report.append(f"✅ {model_result['name']}")
                        report.append(f"   参数数量: {model_result['parameter_count']:,}")
                        report.append(f"   输出形状: {model_result['output_shape']}")
                    else:
                        failed_models.append(model_result['name'])
                        report.append(f"❌ {model_result['name']}")
                        if model_result['error_message']:
                            report.append(f"   错误: {model_result['error_message']}")
        
        # 总结
        report.append("\n" + "="*80)
        report.append("📊 总结")
        report.append("="*80)
        report.append(f"总模型数量: {total_models}")
        report.append(f"成功运行: {successful_models}")
        report.append(f"运行失败: {len(failed_models)}")
        report.append(f"成功率: {successful_models/total_models*100:.1f}%" if total_models > 0 else "成功率: 0%")
        
        if successful_list:
            report.append(f"\n✅ 可正常运行的模型 ({len(successful_list)}个):")
            for model in successful_list:
                report.append(f"   • {model}")
        
        if failed_models:
            report.append(f"\n❌ 运行失败的模型 ({len(failed_models)}个):")
            for model in failed_models:
                report.append(f"   • {model}")
        
        return "\n".join(report)

def main():
    """主函数"""
    checker = NetworkStatusChecker()
    
    # 运行全面检查
    results = checker.run_comprehensive_check()
    
    # 生成报告
    report = checker.generate_summary_report(results)
    
    # 输出报告
    print(report)
    
    # 保存报告到文件
    report_file = project_root / 'network_status_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n📄 详细报告已保存到: {report_file}")
    
    return results

if __name__ == "__main__":
    main()