#!/usr/bin/env python3
"""轻量化Transformer模型测试脚本"""

import torch
import torch.nn as nn
import yaml
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'mymodels'))

try:
    from enhanced_transformer import create_enhanced_transformer1d, create_enhanced_transformer2d
    from mymodels.transformer import TransformerFlowReconstructionModel
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightweightTransformerTester:
    """轻量化Transformer测试器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('device', {}).get('use_cuda', True) else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def create_lightweight_transformer(self, model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
        """创建轻量化Transformer模型"""
        try:
            # 使用底层TransformerFlowReconstructionModel
            model = TransformerFlowReconstructionModel(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=model_config['d_model'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                attention_type=model_config.get('attention_type', 'simplified_self'),
                pe_type=model_config.get('pe_type', 'learnable_1d'),
                time_encoding=model_config.get('time_encoding', 'embedding'),
                max_time_steps=model_config.get('max_time_steps', 100),
                seq_len=input_dim,  # 简化：假设seq_len = input_dim
                input_hw=None,
                output_head_type='global',
                use_memory_film=False,  # 禁用额外功能减少参数
                use_memory_concat=False
            )
            
            # 手动设置dim_feedforward（如果支持）
            if hasattr(model, 'encoder'):
                for layer in model.encoder.layers:
                    if hasattr(layer, 'linear1'):
                        # 重新创建前馈网络以匹配配置
                        d_model = model_config['d_model']
                        dim_feedforward = model_config.get('dim_feedforward', d_model * 2)
                        layer.linear1 = nn.Linear(d_model, dim_feedforward)
                        layer.linear2 = nn.Linear(dim_feedforward, d_model)
            
            if hasattr(model, 'decoder'):
                for layer in model.decoder.layers:
                    if hasattr(layer, 'linear1'):
                        d_model = model_config['d_model']
                        dim_feedforward = model_config.get('dim_feedforward', d_model * 2)
                        layer.linear1 = nn.Linear(d_model, dim_feedforward)
                        layer.linear2 = nn.Linear(dim_feedforward, d_model)
            
            return model
            
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            raise
    
    def test_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个模型"""
        logger.info(f"\n测试模型: {model_name}")
        logger.info(f"配置: {model_config}")
        
        # 获取数据配置
        data_config = self.config['data']
        input_resolution = data_config['input_resolution']
        output_resolution = data_config['output_resolution']
        input_channels = data_config['input_channels']
        output_channels = data_config['output_channels']
        batch_size = data_config['batch_size']
        
        # 计算维度
        input_dim = input_resolution * input_channels
        output_dim = output_resolution * output_channels
        
        try:
            # 创建模型
            start_time = time.time()
            model = self.create_lightweight_transformer(model_config, input_dim, output_dim)
            model = model.to(self.device)
            creation_time = time.time() - start_time
            
            # 计算参数量
            param_count = self.count_parameters(model)
            target_params = model_config.get('target_params', 1000000)
            
            logger.info(f"模型参数量: {param_count:,}")
            logger.info(f"目标参数量: {target_params:,}")
            logger.info(f"参数量比例: {param_count/target_params:.2f}")
            
            # 创建测试数据
            test_input = torch.randn(batch_size, input_dim).to(self.device)
            time_steps = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output = model(test_input, time_steps)
                inference_time = time.time() - start_time
            
            logger.info(f"输入形状: {test_input.shape}")
            logger.info(f"输出形状: {output.shape}")
            logger.info(f"推理时间: {inference_time:.4f}s")
            
            # 检查输出维度
            expected_output_shape = (batch_size, output_dim)
            actual_output_shape = output.shape
            shape_match = actual_output_shape == expected_output_shape
            
            if not shape_match:
                logger.warning(f"输出形状不匹配! 期望: {expected_output_shape}, 实际: {actual_output_shape}")
            
            # 内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_used = 0
            
            return {
                'model_name': model_name,
                'param_count': param_count,
                'target_params': target_params,
                'param_ratio': param_count / target_params,
                'creation_time': creation_time,
                'inference_time': inference_time,
                'memory_used_mb': memory_used,
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'expected_output_shape': list(expected_output_shape),
                'shape_match': shape_match,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"模型 {model_name} 测试失败: {e}")
            return {
                'model_name': model_name,
                'param_count': 0,
                'target_params': model_config.get('target_params', 0),
                'param_ratio': 0,
                'creation_time': 0,
                'inference_time': 0,
                'memory_used_mb': 0,
                'input_shape': [],
                'output_shape': [],
                'expected_output_shape': [],
                'shape_match': False,
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有模型测试"""
        logger.info("开始轻量化Transformer模型测试")
        
        models_config = self.config['models']
        results = {}
        
        for model_name, model_config in models_config.items():
            if model_config.get('model_type') == 'transformer':
                results[model_name] = self.test_model(model_name, model_config)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = ["\n" + "="*80]
        report.append("轻量化Transformer模型测试报告")
        report.append("="*80)
        
        successful_models = [r for r in results.values() if r['success']]
        failed_models = [r for r in results.values() if not r['success']]
        
        report.append(f"\n总测试模型数: {len(results)}")
        report.append(f"成功模型数: {len(successful_models)}")
        report.append(f"失败模型数: {len(failed_models)}")
        
        if successful_models:
            report.append("\n成功模型详情:")
            report.append("-" * 60)
            
            for result in successful_models:
                report.append(f"\n模型: {result['model_name']}")
                report.append(f"  参数量: {result['param_count']:,}")
                report.append(f"  目标参数量: {result['target_params']:,}")
                report.append(f"  参数量比例: {result['param_ratio']:.3f}")
                report.append(f"  创建时间: {result['creation_time']:.4f}s")
                report.append(f"  推理时间: {result['inference_time']:.4f}s")
                report.append(f"  内存使用: {result['memory_used_mb']:.2f}MB")
                report.append(f"  输入形状: {result['input_shape']}")
                report.append(f"  输出形状: {result['output_shape']}")
                report.append(f"  形状匹配: {'✓' if result['shape_match'] else '✗'}")
                
                # 参数量评估
                if result['param_ratio'] <= 1.0:
                    report.append(f"  参数量评估: ✓ 达到目标")
                elif result['param_ratio'] <= 1.2:
                    report.append(f"  参数量评估: ⚠ 略超目标")
                else:
                    report.append(f"  参数量评估: ✗ 严重超标")
        
        if failed_models:
            report.append("\n失败模型详情:")
            report.append("-" * 60)
            
            for result in failed_models:
                report.append(f"\n模型: {result['model_name']}")
                report.append(f"  错误: {result['error']}")
        
        # 参数量排名
        if successful_models:
            report.append("\n参数量排名 (从小到大):")
            report.append("-" * 40)
            sorted_models = sorted(successful_models, key=lambda x: x['param_count'])
            for i, result in enumerate(sorted_models, 1):
                report.append(f"{i}. {result['model_name']}: {result['param_count']:,} 参数")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], report: str):
        """保存测试结果"""
        # 保存详细结果
        results_file = Path(__file__).parent / "lightweight_transformer_test_results.yaml"
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        # 保存报告
        report_file = Path(__file__).parent / "lightweight_transformer_test_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"结果已保存到: {results_file}")
        logger.info(f"报告已保存到: {report_file}")

def main():
    """主函数"""
    config_path = Path(__file__).parent / "lightweight_transformer_config.yaml"
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    tester = LightweightTransformerTester(str(config_path))
    
    try:
        # 运行测试
        results = tester.run_all_tests()
        
        # 生成报告
        report = tester.generate_report(results)
        print(report)
        
        # 保存结果
        tester.save_results(results, report)
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()