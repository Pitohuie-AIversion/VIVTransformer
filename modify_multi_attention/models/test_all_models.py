"""综合测试脚本，验证所有增强模型的功能"""

import torch
import torch.nn as nn
import time
import traceback
from typing import Dict, List, Tuple, Any
import numpy as np

# 导入所有模型
from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d, create_enhanced_unet3d

class ModelTester:
    """模型测试器"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 测试配置
        self.test_configs = {
            '1d': {
                'input_shape': (4, 32, 1),
                'expected_output_shape': (4, 128, 1),
                'input_resolution': 32,
                'output_resolution': 128
            },
            '2d': {
                'input_shape': (4, 32, 32, 1),
                'expected_output_shape': (4, 128, 128, 1),
                'input_resolution': (32, 32),
                'output_resolution': (128, 128)
            },
            '3d': {
                'input_shape': (4, 16, 32, 32, 1),
                'expected_output_shape': (4, 32, 128, 128, 1),
                'input_resolution': (16, 32, 32),
                'output_resolution': (32, 128, 128)
            }
        }
        
        self.results = []
    
    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             num_runs: int = 10) -> float:
        """测量推理时间"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        # 同步GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测量时间
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        return (end_time - start_time) / num_runs
    
    def get_memory_usage(self) -> float:
        """获取GPU内存使用量(MB)"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def test_model(self, model_name: str, model_creator, config_key: str, 
                   **model_kwargs) -> Dict[str, Any]:
        """测试单个模型"""
        print(f"\n{'='*60}")
        print(f"测试模型: {model_name}")
        print(f"{'='*60}")
        
        config = self.test_configs[config_key]
        result = {
            'model_name': model_name,
            'config_key': config_key,
            'success': False,
            'error': None,
            'parameters': 0,
            'inference_time': 0.0,
            'memory_usage': 0.0,
            'output_shape': None,
            'expected_shape': config['expected_output_shape']
        }
        
        try:
            # 创建模型
            model = model_creator(
                input_resolution=config['input_resolution'],
                output_resolution=config['output_resolution'],
                **model_kwargs
            ).to(self.device)
            
            # 计算参数数量
            result['parameters'] = self.count_parameters(model)
            print(f"参数数量: {result['parameters']:,}")
            
            # 创建测试输入
            input_tensor = torch.randn(*config['input_shape']).to(self.device)
            print(f"输入形状: {input_tensor.shape}")
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
            
            result['output_shape'] = tuple(output.shape)
            print(f"输出形状: {output.shape}")
            print(f"期望形状: {config['expected_output_shape']}")
            
            # 检查输出形状
            if output.shape == torch.Size(config['expected_output_shape']):
                print("✅ 输出形状正确")
            else:
                print("⚠️ 输出形状不匹配")
            
            # 测量推理时间
            result['inference_time'] = self.measure_inference_time(model, input_tensor)
            print(f"平均推理时间: {result['inference_time']:.4f}s")
            
            # 测量内存使用
            result['memory_usage'] = self.get_memory_usage()
            print(f"内存使用: {result['memory_usage']:.2f}MB")
            
            # 检查输出值的合理性
            if torch.isnan(output).any():
                print("❌ 输出包含NaN值")
                result['error'] = "Output contains NaN"
            elif torch.isinf(output).any():
                print("❌ 输出包含无穷值")
                result['error'] = "Output contains Inf"
            else:
                print("✅ 输出值正常")
                result['success'] = True
            
            # 清理GPU缓存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ 测试失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
        
        return result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """运行所有模型测试"""
        print("开始综合模型测试...")
        print(f"设备: {self.device}")
        
        # 定义所有测试用例
        test_cases = [
            # FNO模型
            ('FNO1d', create_enhanced_fno1d, '1d', {'num_channels': 1, 'modes': 16, 'width': 64}),
            ('FNO2d', create_enhanced_fno2d, '2d', {'num_channels': 1, 'modes1': 12, 'modes2': 12, 'width': 20}),
            
            # MLP模型
            ('MLP1d', create_enhanced_mlp1d, '1d', {'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6}),
            ('MLP2d', create_enhanced_mlp2d, '2d', {'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6}),
            
            # PINN模型
            ('PINN1d', create_enhanced_pinn1d, '1d', {'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6}),
            ('PINN2d', create_enhanced_pinn2d, '2d', {'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6}),
            
            # UNet模型
            ('UNet1d', create_enhanced_unet1d, '1d', {'in_channels': 1, 'out_channels': 1, 'init_features': 32}),
            ('UNet2d', create_enhanced_unet2d, '2d', {'in_channels': 1, 'out_channels': 1, 'init_features': 32}),
            ('UNet3d', create_enhanced_unet3d, '3d', {'in_channels': 1, 'out_channels': 1, 'init_features': 16}),
        ]
        
        # 运行测试
        for model_name, model_creator, config_key, model_kwargs in test_cases:
            result = self.test_model(model_name, model_creator, config_key, **model_kwargs)
            self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("\n" + "="*80)
        report.append("模型测试综合报告")
        report.append("="*80)
        
        # 统计信息
        total_models = len(self.results)
        successful_models = sum(1 for r in self.results if r['success'])
        failed_models = total_models - successful_models
        
        report.append(f"\n总测试模型数: {total_models}")
        report.append(f"成功模型数: {successful_models}")
        report.append(f"失败模型数: {failed_models}")
        report.append(f"成功率: {successful_models/total_models*100:.1f}%")
        
        # 成功模型详情
        if successful_models > 0:
            report.append("\n" + "-"*60)
            report.append("成功模型详情:")
            report.append("-"*60)
            
            successful_results = [r for r in self.results if r['success']]
            
            # 按参数数量排序
            successful_results.sort(key=lambda x: x['parameters'])
            
            report.append(f"{'模型名称':<12} {'参数数量':<12} {'推理时间(s)':<12} {'内存(MB)':<10} {'输出形状':<20}")
            report.append("-"*80)
            
            for result in successful_results:
                report.append(
                    f"{result['model_name']:<12} "
                    f"{result['parameters']:<12,} "
                    f"{result['inference_time']:<12.4f} "
                    f"{result['memory_usage']:<10.1f} "
                    f"{str(result['output_shape']):<20}"
                )
        
        # 失败模型详情
        if failed_models > 0:
            report.append("\n" + "-"*60)
            report.append("失败模型详情:")
            report.append("-"*60)
            
            failed_results = [r for r in self.results if not r['success']]
            
            for result in failed_results:
                report.append(f"模型: {result['model_name']}")
                report.append(f"错误: {result['error']}")
                report.append("")
        
        # 性能分析
        if successful_models > 0:
            report.append("\n" + "-"*60)
            report.append("性能分析:")
            report.append("-"*60)
            
            # 最快模型
            fastest = min(successful_results, key=lambda x: x['inference_time'])
            report.append(f"最快模型: {fastest['model_name']} ({fastest['inference_time']:.4f}s)")
            
            # 最小参数模型
            smallest = min(successful_results, key=lambda x: x['parameters'])
            report.append(f"最小参数模型: {smallest['model_name']} ({smallest['parameters']:,} 参数)")
            
            # 最省内存模型
            most_efficient = min(successful_results, key=lambda x: x['memory_usage'])
            report.append(f"最省内存模型: {most_efficient['model_name']} ({most_efficient['memory_usage']:.1f}MB)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "model_test_results.txt"):
        """保存测试结果"""
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n测试结果已保存到: {filename}")

def main():
    """主函数"""
    # 创建测试器
    tester = ModelTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 生成并显示报告
    report = tester.generate_report()
    print(report)
    
    # 保存结果
    tester.save_results("modify_multi_attention/models/model_test_results.txt")
    
    return results

if __name__ == "__main__":
    main()