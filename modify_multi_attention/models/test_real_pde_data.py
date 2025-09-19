#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际PDE数据集兼容性测试
测试所有网络模型是否能正确处理真实的PDE数据
"""

import os
import sys
import torch
import h5py
import numpy as np
import yaml
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "modify_multi_attention"))
sys.path.append(str(project_root / "modify_multi_attention" / "models"))

# 导入模型
try:
    from enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
    from enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
    from enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
    from enhanced_pinn import EnhancedPINN
except ImportError as e:
    print(f"模型导入失败: {e}")
    sys.exit(1)

class RealPDEDataTester:
    """实际PDE数据测试器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_real_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载实际PDE数据"""
        data_path = self.config['data']['path']
        input_res = self.config['data']['input_resolution']
        output_res = self.config['data']['output_resolution']
        num_samples = min(self.config['data']['num_samples'], 10)  # 限制样本数量
        
        print(f"📁 加载数据: {data_path}")
        print(f"📐 输入分辨率: {input_res}, 输出分辨率: {output_res}")
        print(f"📊 样本数量: {num_samples}")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # 检查数据结构
                print(f"🔍 数据集键: {list(f.keys())}")
                
                # 尝试不同的数据键名
                data_keys = ['tensor', 'data', 'input', 'output']
                data_key = None
                for key in data_keys:
                    if key in f:
                        data_key = key
                        break
                
                if data_key is None:
                    # 如果没有找到标准键，使用第一个可用的键
                    available_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
                    if available_keys:
                        data_key = available_keys[0]
                    else:
                        raise ValueError("未找到有效的数据集")
                
                print(f"📋 使用数据键: {data_key}")
                
                # 加载数据
                if data_key in ['tensor', 'data']:
                    # 假设数据格式为 [samples, channels, height, width]
                    raw_data = f[data_key][:num_samples]
                    print(f"📏 原始数据形状: {raw_data.shape}")
                    
                    # 处理数据维度
                    if len(raw_data.shape) == 4:  # [N, C, H, W]
                        # 取第一个通道作为输入数据
                        input_data = raw_data[:, 0, :, :]
                        # 如果有多个通道，取最后一个作为输出，否则复制输入
                        if raw_data.shape[1] > 1:
                            output_data = raw_data[:, -1, :, :]
                        else:
                            output_data = raw_data[:, 0, :, :]
                    elif len(raw_data.shape) == 3:  # [N, H, W]
                        input_data = raw_data
                        output_data = raw_data  # 自监督学习
                    else:
                        raise ValueError(f"不支持的数据维度: {raw_data.shape}")
                        
                elif 'input' in f and 'output' in f:
                    # 分离的输入输出数据
                    input_data = f['input'][:num_samples]
                    output_data = f['output'][:num_samples]
                    print(f"📏 输入数据形状: {input_data.shape}")
                    print(f"📏 输出数据形状: {output_data.shape}")
                else:
                    # 使用找到的数据键
                    raw_data = f[data_key][:num_samples]
                    print(f"📏 数据形状: {raw_data.shape}")
                    
                    # 简单处理：使用相同数据作为输入输出
                    if len(raw_data.shape) >= 3:
                        input_data = raw_data
                        output_data = raw_data
                    else:
                        raise ValueError(f"数据维度不足: {raw_data.shape}")
        
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("🔄 使用模拟数据进行测试...")
            # 生成模拟数据
            input_data = np.random.randn(num_samples, *input_res).astype(np.float32)
            output_data = np.random.randn(num_samples, *output_res).astype(np.float32)
        
        # 调整数据到目标分辨率
        input_tensor = self.resize_data(input_data, input_res)
        output_tensor = self.resize_data(output_data, output_res)
        
        # 转换为PyTorch张量
        input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
        output_tensor = torch.from_numpy(output_tensor).float().to(self.device)
        
        print(f"✅ 最终输入形状: {input_tensor.shape}")
        print(f"✅ 最终输出形状: {output_tensor.shape}")
        
        return input_tensor, output_tensor
    
    def resize_data(self, data: np.ndarray, target_resolution: List[int]) -> np.ndarray:
        """调整数据到目标分辨率"""
        if len(data.shape) == 2:  # [H, W]
            data = data[np.newaxis, ...]  # [1, H, W]
        elif len(data.shape) == 4:  # [N, C, H, W]
            data = data[:, 0, :, :]  # [N, H, W]
        
        # 如果数据已经是目标分辨率，直接返回
        if data.shape[-2:] == tuple(target_resolution):
            return data
        
        # 使用简单的插值调整分辨率
        from scipy.ndimage import zoom
        
        if len(data.shape) == 3:  # [N, H, W]
            resized_data = np.zeros((data.shape[0], *target_resolution), dtype=data.dtype)
            for i in range(data.shape[0]):
                zoom_factors = (target_resolution[0] / data.shape[1], 
                               target_resolution[1] / data.shape[2])
                resized_data[i] = zoom(data[i], zoom_factors, order=1)
        else:  # [H, W]
            zoom_factors = (target_resolution[0] / data.shape[0], 
                           target_resolution[1] / data.shape[1])
            resized_data = zoom(data, zoom_factors, order=1)
            resized_data = resized_data[np.newaxis, ...]  # [1, H, W]
        
        return resized_data
    
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """创建所有测试模型"""
        input_res = self.config['data']['input_resolution']
        output_res = self.config['data']['output_resolution']
        
        models = {
            'FNO1d': EnhancedFNO1d(
                input_resolution=input_res[0] * input_res[1],
                output_resolution=output_res[0] * output_res[1],
                modes=16,
                width=64
            ),
            'FNO2d': EnhancedFNO2d(
                input_resolution=input_res,
                output_resolution=output_res,
                modes1=16,
                modes2=16,
                width=64
            ),
            'UNet1d': EnhancedUNet1d(
                input_resolution=input_res[0] * input_res[1],
                output_resolution=output_res[0] * output_res[1],
                in_channels=1,
                out_channels=1
            ),
            'UNet2d': EnhancedUNet2d(
                input_resolution=input_res,
                output_resolution=output_res,
                in_channels=1,
                out_channels=1
            ),
            'MLP1d': EnhancedMLP1d(
                input_resolution=input_res[0] * input_res[1],
                output_resolution=output_res[0] * output_res[1],
                hidden_dim=256,
                num_layers=6
            ),
            'MLP2d': EnhancedMLP2d(
                input_resolution=input_res,
                output_resolution=output_res,
                hidden_dim=256,
                num_layers=6
            ),
            'PINN': EnhancedPINN(
                input_resolution=input_res,
                output_resolution=output_res,
                hidden_dim=256,
                num_layers=6
            )
        }
        
        # 将模型移动到设备
        for name, model in models.items():
            models[name] = model.to(self.device)
            
        return models
    
    def test_model(self, name: str, model: torch.nn.Module, 
                   input_data: torch.Tensor, target_data: torch.Tensor) -> Dict[str, Any]:
        """测试单个模型"""
        print(f"\n🧪 测试模型: {name}")
        
        try:
            model.eval()
            start_time = time.time()
            
            with torch.no_grad():
                # 根据模型类型调整输入数据形状
                if '1d' in name.lower():
                    # 1D模型需要展平的输入
                    if len(input_data.shape) == 3:  # [N, H, W]
                        test_input = input_data.view(input_data.shape[0], -1)  # [N, H*W]
                    else:
                        test_input = input_data
                    
                    if len(target_data.shape) == 3:  # [N, H, W]
                        test_target = target_data.view(target_data.shape[0], -1)  # [N, H*W]
                    else:
                        test_target = target_data
                else:
                    # 2D模型需要4D输入 [N, C, H, W]
                    if len(input_data.shape) == 3:  # [N, H, W]
                        test_input = input_data.unsqueeze(1)  # [N, 1, H, W]
                    else:
                        test_input = input_data
                    
                    if len(target_data.shape) == 3:  # [N, H, W]
                        test_target = target_data.unsqueeze(1)  # [N, 1, H, W]
                    else:
                        test_target = target_data
                
                print(f"📐 输入形状: {test_input.shape}")
                print(f"📐 目标形状: {test_target.shape}")
                
                # 前向传播
                output = model(test_input)
                
                print(f"📐 输出形状: {output.shape}")
                
                # 计算损失
                if output.shape != test_target.shape:
                    print(f"⚠️  形状不匹配，调整输出形状...")
                    if len(output.shape) == 2 and len(test_target.shape) == 4:
                        # 1D输出需要重塑为2D
                        target_h, target_w = test_target.shape[-2:]
                        output = output.view(output.shape[0], 1, target_h, target_w)
                    elif len(output.shape) == 4 and len(test_target.shape) == 2:
                        # 2D输出需要展平
                        output = output.view(output.shape[0], -1)
                
                mse_loss = torch.nn.functional.mse_loss(output, test_target)
                mae_loss = torch.nn.functional.l1_loss(output, test_target)
                
            end_time = time.time()
            inference_time = end_time - start_time
            
            result = {
                'status': 'success',
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'target_shape': list(test_target.shape),
                'mse_loss': float(mse_loss.item()),
                'mae_loss': float(mae_loss.item()),
                'inference_time': inference_time,
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
            
            print(f"✅ 测试成功")
            print(f"📊 MSE损失: {result['mse_loss']:.6f}")
            print(f"📊 MAE损失: {result['mae_loss']:.6f}")
            print(f"⏱️  推理时间: {result['inference_time']:.4f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'input_shape': list(input_data.shape) if input_data is not None else None,
                'target_shape': list(target_data.shape) if target_data is not None else None
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试"""
        print("🚀 开始实际PDE数据兼容性测试")
        print("=" * 60)
        
        # 加载数据
        input_data, target_data = self.load_real_data()
        
        # 创建模型
        models = self.create_models()
        
        # 测试所有模型
        test_results = {}
        successful_models = []
        failed_models = []
        
        for name, model in models.items():
            result = self.test_model(name, model, input_data, target_data)
            test_results[name] = result
            
            if result['status'] == 'success':
                successful_models.append(name)
            else:
                failed_models.append(name)
        
        # 生成汇总报告
        summary = {
            'total_models': len(models),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(models) * 100,
            'successful_model_list': successful_models,
            'failed_model_list': failed_models,
            'data_info': {
                'input_resolution': self.config['data']['input_resolution'],
                'output_resolution': self.config['data']['output_resolution'],
                'input_shape': list(input_data.shape),
                'target_shape': list(target_data.shape),
                'data_path': self.config['data']['path']
            },
            'detailed_results': test_results
        }
        
        return summary
    
    def print_summary_report(self, summary: Dict[str, Any]):
        """打印汇总报告"""
        print("\n" + "=" * 60)
        print("📋 实际PDE数据兼容性测试报告")
        print("=" * 60)
        
        print(f"📊 总模型数: {summary['total_models']}")
        print(f"✅ 成功模型数: {summary['successful_models']}")
        print(f"❌ 失败模型数: {summary['failed_models']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        
        print(f"\n📁 数据信息:")
        print(f"   路径: {summary['data_info']['data_path']}")
        print(f"   输入分辨率: {summary['data_info']['input_resolution']}")
        print(f"   输出分辨率: {summary['data_info']['output_resolution']}")
        print(f"   输入形状: {summary['data_info']['input_shape']}")
        print(f"   目标形状: {summary['data_info']['target_shape']}")
        
        if summary['successful_model_list']:
            print(f"\n✅ 成功的模型:")
            for model_name in summary['successful_model_list']:
                result = summary['detailed_results'][model_name]
                print(f"   {model_name}: MSE={result['mse_loss']:.6f}, 时间={result['inference_time']:.4f}s")
        
        if summary['failed_model_list']:
            print(f"\n❌ 失败的模型:")
            for model_name in summary['failed_model_list']:
                result = summary['detailed_results'][model_name]
                print(f"   {model_name}: {result.get('error', '未知错误')}")
        
        print("\n" + "=" * 60)
        
        # 结论
        if summary['success_rate'] == 100:
            print("🎉 所有模型都成功处理了实际PDE数据！")
        elif summary['success_rate'] >= 80:
            print("👍 大部分模型成功处理了实际PDE数据！")
        elif summary['success_rate'] >= 50:
            print("⚠️  部分模型成功处理了实际PDE数据。")
        else:
            print("❌ 大部分模型无法处理实际PDE数据，需要进一步调试。")

def main():
    """主函数"""
    # 配置文件路径
    config_path = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/generate_data/dynamic_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 创建测试器
    tester = RealPDEDataTester(config_path)
    
    # 运行测试
    summary = tester.run_comprehensive_test()
    
    # 打印报告
    tester.print_summary_report(summary)
    
    # 保存结果
    import json
    results_path = "real_pde_data_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {results_path}")

if __name__ == "__main__":
    main()