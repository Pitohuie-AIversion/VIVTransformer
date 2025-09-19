"""真实数据测试脚本，验证模型在实际数据上的性能"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path

# 导入所有模型
from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d, create_enhanced_unet3d

class RealDataTester:
    """真实数据测试器"""
    
    def __init__(self, device='auto', data_dir='../../../generate_data/preprocessed_data'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.data_dir = Path(data_dir)
        print(f"使用设备: {self.device}")
        print(f"数据目录: {self.data_dir}")
        
        # 测试结果存储
        self.test_results = {}
        
        # 模型配置
        self.model_configs = {
            'FNO1d': {
                'creator': create_enhanced_fno1d,
                'params': {'num_channels': 1, 'modes': 16, 'width': 64}
            },
            'FNO2d': {
                'creator': create_enhanced_fno2d,
                'params': {'num_channels': 1, 'modes1': 12, 'modes2': 12, 'width': 20}
            },
            'MLP1d': {
                'creator': create_enhanced_mlp1d,
                'params': {'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'MLP2d': {
                'creator': create_enhanced_mlp2d,
                'params': {'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'PINN1d': {
                'creator': create_enhanced_pinn1d,
                'params': {'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'PINN2d': {
                'creator': create_enhanced_pinn2d,
                'params': {'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'UNet1d': {
                'creator': create_enhanced_unet1d,
                'params': {'in_channels': 1, 'out_channels': 1, 'init_features': 32}
            },
            'UNet2d': {
                'creator': create_enhanced_unet2d,
                'params': {'in_channels': 1, 'out_channels': 1, 'init_features': 32}
            }
        }
    
    def find_data_files(self) -> Dict[str, List[str]]:
        """查找可用的数据文件"""
        data_files = {
            'h5': [],
            'pt': [],
            'npy': []
        }
        
        if not self.data_dir.exists():
            print(f"数据目录不存在: {self.data_dir}")
            return data_files
        
        # 递归查找数据文件
        for ext in ['h5', 'hdf5', 'pt', 'pth', 'npy']:
            files = list(self.data_dir.rglob(f'*.{ext}'))
            if ext in ['h5', 'hdf5']:
                data_files['h5'].extend([str(f) for f in files])
            elif ext in ['pt', 'pth']:
                data_files['pt'].extend([str(f) for f in files])
            elif ext == 'npy':
                data_files['npy'].extend([str(f) for f in files])
        
        return data_files
    
    def load_sample_data(self, file_path: str, max_samples: int = 10) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """加载样本数据"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix in ['.h5', '.hdf5']:
                return self._load_h5_data(file_path, max_samples)
            elif file_path.suffix in ['.pt', '.pth']:
                return self._load_pt_data(file_path, max_samples)
            elif file_path.suffix == '.npy':
                return self._load_npy_data(file_path, max_samples)
            else:
                print(f"不支持的文件格式: {file_path.suffix}")
                return None
                
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return None
    
    def _load_h5_data(self, file_path: Path, max_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """加载HDF5数据"""
        with h5py.File(file_path, 'r') as f:
            # 查找可能的数据键
            keys = list(f.keys())
            print(f"HDF5文件键: {keys}")
            
            # 尝试常见的键名
            data_keys = ['data', 'input', 'x', 'features']
            target_keys = ['target', 'output', 'y', 'labels']
            
            input_data = None
            target_data = None
            
            # 查找输入数据
            for key in data_keys:
                if key in f:
                    input_data = torch.tensor(f[key][:max_samples], dtype=torch.float32)
                    break
            
            # 查找目标数据
            for key in target_keys:
                if key in f:
                    target_data = torch.tensor(f[key][:max_samples], dtype=torch.float32)
                    break
            
            # 如果没找到标准键名，使用前两个数组
            if input_data is None and len(keys) >= 1:
                input_data = torch.tensor(f[keys[0]][:max_samples], dtype=torch.float32)
            
            if target_data is None and len(keys) >= 2:
                target_data = torch.tensor(f[keys[1]][:max_samples], dtype=torch.float32)
            
            # 如果只有一个数据集，创建简单的目标
            if input_data is not None and target_data is None:
                # 创建一个简单的回归目标（例如，数据的均值）
                if len(input_data.shape) > 2:
                    target_data = input_data.mean(dim=tuple(range(2, len(input_data.shape))), keepdim=True)
                else:
                    target_data = input_data.clone()
            
            return input_data, target_data
    
    def _load_pt_data(self, file_path: Path, max_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """加载PyTorch数据"""
        data = torch.load(file_path, map_location='cpu')
        
        if isinstance(data, dict):
            # 尝试常见的键名
            input_keys = ['input', 'x', 'data', 'features']
            target_keys = ['target', 'y', 'output', 'labels']
            
            input_data = None
            target_data = None
            
            for key in input_keys:
                if key in data:
                    input_data = data[key][:max_samples]
                    break
            
            for key in target_keys:
                if key in data:
                    target_data = data[key][:max_samples]
                    break
            
            # 如果没找到，使用字典中的前两个张量
            if input_data is None or target_data is None:
                tensor_items = [(k, v) for k, v in data.items() if isinstance(v, torch.Tensor)]
                if len(tensor_items) >= 1:
                    input_data = tensor_items[0][1][:max_samples]
                if len(tensor_items) >= 2:
                    target_data = tensor_items[1][1][:max_samples]
                elif len(tensor_items) == 1:
                    # 只有一个张量，创建简单目标
                    target_data = input_data.clone()
            
            return input_data, target_data
        
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            return data[0][:max_samples], data[1][:max_samples]
        
        elif isinstance(data, torch.Tensor):
            # 单个张量，创建简单的回归任务
            input_data = data[:max_samples]
            target_data = input_data.clone()
            return input_data, target_data
        
        return None
    
    def _load_npy_data(self, file_path: Path, max_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """加载NumPy数据"""
        data = np.load(file_path)
        
        if data.ndim >= 2:
            input_data = torch.tensor(data[:max_samples], dtype=torch.float32)
            # 创建简单的回归目标
            if data.ndim > 2:
                target_data = input_data.mean(dim=tuple(range(2, len(input_data.shape))), keepdim=True)
            else:
                target_data = input_data.clone()
            
            return input_data, target_data
        
        return None
    
    def prepare_data_for_model(self, input_data: torch.Tensor, target_data: torch.Tensor, 
                              model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """为特定模型准备数据"""
        # 确保数据有正确的维度
        if input_data.dim() < 3:
            # 添加通道维度
            input_data = input_data.unsqueeze(-1)
        
        if target_data.dim() < 3:
            target_data = target_data.unsqueeze(-1)
        
        # 根据模型类型调整数据形状
        if '1d' in model_name.lower():
            # 1D模型：(batch, length, channels)
            if input_data.dim() > 3:
                # 如果是多维数据，取第一个维度
                input_data = input_data.view(input_data.size(0), -1, input_data.size(-1))
                target_data = target_data.view(target_data.size(0), -1, target_data.size(-1))
        
        elif '2d' in model_name.lower():
            # 2D模型：(batch, height, width, channels)
            if input_data.dim() == 3:
                # 如果是1D数据，重塑为2D
                size = int(np.sqrt(input_data.size(1)))
                if size * size == input_data.size(1):
                    input_data = input_data.view(input_data.size(0), size, size, input_data.size(-1))
                    target_data = target_data.view(target_data.size(0), size, size, target_data.size(-1))
                else:
                    # 如果不能完美重塑，填充到最近的平方数
                    target_size = int(np.ceil(np.sqrt(input_data.size(1))))
                    pad_size = target_size * target_size - input_data.size(1)
                    input_data = torch.cat([input_data, torch.zeros(input_data.size(0), pad_size, input_data.size(-1))], dim=1)
                    target_data = torch.cat([target_data, torch.zeros(target_data.size(0), pad_size, target_data.size(-1))], dim=1)
                    input_data = input_data.view(input_data.size(0), target_size, target_size, input_data.size(-1))
                    target_data = target_data.view(target_data.size(0), target_size, target_size, target_data.size(-1))
            elif input_data.dim() > 4:
                # 如果是3D或更高维数据，取前两个空间维度
                input_data = input_data.view(input_data.size(0), input_data.size(1), input_data.size(2), -1)
                target_data = target_data.view(target_data.size(0), target_data.size(1), target_data.size(2), -1)
        
        return input_data, target_data
    
    def test_model_on_data(self, model_name: str, input_data: torch.Tensor, 
                          target_data: torch.Tensor) -> Dict[str, Any]:
        """在真实数据上测试模型"""
        print(f"\n测试模型 {model_name} 在真实数据上...")
        
        config = self.model_configs[model_name]
        
        # 准备数据
        input_data, target_data = self.prepare_data_for_model(input_data, target_data, model_name)
        
        print(f"输入数据形状: {input_data.shape}")
        print(f"目标数据形状: {target_data.shape}")
        
        # 确定输入和输出分辨率
        if '1d' in model_name.lower():
            input_resolution = input_data.size(1)
            output_resolution = target_data.size(1)
        elif '2d' in model_name.lower():
            input_resolution = (input_data.size(1), input_data.size(2))
            output_resolution = (target_data.size(1), target_data.size(2))
        else:
            input_resolution = input_data.shape[1:-1]
            output_resolution = target_data.shape[1:-1]
        
        result = {
            'model_name': model_name,
            'input_shape': tuple(input_data.shape),
            'target_shape': tuple(target_data.shape),
            'input_resolution': input_resolution,
            'output_resolution': output_resolution,
            'success': False,
            'error': None
        }
        
        try:
            # 创建模型
            model_params = config['params'].copy()
            model_params['input_resolution'] = input_resolution
            model_params['output_resolution'] = output_resolution
            
            model = config['creator'](**model_params).to(self.device)
            
            # 移动数据到设备
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output = model(input_data)
                inference_time = time.time() - start_time
            
            print(f"输出形状: {output.shape}")
            print(f"推理时间: {inference_time:.4f}s")
            
            # 计算基本指标
            if output.shape == target_data.shape:
                mse = torch.mean((output - target_data) ** 2).item()
                mae = torch.mean(torch.abs(output - target_data)).item()
                
                # 计算相对误差
                target_norm = torch.norm(target_data).item()
                if target_norm > 0:
                    relative_error = torch.norm(output - target_data).item() / target_norm
                else:
                    relative_error = float('inf')
                
                result.update({
                    'mse': mse,
                    'mae': mae,
                    'relative_error': relative_error,
                    'inference_time': inference_time,
                    'output_shape': tuple(output.shape),
                    'success': True
                })
                
                print(f"MSE: {mse:.6f}")
                print(f"MAE: {mae:.6f}")
                print(f"相对误差: {relative_error:.6f}")
                
            else:
                result['error'] = f"输出形状不匹配: 期望 {target_data.shape}, 得到 {output.shape}"
                print(f"❌ {result['error']}")
            
            # 检查输出的数值稳定性
            if torch.isnan(output).any():
                result['error'] = "输出包含NaN值"
                print("❌ 输出包含NaN值")
            elif torch.isinf(output).any():
                result['error'] = "输出包含无穷值"
                print("❌ 输出包含无穷值")
            elif result['success']:
                print("✅ 测试成功")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ 测试失败: {e}")
        
        return result
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合真实数据测试"""
        print("开始真实数据综合测试...")
        
        # 查找数据文件
        data_files = self.find_data_files()
        
        print(f"\n找到的数据文件:")
        for file_type, files in data_files.items():
            print(f"{file_type.upper()}: {len(files)} 个文件")
            for file in files[:3]:  # 只显示前3个
                print(f"  - {file}")
            if len(files) > 3:
                print(f"  ... 还有 {len(files) - 3} 个文件")
        
        # 如果没有找到数据文件，创建合成数据进行测试
        if not any(data_files.values()):
            print("\n没有找到真实数据文件，使用合成数据进行测试...")
            return self._test_with_synthetic_data()
        
        # 测试每种类型的数据文件
        all_results = {}
        
        for file_type, files in data_files.items():
            if not files:
                continue
            
            print(f"\n{'='*60}")
            print(f"测试 {file_type.upper()} 数据文件")
            print(f"{'='*60}")
            
            # 选择第一个文件进行测试
            test_file = files[0]
            print(f"使用文件: {test_file}")
            
            # 加载数据
            data = self.load_sample_data(test_file, max_samples=5)
            if data is None:
                print(f"跳过文件 {test_file}：无法加载数据")
                continue
            
            input_data, target_data = data
            print(f"加载的数据形状: 输入 {input_data.shape}, 目标 {target_data.shape}")
            
            # 测试所有模型
            file_results = {}
            for model_name in self.model_configs.keys():
                try:
                    result = self.test_model_on_data(model_name, input_data.clone(), target_data.clone())
                    file_results[model_name] = result
                except Exception as e:
                    print(f"模型 {model_name} 测试失败: {e}")
                    file_results[model_name] = {'error': str(e), 'success': False}
            
            all_results[f"{file_type}_{Path(test_file).stem}"] = {
                'file_path': test_file,
                'file_type': file_type,
                'data_shape': {
                    'input': tuple(input_data.shape),
                    'target': tuple(target_data.shape)
                },
                'model_results': file_results
            }
        
        self.test_results = all_results
        return all_results
    
    def _test_with_synthetic_data(self) -> Dict[str, Any]:
        """使用合成数据进行测试"""
        print("生成合成测试数据...")
        
        # 生成不同类型的合成数据
        synthetic_datasets = {
            'sine_wave_1d': {
                'input': torch.randn(5, 64, 1),
                'target': torch.sin(torch.linspace(0, 4*np.pi, 128)).unsqueeze(0).unsqueeze(-1).repeat(5, 1, 1)
            },
            'gaussian_2d': {
                'input': torch.randn(5, 32, 32, 1),
                'target': torch.randn(5, 64, 64, 1)
            },
            'random_sequence': {
                'input': torch.randn(5, 50, 1),
                'target': torch.randn(5, 100, 1)
            }
        }
        
        all_results = {}
        
        for dataset_name, data in synthetic_datasets.items():
            print(f"\n{'='*60}")
            print(f"测试合成数据集: {dataset_name}")
            print(f"{'='*60}")
            
            input_data = data['input']
            target_data = data['target']
            
            print(f"数据形状: 输入 {input_data.shape}, 目标 {target_data.shape}")
            
            # 测试所有模型
            dataset_results = {}
            for model_name in self.model_configs.keys():
                try:
                    result = self.test_model_on_data(model_name, input_data.clone(), target_data.clone())
                    dataset_results[model_name] = result
                except Exception as e:
                    print(f"模型 {model_name} 测试失败: {e}")
                    dataset_results[model_name] = {'error': str(e), 'success': False}
            
            all_results[dataset_name] = {
                'data_type': 'synthetic',
                'data_shape': {
                    'input': tuple(input_data.shape),
                    'target': tuple(target_data.shape)
                },
                'model_results': dataset_results
            }
        
        self.test_results = all_results
        return all_results
    
    def generate_test_report(self) -> str:
        """生成测试报告"""
        if not self.test_results:
            return "没有测试结果可用"
        
        report = []
        report.append("="*80)
        report.append("真实数据测试报告")
        report.append("="*80)
        
        # 统计信息
        total_tests = 0
        successful_tests = 0
        
        for dataset_name, dataset_result in self.test_results.items():
            model_results = dataset_result['model_results']
            total_tests += len(model_results)
            successful_tests += sum(1 for r in model_results.values() if r.get('success', False))
        
        report.append(f"\n总测试数: {total_tests}")
        report.append(f"成功测试数: {successful_tests}")
        report.append(f"成功率: {successful_tests/total_tests*100:.1f}%")
        
        # 每个数据集的详细结果
        for dataset_name, dataset_result in self.test_results.items():
            report.append(f"\n{'='*60}")
            report.append(f"数据集: {dataset_name}")
            report.append(f"{'='*60}")
            
            if 'file_path' in dataset_result:
                report.append(f"文件路径: {dataset_result['file_path']}")
            
            report.append(f"数据形状: {dataset_result['data_shape']}")
            
            # 模型结果表格
            model_results = dataset_result['model_results']
            successful_models = {k: v for k, v in model_results.items() if v.get('success', False)}
            
            if successful_models:
                report.append("\n成功的模型:")
                report.append("-"*60)
                report.append(f"{'模型':<12} {'MSE':<12} {'MAE':<12} {'相对误差':<12} {'推理时间(s)':<12}")
                report.append("-"*60)
                
                for model_name, result in successful_models.items():
                    report.append(
                        f"{model_name:<12} "
                        f"{result.get('mse', 0):<12.6f} "
                        f"{result.get('mae', 0):<12.6f} "
                        f"{result.get('relative_error', 0):<12.6f} "
                        f"{result.get('inference_time', 0):<12.4f}"
                    )
            
            # 失败的模型
            failed_models = {k: v for k, v in model_results.items() if not v.get('success', False)}
            if failed_models:
                report.append("\n失败的模型:")
                report.append("-"*40)
                for model_name, result in failed_models.items():
                    report.append(f"{model_name}: {result.get('error', '未知错误')}")
        
        # 跨数据集性能比较
        if len(self.test_results) > 1:
            report.append(f"\n{'='*60}")
            report.append("跨数据集性能比较")
            report.append(f"{'='*60}")
            
            # 收集所有成功的结果
            all_successful = {}
            for dataset_name, dataset_result in self.test_results.items():
                for model_name, result in dataset_result['model_results'].items():
                    if result.get('success', False):
                        if model_name not in all_successful:
                            all_successful[model_name] = []
                        all_successful[model_name].append(result)
            
            # 计算平均性能
            if all_successful:
                report.append("\n平均性能 (跨所有数据集):")
                report.append("-"*60)
                report.append(f"{'模型':<12} {'平均MSE':<12} {'平均MAE':<12} {'平均时间(s)':<12} {'成功次数':<10}")
                report.append("-"*60)
                
                for model_name, results in all_successful.items():
                    avg_mse = np.mean([r.get('mse', 0) for r in results])
                    avg_mae = np.mean([r.get('mae', 0) for r in results])
                    avg_time = np.mean([r.get('inference_time', 0) for r in results])
                    success_count = len(results)
                    
                    report.append(
                        f"{model_name:<12} "
                        f"{avg_mse:<12.6f} "
                        f"{avg_mae:<12.6f} "
                        f"{avg_time:<12.4f} "
                        f"{success_count:<10}"
                    )
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "real_data_test_results.txt"):
        """保存测试结果"""
        report = self.generate_test_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n测试结果已保存到: {filename}")

def main():
    """主函数"""
    # 创建测试器
    tester = RealDataTester()
    
    # 运行综合测试
    results = tester.run_comprehensive_test()
    
    # 生成并显示报告
    report = tester.generate_test_report()
    print(report)
    
    # 保存结果
    tester.save_results("modify_multi_attention/models/real_data_test_results.txt")
    
    return results

if __name__ == "__main__":
    main()