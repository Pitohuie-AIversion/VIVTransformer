#!/usr/bin/env python3
"""
优化后的Transformer模型测试脚本
使用轻量化配置和简化损失函数验证改进效果
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

import matplotlib
matplotlib.use('Agg')

import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(Path(__file__).parent))

# 导入数据处理模块
from simple_data_loader import (
    DynamicResolutionDataset, get_dynamic_loaders, create_dynamic_config,
    log_gpu_memory, cleanup_memory
)

# 导入Transformer模型
try:
    from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
    from modify_multi_attention.training.trainer import train_model, test_model
except ImportError:
    try:
        sys.path.append(str(modify_multi_attention_path))
        from mymodels.transformer import TransformerFlowReconstructionModel
        from training.trainer import train_model, test_model
    except ImportError as e:
        logger.error(f"无法导入Transformer模型: {e}")
        TransformerFlowReconstructionModel = None

# 导入简化损失函数
try:
    from simplified_loss import create_simplified_loss
except ImportError:
    logger.warning("无法导入简化损失函数，将使用标准MSE损失")
    create_simplified_loss = None

@dataclass
class OptimizedModelConfig:
    """优化模型配置"""
    name: str
    model_type: str
    enabled: bool
    params: Dict[str, Any]
    target_params: Optional[int] = None

@dataclass
class TestResult:
    """测试结果"""
    model_name: str
    model_type: str
    train_time: float
    test_time: float
    memory_usage: float
    param_count: int
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    success: bool = True
    error_msg: str = ""
    loss_components: Optional[Dict[str, float]] = None

class PerformanceMetrics:
    """性能指标计算"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_error(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())))
    
    @staticmethod
    def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 1.0) -> float:
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        if mse == 0:
            return float('inf')
        return float(20 * np.log10(max_val / np.sqrt(mse)))
    
    @staticmethod
    def ssim_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """简化的SSIM计算"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        mu1, mu2 = np.mean(y_true_flat), np.mean(y_pred_flat)
        sigma1, sigma2 = np.std(y_true_flat), np.std(y_pred_flat)
        sigma12 = np.mean((y_true_flat - mu1) * (y_pred_flat - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return float(np.clip(ssim, 0, 1))
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    @classmethod
    def compute_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'mse': cls.mse(y_true, y_pred),
            'mae': cls.mae(y_true, y_pred),
            'rmse': cls.rmse(y_true, y_pred),
            'psnr': cls.psnr(y_true, y_pred),
            'ssim': cls.ssim_simple(y_true, y_pred),
            'correlation': cls.correlation(y_true, y_pred)
        }

class OptimizedModelFactory:
    """优化模型工厂"""
    
    @staticmethod
    def create_model(config: OptimizedModelConfig, input_dim: int, output_dim: int, 
                    input_hw: Tuple[int, int], device: torch.device) -> torch.nn.Module:
        """创建优化的模型"""
        if config.model_type == 'transformer':
            if TransformerFlowReconstructionModel is None:
                raise ImportError("Transformer模型未正确导入")
            
            # 从配置中提取参数
            params = config.params.copy()
            
            # 设置基本参数
            model_params = {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'num_heads': params.get('num_heads', 4),
                'num_layers': params.get('num_layers', 3),
                'd_model': params.get('d_model', 128),
                'attention_type': params.get('attention_type', 'simplified_self_attention'),
                'seq_len': params.get('seq_len', 32),
                'input_hw': params.get('input_hw', input_hw),
                'pe_type': params.get('pe_type', 'learnable_2d'),
                'output_head_type': params.get('output_head_type', 'global'),
                'time_encoding': params.get('time_encoding', 'embedding'),
                'use_memory_film': params.get('use_memory_film', False),
                'use_memory_concat': params.get('use_memory_concat', False)
            }
            
            logger.info(f"创建优化Transformer模型: {config.name}")
            logger.info(f"参数: d_model={model_params['d_model']}, num_heads={model_params['num_heads']}, "
                       f"num_layers={model_params['num_layers']}, attention_type={model_params['attention_type']}")
            
            model = TransformerFlowReconstructionModel(**model_params)
            model = model.to(device)
            
            return model
        else:
            raise ValueError(f"不支持的模型类型: {config.model_type}")

class OptimizedTransformerTest:
    """优化Transformer测试类"""
    
    def __init__(self, config_path: str, results_dir: str = "./optimized_results"):
        self.config_path = config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化指标计算器
        self.metrics = PerformanceMetrics()
        
        # 存储结果
        self.results = []
    
    def load_models_config(self) -> List[OptimizedModelConfig]:
        """加载模型配置"""
        models_config = []
        
        for model_name, model_info in self.config['models'].items():
            if model_info.get('enabled', False):
                target_params = model_info.get('target_params')
                if target_params is not None:
                    target_params = int(target_params) if isinstance(target_params, str) else target_params
                
                config = OptimizedModelConfig(
                    name=model_name,
                    model_type=model_info['model_type'],
                    enabled=model_info['enabled'],
                    params=model_info['params'],
                    target_params=target_params
                )
                models_config.append(config)
        
        logger.info(f"加载了 {len(models_config)} 个模型配置")
        return models_config
    
    def prepare_data(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """准备数据"""
        logger.info("准备数据...")
        
        # 创建动态配置
        dynamic_config = create_dynamic_config(
            data_path=self.config['data']['data_path'],
            input_resolution=tuple(self.config['data']['input_resolution']),
            output_resolution=tuple(self.config['data']['output_resolution']),
            num_samples=self.config['data']['num_samples'],
            batch_size=self.config['data']['batch_size']
        )
        
        # 获取数据加载器
        train_loader, valid_loader, test_loader, dataset = get_dynamic_loaders(
            dynamic_config,
            train_split=self.config['data']['train_split'],
            valid_split=self.config['data']['valid_split'],
            test_split=self.config['data']['test_split']
        )
        
        logger.info(f"数据准备完成: 训练={len(train_loader)}, 验证={len(valid_loader)}, 测试={len(test_loader)}")
        return train_loader, valid_loader, test_loader, dataset
    
    def create_loss_function(self):
        """创建损失函数"""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'mse_l1_combo')
        
        if create_simplified_loss is not None and loss_type == 'mse_l1_combo':
            # 使用简化的组合损失
            loss_fn = create_simplified_loss(
                loss_type='mse_l1_combo',
                mse_weight=loss_config.get('mse_weight', 0.8),
                l1_weight=loss_config.get('l1_weight', 0.2)
            )
            logger.info(f"使用简化MSE+L1组合损失: MSE权重={loss_config.get('mse_weight', 0.8)}, "
                       f"L1权重={loss_config.get('l1_weight', 0.2)}")
        else:
            # 回退到标准MSE损失
            loss_fn = torch.nn.MSELoss()
            logger.info("使用标准MSE损失")
        
        return loss_fn
    
    def train_and_evaluate_model(self, model_config: OptimizedModelConfig, 
                                train_loader, valid_loader, test_loader, 
                                dataset) -> TestResult:
        """训练和评估单个模型"""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试优化模型: {model_config.name} ({model_config.model_type})")
        logger.info(f"{'='*60}")
        
        try:
            # 获取数据维度信息
            sample_batch = next(iter(train_loader))
            if len(sample_batch) == 3:
                input_data, target_data, _ = sample_batch
            else:
                input_data, target_data = sample_batch
            
            input_dim = input_data.shape[-1] if len(input_data.shape) > 2 else input_data.shape[1]
            output_dim = target_data.shape[-1] if len(target_data.shape) > 2 else target_data.shape[1]
            input_hw = tuple(self.config['data']['input_resolution'])
            
            # 创建模型
            start_time = time.time()
            model = OptimizedModelFactory.create_model(model_config, input_dim, output_dim, input_hw, self.device)
            
            # 计算参数数量
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"模型参数数量: {param_count:,}")
            
            # 检查是否达到目标参数量
            if model_config.target_params and param_count > model_config.target_params * 1.2:
                logger.warning(f"参数量 {param_count:,} 超过目标 {model_config.target_params:,} 的20%")
            
            # 记录内存使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = 0
            
            # 创建优化器和损失函数
            optimizer_config = self.config.get('optimizer', {})
            training_config = self.config.get('training', {})
            
            # 确保学习率是浮点数
            lr = training_config.get('learning_rate', optimizer_config.get('lr', 2e-3))
            lr = float(lr) if isinstance(lr, str) else lr
            
            weight_decay = optimizer_config.get('weight_decay', 1e-4)
            weight_decay = float(weight_decay) if isinstance(weight_decay, str) else weight_decay
            
            eps = optimizer_config.get('eps', 1e-8)
            eps = float(eps) if isinstance(eps, str) else eps
            
            betas = optimizer_config.get('betas', [0.9, 0.999])
            if isinstance(betas, list):
                betas = [float(b) if isinstance(b, str) else b for b in betas]
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
            
            criterion = self.create_loss_function()
            
            # 学习率调度器
            scheduler_config = self.config.get('training', {}).get('scheduler', {})
            if scheduler_config.get('type') == 'cosine':
                T_max = scheduler_config.get('T_max', 15)
                T_max = int(T_max) if isinstance(T_max, str) else T_max
                
                eta_min = scheduler_config.get('eta_min', 1e-5)
                eta_min = float(eta_min) if isinstance(eta_min, str) else eta_min
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
            else:
                scheduler = None
            
            # 训练模型
            train_start = time.time()
            model.train()
            train_losses = []
            
            epochs = self.config.get('training', {}).get('epochs', 15)
            gradient_clip_config = self.config.get('training', {}).get('gradient_clipping', {})
            gradient_clip = gradient_clip_config.get('max_norm', 0.5) if isinstance(gradient_clip_config, dict) else 0.5
            gradient_clip = float(gradient_clip) if gradient_clip != 'none' else 0.0
            
            logger.info(f"开始训练 {epochs} 个epoch...")
            
            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = 0
                
                for batch_idx, batch_data in enumerate(train_loader):
                    if batch_idx >= 20:  # 限制训练批次以加快测试
                        break
                    
                    # 处理数据解包
                    if len(batch_data) == 3:
                        data, target, _ = batch_data
                    else:
                        data, target = batch_data
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 为Transformer模型添加时间步参数
                    time_steps = torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                    output = model(data, time_steps)
                    
                    # 确保输出和目标形状匹配
                    if output.shape != target.shape:
                        if len(output.shape) == 2 and len(target.shape) == 4:
                            target = target.view(target.shape[0], -1)
                        elif len(output.shape) == 4 and len(target.shape) == 2:
                            output = output.view(output.shape[0], -1)
                    
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # 梯度裁剪
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / max(1, num_batches)
                train_losses.append(avg_loss)
                
                # 更新学习率
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
            train_time = time.time() - train_start
            
            # 测试模型
            test_start = time.time()
            model.eval()
            
            all_predictions = []
            all_targets = []
            test_losses = []
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    if batch_idx >= 10:  # 限制测试批次
                        break
                    
                    # 处理数据解包
                    if len(batch_data) == 3:
                        data, target, _ = batch_data
                    else:
                        data, target = batch_data
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 为Transformer模型添加时间步参数
                    time_steps = torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                    output = model(data, time_steps)
                    
                    # 确保输出和目标形状匹配
                    if output.shape != target.shape:
                        if len(output.shape) == 2 and len(target.shape) == 4:
                            target = target.view(target.shape[0], -1)
                        elif len(output.shape) == 4 and len(target.shape) == 2:
                            output = output.view(output.shape[0], -1)
                    
                    test_loss = criterion(output, target)
                    test_losses.append(test_loss.item())
                    
                    all_predictions.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
            
            test_time = time.time() - test_start
            
            # 合并预测结果
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            # 计算性能指标
            metrics = self.metrics.compute_all_metrics(targets, predictions)
            
            # 记录内存使用
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage = (memory_after - memory_before) / 1024**2  # MB
            else:
                memory_usage = 0
            
            # 获取损失组件信息（如果支持）
            loss_components = None
            if hasattr(criterion, 'get_component_losses'):
                with torch.no_grad():
                    sample_pred = torch.tensor(predictions[:1], device=self.device)
                    sample_target = torch.tensor(targets[:1], device=self.device)
                    loss_components = criterion.get_component_losses(sample_pred, sample_target)
                    loss_components = {k: float(v.cpu()) for k, v in loss_components.items()}
            
            # 创建测试结果
            result = TestResult(
                model_name=model_config.name,
                model_type=model_config.model_type,
                train_time=train_time,
                test_time=test_time,
                memory_usage=memory_usage,
                param_count=param_count,
                metrics=metrics,
                predictions=predictions[:50],  # 只保存前50个样本
                targets=targets[:50],
                success=True,
                loss_components=loss_components
            )
            
            logger.info(f"模型 {model_config.name} 测试完成")
            logger.info(f"训练时间: {train_time:.2f}s, 测试时间: {test_time:.2f}s")
            logger.info(f"内存使用: {memory_usage:.2f}MB, 参数数量: {param_count:,}")
            logger.info(f"MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}, 相关性: {metrics['correlation']:.4f}")
            
            if loss_components:
                logger.info(f"损失组件: {loss_components}")
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"模型 {model_config.name} 测试失败: {str(e)}")
            logger.error(f"详细错误信息: {error_trace}")
            return TestResult(
                model_name=model_config.name,
                model_type=model_config.model_type,
                train_time=0,
                test_time=0,
                memory_usage=0,
                param_count=0,
                metrics={},
                success=False,
                error_msg=str(e)
            )
    
    def run_comparison(self):
        """运行模型对比测试"""
        logger.info("开始优化Transformer模型对比测试")
        
        # 加载模型配置
        models_config = self.load_models_config()
        
        # 准备数据
        train_loader, valid_loader, test_loader, dataset = self.prepare_data()
        
        # 测试每个模型
        for model_config in models_config:
            result = self.train_and_evaluate_model(
                model_config, train_loader, valid_loader, test_loader, dataset
            )
            self.results.append(result)
            
            # 清理内存
            cleanup_memory()
        
        logger.info(f"所有模型测试完成，共测试 {len(self.results)} 个模型")
    
    def generate_report(self):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        # 分离成功和失败的结果
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # 生成文本报告
        self._generate_text_report(successful_results, failed_results)
        
        # 保存详细结果
        self._save_detailed_results()
        
        # 生成可视化报告（如果有成功的结果）
        if successful_results:
            self._generate_visualization_report(successful_results)
        
        logger.info(f"报告已保存到: {self.results_dir}")
    
    def _generate_text_report(self, successful_results: List[TestResult], 
                             failed_results: List[TestResult]):
        """生成文本报告"""
        report_path = self.results_dir / "optimization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("优化Transformer模型测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总模型数: {len(self.results)}\n")
            f.write(f"成功模型数: {len(successful_results)}\n")
            f.write(f"失败模型数: {len(failed_results)}\n\n")
            
            if successful_results:
                f.write("成功模型结果:\n")
                f.write("-" * 30 + "\n")
                
                # 按PSNR排序
                successful_results.sort(key=lambda x: x.metrics.get('psnr', 0), reverse=True)
                
                for result in successful_results:
                    f.write(f"\n模型: {result.model_name}\n")
                    f.write(f"  参数数量: {result.param_count:,}\n")
                    f.write(f"  训练时间: {result.train_time:.2f}s\n")
                    f.write(f"  测试时间: {result.test_time:.2f}s\n")
                    f.write(f"  内存使用: {result.memory_usage:.2f}MB\n")
                    f.write(f"  MSE: {result.metrics.get('mse', 0):.6f}\n")
                    f.write(f"  PSNR: {result.metrics.get('psnr', 0):.2f}\n")
                    f.write(f"  相关性: {result.metrics.get('correlation', 0):.4f}\n")
                    
                    if result.loss_components:
                        f.write(f"  损失组件: {result.loss_components}\n")
            
            if failed_results:
                f.write("\n\n失败模型:\n")
                f.write("-" * 30 + "\n")
                for result in failed_results:
                    f.write(f"\n模型: {result.model_name}\n")
                    f.write(f"  错误: {result.error_msg}\n")
    
    def _generate_visualization_report(self, results: List[TestResult]):
        """生成可视化报告"""
        try:
            # 性能对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('优化Transformer模型性能对比', fontsize=16)
            
            model_names = [r.model_name for r in results]
            
            # PSNR对比
            psnr_values = [r.metrics.get('psnr', 0) for r in results]
            axes[0, 0].bar(model_names, psnr_values, color='skyblue')
            axes[0, 0].set_title('PSNR对比')
            axes[0, 0].set_ylabel('PSNR (dB)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # MSE对比
            mse_values = [r.metrics.get('mse', 0) for r in results]
            axes[0, 1].bar(model_names, mse_values, color='lightcoral')
            axes[0, 1].set_title('MSE对比')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 参数数量对比
            param_counts = [r.param_count / 1000 for r in results]  # 转换为K
            axes[1, 0].bar(model_names, param_counts, color='lightgreen')
            axes[1, 0].set_title('参数数量对比')
            axes[1, 0].set_ylabel('参数数量 (K)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 训练时间对比
            train_times = [r.train_time for r in results]
            axes[1, 1].bar(model_names, train_times, color='orange')
            axes[1, 1].set_title('训练时间对比')
            axes[1, 1].set_ylabel('训练时间 (s)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'optimization_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 效率分析图
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 参数数量 vs PSNR 散点图
            param_counts = [r.param_count / 1000 for r in results]
            psnr_values = [r.metrics.get('psnr', 0) for r in results]
            
            scatter = ax.scatter(param_counts, psnr_values, s=100, alpha=0.7, c=range(len(results)), cmap='viridis')
            
            for i, result in enumerate(results):
                ax.annotate(result.model_name, (param_counts[i], psnr_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlabel('参数数量 (K)')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('模型效率分析: 参数数量 vs 性能')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"生成可视化报告失败: {e}")
    
    def _save_detailed_results(self):
        """保存详细结果"""
        detailed_results = []
        
        for result in self.results:
            detailed_result = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'success': result.success,
                'train_time': result.train_time,
                'test_time': result.test_time,
                'memory_usage': result.memory_usage,
                'param_count': result.param_count,
                'metrics': result.metrics,
                'error_msg': result.error_msg,
                'loss_components': result.loss_components
            }
            detailed_results.append(detailed_result)
        
        # 保存为JSON
        with open(self.results_dir / 'detailed_optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    config_path = "optimized_config.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    # 创建测试实例
    tester = OptimizedTransformerTest(config_path)
    
    try:
        # 运行对比测试
        tester.run_comparison()
        
        # 生成报告
        tester.generate_report()
        
        logger.info("优化测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()