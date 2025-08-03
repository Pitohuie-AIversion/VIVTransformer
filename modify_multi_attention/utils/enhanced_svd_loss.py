import torch
import torch.nn as nn
import time
import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class SVDPerformanceMonitor:
    """SVD性能和稳定性监控器"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.svd_compute_times = []
        self.fallback_usage = defaultdict(int)
        self.total_calls = 0
        self.nan_inf_count = 0
        self.zero_mode_count = 0
        self.precision_conversions = 0
        
    def record_svd_time(self, compute_time: float):
        """记录SVD计算时间"""
        self.svd_compute_times.append(compute_time)
        
    def record_fallback(self, strategy_idx: int):
        """记录fallback策略使用"""
        self.fallback_usage[strategy_idx] += 1
        
    def record_call(self):
        """记录函数调用"""
        self.total_calls += 1
        
    def record_numerical_issue(self, issue_type: str):
        """记录数值问题"""
        if issue_type == 'nan_inf':
            self.nan_inf_count += 1
        elif issue_type == 'zero_mode':
            self.zero_mode_count += 1
        elif issue_type == 'precision_conversion':
            self.precision_conversions += 1
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.svd_compute_times:
            avg_time = 0
            total_time = 0
        else:
            avg_time = sum(self.svd_compute_times) / len(self.svd_compute_times)
            total_time = sum(self.svd_compute_times)
            
        return {
            'total_calls': self.total_calls,
            'avg_svd_time_ms': avg_time * 1000,
            'total_svd_time_ms': total_time * 1000,
            'fallback_usage': dict(self.fallback_usage),
            'nan_inf_rate': self.nan_inf_count / max(1, self.total_calls),
            'zero_mode_rate': self.zero_mode_count / max(1, self.total_calls),
            'precision_conversion_rate': self.precision_conversions / max(1, self.total_calls)
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n=== SVD性能监控报告 ===")
        print(f"总调用次数: {stats['total_calls']}")
        print(f"平均SVD计算时间: {stats['avg_svd_time_ms']:.2f}ms")
        print(f"总SVD计算时间: {stats['total_svd_time_ms']:.2f}ms")
        print(f"NaN/Inf发生率: {stats['nan_inf_rate']:.2%}")
        print(f"零模态使用率: {stats['zero_mode_rate']:.2%}")
        print(f"精度转换率: {stats['precision_conversion_rate']:.2%}")
        
        if stats['fallback_usage']:
            print("Fallback策略使用情况:")
            for strategy, count in stats['fallback_usage'].items():
                print(f"  策略{strategy + 1}: {count}次")
        print("========================\n")

# 全局监控器实例
svd_monitor = SVDPerformanceMonitor()

def get_svd_modes_enhanced(tensor, topk=10, mixed_precision_mode=False, 
                          enable_monitoring=True, fallback_level=2):
    """
    增强版SVD模态提取函数
    
    Args:
        tensor: 输入张量
        topk: 提取的模态数量
        mixed_precision_mode: 是否启用混合精度优化
        enable_monitoring: 是否启用性能监控
        fallback_level: 错误处理级别 (1=基础, 2=标准, 3=完整)
    """
    if enable_monitoring:
        svd_monitor.record_call()
        start_time = time.time()
    
    # 输入维度处理
    if tensor.dim() == 2:
        N = tensor.shape[1]
        hw = int(N ** 0.5)
        assert hw * hw == N
        tensor = tensor.view(-1, hw, hw)
    
    B, H, W = tensor.shape
    modes = []
    
    # 记录原始数据类型
    original_dtype = tensor.dtype
    
    # 混合精度优化：自动检测和转换
    if mixed_precision_mode and original_dtype == torch.float16:
        if enable_monitoring:
            svd_monitor.record_numerical_issue('precision_conversion')
        # 在混合精度模式下，SVD计算使用float32以提高稳定性
        compute_dtype = torch.float32
    else:
        compute_dtype = original_dtype if original_dtype != torch.float16 else torch.float32
    
    def create_zero_mode(reference_shape, dtype):
        """统一的零模态创建函数"""
        zero_mode = torch.zeros(reference_shape, device=tensor.device, dtype=torch.float32)
        if dtype == torch.float16:
            zero_mode = zero_mode.half()
        return zero_mode
    
    # 根据fallback_level定义策略
    if fallback_level == 1:  # 基础级别
        strategies = [
            lambda t: torch.linalg.svd(t, full_matrices=False),
            lambda t: torch.linalg.svd(t + 1e-8 * torch.randn_like(t), full_matrices=False)
        ]
    elif fallback_level == 2:  # 标准级别
        strategies = [
            lambda t: torch.linalg.svd(t, full_matrices=False),
            lambda t: torch.linalg.svd(t + 1e-8 * torch.randn_like(t), full_matrices=False),
            lambda t: torch.linalg.svd(t + 1e-6 * torch.eye(min(t.shape[-2:]), device=t.device, dtype=t.dtype), full_matrices=False)
        ]
    else:  # 完整级别
        strategies = [
            lambda t: torch.linalg.svd(t, full_matrices=False),
            lambda t: torch.linalg.svd(t + 1e-8 * torch.randn_like(t), full_matrices=False),
            lambda t: torch.linalg.svd(t + 1e-6 * torch.eye(min(t.shape[-2:]), device=t.device, dtype=t.dtype), full_matrices=False),
            lambda t: torch.linalg.svd(t + 1e-4 * torch.randn_like(t), full_matrices=False),
            lambda t: torch.linalg.svd(torch.clamp(t, -1e6, 1e6), full_matrices=False)
        ]
    
    for i in range(B):
        current_tensor = tensor[i]
        
        # 精度转换
        if current_tensor.dtype != compute_dtype:
            current_tensor = current_tensor.to(compute_dtype)
        
        u, s, vh = None, None, None
        success = False
        
        for strategy_idx, strategy in enumerate(strategies):
            try:
                u, s, vh = strategy(current_tensor)
                success = True
                if strategy_idx > 0 and enable_monitoring:
                    svd_monitor.record_fallback(strategy_idx)
                break
            except (RuntimeError, torch._C._LinAlgError):
                if strategy_idx == len(strategies) - 1:
                    success = False
                    break
                continue
        
        if not success:
            if enable_monitoring:
                svd_monitor.record_numerical_issue('zero_mode')
            single_modes = [create_zero_mode(tensor[i].shape, original_dtype) for _ in range(topk)]
        else:
            # 数值稳定性检查
            s = torch.clamp(s, min=1e-12)
            
            if torch.isnan(s).any() or torch.isinf(s).any():
                if enable_monitoring:
                    svd_monitor.record_numerical_issue('nan_inf')
                single_modes = [create_zero_mode(tensor[i].shape, original_dtype) for _ in range(topk)]
            else:
                single_modes = []
                for k in range(min(topk, len(s))):
                    try:
                        mode_k = s[k] * torch.outer(u[:, k], vh[k, :])
                        
                        # 数值检查
                        if torch.isnan(mode_k).any() or torch.isinf(mode_k).any():
                            mode_k = create_zero_mode(current_tensor.shape, original_dtype)
                        else:
                            # 转换回原始数据类型
                            if original_dtype == torch.float16:
                                mode_k = mode_k.half()
                        
                        single_modes.append(mode_k)
                    except Exception:
                        single_modes.append(create_zero_mode(current_tensor.shape, original_dtype))
                
                # 填充不足的模态
                while len(single_modes) < topk:
                    single_modes.append(create_zero_mode(tensor[i].shape, original_dtype))
        
        # 组织模态数据
        for k in range(topk):
            if len(modes) <= k:
                modes.append([])
            modes[k].append(single_modes[k])
    
    # 堆叠所有批次的模态
    try:
        modes = [torch.stack(modes[k], dim=0) for k in range(topk)]
    except Exception:
        zero_mode = torch.zeros_like(tensor, dtype=torch.float32)
        if original_dtype == torch.float16:
            zero_mode = zero_mode.half()
        modes = [zero_mode for _ in range(topk)]
    
    if enable_monitoring:
        end_time = time.time()
        svd_monitor.record_svd_time(end_time - start_time)
    
    return modes

def svd_topk_losses_enhanced(pred, target, topk=10, mixed_precision_mode=False, 
                            enable_monitoring=True, fallback_level=2):
    """增强版SVD损失计算"""
    pred_modes = get_svd_modes_enhanced(pred, topk=topk, 
                                       mixed_precision_mode=mixed_precision_mode,
                                       enable_monitoring=enable_monitoring,
                                       fallback_level=fallback_level)
    target_modes = get_svd_modes_enhanced(target, topk=topk,
                                         mixed_precision_mode=mixed_precision_mode,
                                         enable_monitoring=enable_monitoring,
                                         fallback_level=fallback_level)
    losses = []
    for k in range(topk):
        loss_k = ((pred_modes[k] - target_modes[k]) ** 2).mean()
        losses.append(loss_k)
    return losses

class AdaptiveWeightManager:
    """自适应权重管理器"""
    
    def __init__(self, initial_base_weight=0.5, initial_svd_weights=None, topk=10,
                 adaptation_enabled=True, adaptation_interval=10):
        self.initial_base_weight = initial_base_weight
        self.initial_svd_weights = initial_svd_weights or [0.5 / topk] * topk
        self.topk = topk
        self.adaptation_enabled = adaptation_enabled
        self.adaptation_interval = adaptation_interval
        
        # 当前权重
        self.current_base_weight = initial_base_weight
        self.current_svd_weights = self.initial_svd_weights.copy()
        
        # 适应性调整相关
        self.epoch = 0
        self.loss_history = []
        self.weight_history = []
        
    def update_epoch(self, epoch: int, avg_loss: float):
        """更新epoch和损失历史"""
        self.epoch = epoch
        self.loss_history.append(avg_loss)
        
        if self.adaptation_enabled and epoch > 0 and epoch % self.adaptation_interval == 0:
            self._adjust_weights()
    
    def _adjust_weights(self):
        """根据训练进度调整权重"""
        if len(self.loss_history) < 2:
            return
        
        # 计算损失变化趋势
        recent_losses = self.loss_history[-self.adaptation_interval:]
        loss_trend = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]
        
        # 根据损失趋势调整权重
        if loss_trend > 0.1:  # 损失增加，增强基础权重
            adjustment_factor = 1.1
            self.current_base_weight = min(0.9, self.current_base_weight * adjustment_factor)
            # 相应减少SVD权重
            svd_reduction = (adjustment_factor - 1) * self.current_base_weight / len(self.current_svd_weights)
            self.current_svd_weights = [max(0.01, w - svd_reduction) for w in self.current_svd_weights]
        elif loss_trend < -0.1:  # 损失减少，可以增强SVD权重
            adjustment_factor = 0.95
            self.current_base_weight = max(0.1, self.current_base_weight * adjustment_factor)
            # 相应增加SVD权重
            svd_increase = (1 - adjustment_factor) * self.current_base_weight / len(self.current_svd_weights)
            self.current_svd_weights = [w + svd_increase for w in self.current_svd_weights]
        
        # 记录权重变化
        self.weight_history.append({
            'epoch': self.epoch,
            'base_weight': self.current_base_weight,
            'svd_weights': self.current_svd_weights.copy(),
            'loss_trend': loss_trend
        })
    
    def get_current_weights(self) -> Tuple[float, List[float]]:
        """获取当前权重"""
        return self.current_base_weight, self.current_svd_weights.copy()
    
    def get_adaptation_history(self) -> List[Dict]:
        """获取适应性调整历史"""
        return self.weight_history.copy()

class EnhancedTotalLossWithSVD(nn.Module):
    """增强版SVD总损失函数"""
    
    def __init__(self, base_weight=0.5, svd_weights=None, topk=10,
                 mixed_precision_mode=False, enable_monitoring=True,
                 fallback_level=2, adaptive_weights=False,
                 adaptation_interval=10, evaluation_mode=False):
        super().__init__()
        
        # 基础参数
        self.topk = topk
        self.mixed_precision_mode = mixed_precision_mode
        self.enable_monitoring = enable_monitoring
        self.fallback_level = fallback_level
        self.evaluation_mode = evaluation_mode
        
        # 权重管理
        if adaptive_weights:
            self.weight_manager = AdaptiveWeightManager(
                initial_base_weight=base_weight,
                initial_svd_weights=svd_weights,
                topk=topk,
                adaptation_enabled=True,
                adaptation_interval=adaptation_interval
            )
        else:
            self.weight_manager = AdaptiveWeightManager(
                initial_base_weight=base_weight,
                initial_svd_weights=svd_weights,
                topk=topk,
                adaptation_enabled=False
            )
        
        self.base_loss = nn.MSELoss()
        
        # 验证初始权重
        current_base, current_svd = self.weight_manager.get_current_weights()
        self._validate_weights(current_base, current_svd)
    
    def _validate_weights(self, base_weight: float, svd_weights: List[float]):
        """验证权重参数"""
        if base_weight < 0:
            raise ValueError(f"基础权重必须为非负数，当前值: {base_weight}")
        if any(w < 0 for w in svd_weights):
            raise ValueError(f"SVD权重必须为非负数，当前值: {svd_weights}")
        if len(svd_weights) != self.topk:
            raise ValueError(f"SVD权重数量({len(svd_weights)})必须等于topk({self.topk})")
    
    def update_epoch(self, epoch: int, avg_loss: float):
        """更新训练epoch信息"""
        self.weight_manager.update_epoch(epoch, avg_loss)
    
    def forward(self, pred, target):
        """前向传播"""
        # 输入有效性检查
        if torch.isnan(pred).any() or torch.isinf(pred).any() or \
           torch.isnan(target).any() or torch.isinf(target).any():
            if self.enable_monitoring:
                svd_monitor.record_numerical_issue('nan_inf')
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        # 获取当前权重
        base_weight, svd_weights = self.weight_manager.get_current_weights()
        
        # 权重归一化
        all_weights = [base_weight] + svd_weights
        weight_sum = sum(all_weights)
        if weight_sum == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        normalized_base_weight = base_weight / weight_sum
        normalized_svd_weights = [w / weight_sum for w in svd_weights]
        
        # 计算基础损失
        loss_base = self.base_loss(pred, target)
        
        if torch.isnan(loss_base) or torch.isinf(loss_base):
            if self.enable_monitoring:
                svd_monitor.record_numerical_issue('nan_inf')
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        # 在评估模式下，可以选择只使用基础损失或调整SVD权重
        if self.evaluation_mode:
            # 评估模式下降低SVD权重，增强基础MSE权重
            eval_base_weight = 0.8
            eval_svd_total = 0.2
            normalized_base_weight = eval_base_weight
            normalized_svd_weights = [eval_svd_total / len(normalized_svd_weights)] * len(normalized_svd_weights)
        
        # 计算SVD损失
        try:
            loss_svds = svd_topk_losses_enhanced(
                pred, target, topk=self.topk,
                mixed_precision_mode=self.mixed_precision_mode,
                enable_monitoring=self.enable_monitoring,
                fallback_level=self.fallback_level
            )
            
            # 检查SVD损失有效性
            valid_svd_losses = []
            for i, l in enumerate(loss_svds):
                if torch.isnan(l) or torch.isinf(l):
                    if self.enable_monitoring:
                        svd_monitor.record_numerical_issue('nan_inf')
                    valid_svd_losses.append(torch.tensor(0.0, device=l.device, dtype=l.dtype))
                else:
                    valid_svd_losses.append(l)
            loss_svds = valid_svd_losses
            
        except Exception:
            # SVD计算完全失败，只使用基础损失
            return loss_base
        
        # 计算总损失
        total_loss = normalized_base_weight * loss_base
        for w, l in zip(normalized_svd_weights, loss_svds):
            total_loss += w * l
        
        # 最终有效性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if self.enable_monitoring:
                svd_monitor.record_numerical_issue('nan_inf')
            return loss_base
        
        return total_loss
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return svd_monitor.get_stats()
    
    def print_performance_stats(self):
        """打印性能统计信息"""
        svd_monitor.print_stats()
    
    def reset_performance_stats(self):
        """重置性能统计信息"""
        svd_monitor.reset_stats()
    
    def get_weight_info(self) -> Dict:
        """获取权重信息"""
        current_base, current_svd = self.weight_manager.get_current_weights()
        return {
            'current_base_weight': current_base,
            'current_svd_weights': current_svd,
            'initial_base_weight': self.weight_manager.initial_base_weight,
            'initial_svd_weights': self.weight_manager.initial_svd_weights,
            'adaptation_enabled': self.weight_manager.adaptation_enabled,
            'adaptation_history': self.weight_manager.get_adaptation_history(),
            'topk': self.topk,
            'mixed_precision_mode': self.mixed_precision_mode,
            'fallback_level': self.fallback_level,
            'evaluation_mode': self.evaluation_mode
        }
    
    def print_weight_info(self):
        """打印权重信息"""
        info = self.get_weight_info()
        print("\n=== 增强版SVD损失权重信息 ===")
        print(f"当前基础权重: {info['current_base_weight']:.4f}")
        print(f"当前SVD权重: {[f'{w:.4f}' for w in info['current_svd_weights']]}")
        print(f"初始基础权重: {info['initial_base_weight']:.4f}")
        print(f"自适应权重: {'启用' if info['adaptation_enabled'] else '禁用'}")
        print(f"混合精度模式: {'启用' if info['mixed_precision_mode'] else '禁用'}")
        print(f"Fallback级别: {info['fallback_level']}")
        print(f"评估模式: {'启用' if info['evaluation_mode'] else '禁用'}")
        print(f"TopK模态数: {info['topk']}")
        
        if info['adaptation_history']:
            print(f"权重调整历史: {len(info['adaptation_history'])}次")
            for record in info['adaptation_history'][-3:]:  # 显示最近3次调整
                print(f"  Epoch {record['epoch']}: base={record['base_weight']:.4f}, trend={record['loss_trend']:.4f}")
        
        print("==============================\n")

# 便捷函数
def create_enhanced_svd_loss(base_weight=0.5, svd_weights=None, topk=10,
                             mixed_precision=False, adaptive_weights=False,
                             monitoring=True, fallback_level=2,
                             evaluation_mode=False):
    """创建增强版SVD损失函数的便捷函数"""
    return EnhancedTotalLossWithSVD(
        base_weight=base_weight,
        svd_weights=svd_weights,
        topk=topk,
        mixed_precision_mode=mixed_precision,
        enable_monitoring=monitoring,
        fallback_level=fallback_level,
        adaptive_weights=adaptive_weights,
        evaluation_mode=evaluation_mode
    )

# 全局监控器访问函数
def get_global_svd_stats():
    """获取全局SVD统计信息"""
    return svd_monitor.get_stats()

def print_global_svd_stats():
    """打印全局SVD统计信息"""
    svd_monitor.print_stats()

def reset_global_svd_stats():
    """重置全局SVD统计信息"""
    svd_monitor.reset_stats()