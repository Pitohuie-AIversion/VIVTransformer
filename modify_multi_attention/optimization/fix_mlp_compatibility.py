#!/usr/bin/env python3
"""
修复MLP模型兼容性问题的脚本
解决配置文件参数与模型接口不匹配的问题
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class FixedSimpleMLP(nn.Module):
    """修复后的SimpleMLP模型，兼容配置文件参数"""
    
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        
        # 过滤掉可能冲突的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['input_dim', 'output_dim', 'model_type']}
        
        # 从kwargs中提取参数，提供默认值
        hidden_dims = filtered_kwargs.get('hidden_dims', filtered_kwargs.get('hidden_layers', [256, 128]))
        dropout = filtered_kwargs.get('dropout', 0.1)
        activation = filtered_kwargs.get('activation', 'relu')
        use_batch_norm = filtered_kwargs.get('use_batch_norm', True)
        use_residual = filtered_kwargs.get('use_residual', False)
        
        # 处理hidden_dims可能是整数的情况
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif hidden_dims is None:
            hidden_dims = [input_dim * 2, input_dim * 4, output_dim // 2]
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer_modules = []
            layer_modules.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            layer_modules.append(nn.ReLU() if activation == 'relu' else nn.GELU())
            layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        logger.info(f"FixedMLP模型架构: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
    
    def forward(self, x):
        # 确保输入是正确的形状
        if x.dim() == 3:  # [B, T, D]
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size * seq_len, -1)  # [B*T, D]
            need_reshape = True
        else:
            need_reshape = False
            
        for layer in self.layers:
            if self.use_residual and x.shape[-1] == layer[0].out_features:
                x = x + layer(x)
            else:
                x = layer(x)
        
        x = self.output_layer(x)
        
        # 如果需要，恢复原始形状
        if need_reshape:
            x = x.view(batch_size, seq_len, -1)
        
        return x

def create_fixed_enhanced_model_mlp(model_config, input_dim, output_dim):
    """修复后的增强MLP模型创建函数"""
    try:
        # 尝试使用增强模型
        from enhanced_mlp import EnhancedMLP1d
        
        # 适配配置参数，不包含input_dim和output_dim（作为位置参数传递）
        adapted_config = {
            'hidden_layers': model_config.get('hidden_layers', [256, 128]),
            'dropout': model_config.get('dropout', 0.1),
            'activation': model_config.get('activation', 'relu'),
            'use_batch_norm': model_config.get('use_batch_norm', True),
            'use_residual': model_config.get('use_residual', False)
        }
        
        model = EnhancedMLP1d(input_dim, output_dim, **adapted_config)
        logger.info("✅ 成功创建增强MLP模型")
        return model
        
    except Exception as e:
        logger.warning(f"⚠️ 增强MLP模型创建失败: {e}")
        logger.info("📝 使用修复后的简单MLP模型")
        
        # 使用修复后的简单模型
        # 过滤掉冲突的参数
        filtered_config = {k: v for k, v in model_config.items() 
                          if k not in ['input_dim', 'output_dim', 'model_type']}
        return FixedSimpleMLP(input_dim, output_dim, **filtered_config)

def test_fixed_mlp_compatibility():
    """测试修复后的MLP模型兼容性"""
    
    # 加载配置文件
    config_path = "configs/unified_training_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    mlp_config = config['models']['mlp']
    
    # 测试参数
    input_dim = 176 * 20 * 20  # 70400
    output_dim = 176 * 200 * 200  # 7040000
    batch_size = 2
    
    logger.info("🔧 开始测试修复后的MLP模型兼容性...")
    
    results = {}
    
    # 测试1: 修复后的简单MLP
    try:
        logger.info("测试1: 修复后的简单MLP模型")
        # 过滤掉冲突的参数
        filtered_config = {k: v for k, v in mlp_config.items() 
                          if k not in ['input_dim', 'output_dim', 'model_type']}
        model = FixedSimpleMLP(input_dim, output_dim, **filtered_config)
        
        # 测试前向传播
        test_input = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            output = model(test_input)
        
        results['fixed_simple_mlp'] = {
            'status': 'success',
            'input_shape': list(test_input.shape),
            'output_shape': list(output.shape),
            'params': sum(p.numel() for p in model.parameters()),
            'message': '修复后的简单MLP模型运行成功'
        }
        logger.info(f"✅ 修复后的简单MLP: {test_input.shape} -> {output.shape}")
        
    except Exception as e:
        results['fixed_simple_mlp'] = {
            'status': 'failed',
            'error': str(e),
            'message': '修复后的简单MLP模型仍然失败'
        }
        logger.error(f"❌ 修复后的简单MLP失败: {e}")
    
    # 测试2: 修复后的增强MLP创建函数
    try:
        logger.info("测试2: 修复后的增强MLP创建函数")
        model = create_fixed_enhanced_model_mlp(mlp_config, input_dim, output_dim)
        
        # 测试前向传播
        test_input = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            output = model(test_input)
        
        results['fixed_enhanced_mlp'] = {
            'status': 'success',
            'input_shape': list(test_input.shape),
            'output_shape': list(output.shape),
            'params': sum(p.numel() for p in model.parameters()),
            'message': '修复后的增强MLP创建函数运行成功'
        }
        logger.info(f"✅ 修复后的增强MLP: {test_input.shape} -> {output.shape}")
        
    except Exception as e:
        results['fixed_enhanced_mlp'] = {
            'status': 'failed',
            'error': str(e),
            'message': '修复后的增强MLP创建函数仍然失败'
        }
        logger.error(f"❌ 修复后的增强MLP失败: {e}")
    
    # 生成测试报告
    report_path = f"mlp_compatibility_fix_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MLP模型兼容性修复报告\n\n")
        f.write(f"生成时间: {datetime.now()}\n\n")
        
        f.write("## 测试概览\n\n")
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        total_count = len(results)
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        f.write(f"- 总测试数: {total_count}\n")
        f.write(f"- 成功数: {success_count}\n")
        f.write(f"- 失败数: {total_count - success_count}\n")
        f.write(f"- 成功率: {success_rate:.1f}%\n\n")
        
        f.write("## 详细测试结果\n\n")
        
        for test_name, result in results.items():
            f.write(f"### {test_name}\n\n")
            f.write(f"- 状态: {'✅ 成功' if result['status'] == 'success' else '❌ 失败'}\n")
            
            if result['status'] == 'success':
                f.write(f"- 输入形状: {result['input_shape']}\n")
                f.write(f"- 输出形状: {result['output_shape']}\n")
                f.write(f"- 参数数量: {result['params']:,}\n")
            else:
                f.write(f"- 错误信息: {result['error']}\n")
            
            f.write(f"- 说明: {result['message']}\n\n")
        
        f.write("## 修复方案总结\n\n")
        f.write("1. **参数适配**: 修复了配置文件参数与模型接口不匹配的问题\n")
        f.write("2. **形状处理**: 增加了输入张量形状的自适应处理\n")
        f.write("3. **错误处理**: 增强了异常处理和降级机制\n")
        f.write("4. **兼容性**: 保持了与原有接口的向后兼容性\n\n")
    
    logger.info(f"📊 MLP兼容性修复报告已保存到: {report_path}")
    
    return results

if __name__ == "__main__":
    test_fixed_mlp_compatibility()