#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的论文对比测试脚本
避免复杂的递归调用和多进程问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 避免循环导入，直接定义简单模型
class SimpleMLP(nn.Module):
    """简单的MLP模型"""
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [min(input_dim * 2, 2048), min(input_dim, 1024), max(output_dim // 2, 256)]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MLP模型: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}, 参数量: {total_params:,}")
    
    def forward(self, x):
        return self.model(x)

class SimpleTransformer(nn.Module):
    """简单的Transformer模型"""
    def __init__(self, input_dim, output_dim, d_model=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Transformer模型: {input_dim} -> {d_model} -> {output_dim}, 参数量: {total_params:,}")
    
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.transformer(x).squeeze(1)  # (batch, d_model)
        return self.output_projection(x)

class SimpleFNO(nn.Module):
    """简单的FNO替代模型"""
    def __init__(self, input_dim, output_dim, width=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, output_dim)
        )
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"FNO模型: {input_dim} -> {output_dim}, 参数量: {total_params:,}")
    
    def forward(self, x):
        return self.model(x)

class SimpleUNet(nn.Module):
    """简单的UNet替代模型"""
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"UNet模型: {input_dim} -> {output_dim}, 参数量: {total_params:,}")
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

def load_data_simple(config):
    """简化的数据加载"""
    try:
        # 尝试加载真实数据
        sys.path.append('.')
        from data.enhanced_crop_dataloader import create_enhanced_crop_dataloader
        logger.info("✅ 使用真实数据")
        return create_enhanced_crop_dataloader(config)
    except Exception as e:
        logger.warning(f"真实数据加载失败: {e}，使用合成数据")
        # 创建合成数据
        batch_size = config['data']['batch_size']
        input_dim = config['data']['input_dim']
        output_dim = config['data']['output_dim']
        
        class SimpleDataset:
            def __init__(self, size=1000):
                self.size = size
                self.inputs = torch.randn(size, input_dim) * 0.5
                self.outputs = torch.randn(size, output_dim) * 0.5
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.outputs[idx]
        
        from torch.utils.data import DataLoader
        dataset = SimpleDataset(1000)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, None

def train_simple_model(model, train_loader, val_loader, config, device):
    """简化的模型训练"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    epochs = config['training']['epochs']
    train_losses = []
    val_losses = []
    
    logger.info(f"开始训练，共 {epochs} 轮")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_simple_model(model, test_loader, device):
    """简化的模型评估"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    
    # 计算R²
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mse': avg_loss,
        'r2_score': r2_score
    }

def run_paper_comparison_simple(config_path):
    """运行简化的论文对比实验"""
    logger.info("🎯 开始简化论文对比实验")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, normalizer = load_data_simple(config)
    
    # 获取数据维度
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    output_dim = sample_batch[1].shape[-1]
    logger.info(f"数据维度: {input_dim} -> {output_dim}")
    
    # 定义要测试的模型（根据配置文件中的参数）
    models_config = config.get('models', {})
    models_to_test = {}
    
    if 'transformer' in models_config:
        tf_config = models_config['transformer']
        models_to_test['transformer'] = lambda: SimpleTransformer(
            input_dim, output_dim,
            d_model=tf_config.get('d_model', 128),
            num_heads=tf_config.get('num_heads', 8),
            num_layers=tf_config.get('num_layers', 4)
        )
    
    if 'mlp' in models_config:
        mlp_config = models_config['mlp']
        models_to_test['mlp'] = lambda: SimpleMLP(
            input_dim, output_dim,
            hidden_dims=mlp_config.get('hidden_dims', [2048, 1024, 512])
        )
    
    if 'fno' in models_config:
        fno_config = models_config['fno']
        models_to_test['fno'] = lambda: SimpleFNO(
            input_dim, output_dim,
            width=fno_config.get('width', 64)
        )
    
    if 'unet' in models_config:
        models_to_test['unet'] = lambda: SimpleUNet(input_dim, output_dim, hidden_dim=512)
    
    results = {}
    
    # 测试每个模型
    for model_name, model_creator in models_to_test.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            # 创建模型
            model = model_creator()
            param_count = sum(p.numel() for p in model.parameters())
            
            # 训练模型
            start_time = time.time()
            train_losses, val_losses = train_simple_model(model, train_loader, val_loader, config, device)
            train_time = time.time() - start_time
            
            # 评估模型
            eval_results = evaluate_simple_model(model, test_loader, device)
            
            # 保存结果
            results[model_name] = {
                'param_count': param_count,
                'train_time': train_time,
                'test_mse': eval_results['mse'],
                'test_r2': eval_results['r2_score'],
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            logger.info(f"✅ {model_name} 完成: MSE={eval_results['mse']:.6f}, R²={eval_results['r2_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ {model_name} 失败: {e}")
            results[model_name] = {'error': str(e)}
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"simple_paper_comparison_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 简化论文对比实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {config_path}\n\n")
        
        f.write("## 模型性能对比\n\n")
        f.write("| 模型 | 参数量 | 测试MSE | 测试R² | 训练时间(s) | 状态 |\n")
        f.write("|------|--------|---------|--------|-------------|------|\n")
        
        successful_models = []
        for model_name, result in results.items():
            if 'error' in result:
                f.write(f"| {model_name} | - | - | - | - | ❌ 失败 |\n")
            else:
                f.write(f"| {model_name} | {result['param_count']:,} | {result['test_mse']:.6f} | "
                       f"{result['test_r2']:.4f} | {result['train_time']:.2f} | ✅ 成功 |\n")
                successful_models.append((model_name, result))
        
        # 性能排名
        if successful_models:
            f.write("\n## 性能排名\n\n")
            f.write("### 按MSE排名 (越小越好)\n")
            sorted_by_mse = sorted(successful_models, key=lambda x: x[1]['test_mse'])
            for i, (name, result) in enumerate(sorted_by_mse, 1):
                f.write(f"{i}. **{name}**: MSE={result['test_mse']:.6f}\n")
            
            f.write("\n### 按R²排名 (越大越好)\n")
            sorted_by_r2 = sorted(successful_models, key=lambda x: x[1]['test_r2'], reverse=True)
            for i, (name, result) in enumerate(sorted_by_r2, 1):
                f.write(f"{i}. **{name}**: R²={result['test_r2']:.4f}\n")
            
            f.write("\n### 参数效率分析\n")
            for name, result in successful_models:
                efficiency = result['test_r2'] / (result['param_count'] / 1000000)  # R²/百万参数
                f.write(f"- **{name}**: {efficiency:.4f} (R²/百万参数)\n")
    
    logger.info(f"📊 报告已保存: {report_path}")
    return results

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/paper_standard_small.yaml"
    run_paper_comparison_simple(config_path)