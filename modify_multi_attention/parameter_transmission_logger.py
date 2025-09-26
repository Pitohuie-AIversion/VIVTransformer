#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数传输日志和调试信息模块
提供详细的参数传输跟踪、日志记录和调试功能
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from functools import wraps
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_transmission.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ParameterTransmission')

class ParameterTracker:
    """参数传输跟踪器"""
    
    def __init__(self):
        self.transmission_log = []
        self.error_log = []
        self.start_time = time.time()
        self.checkpoints = {}
        
    def log_transmission(self, source: str, target: str, parameters: Dict[str, Any], 
                        status: str = 'success', error_msg: str = None):
        """记录参数传输"""
        timestamp = datetime.now().isoformat()
        
        transmission_record = {
            'timestamp': timestamp,
            'source': source,
            'target': target,
            'parameters': self._serialize_parameters(parameters),
            'status': status,
            'error_message': error_msg
        }
        
        self.transmission_log.append(transmission_record)
        
        if status == 'success':
            logger.info(f"✅ 参数传输成功: {source} -> {target}")
            logger.info(f"   传输参数: {list(parameters.keys())}")
        else:
            logger.error(f"❌ 参数传输失败: {source} -> {target}")
            logger.error(f"   错误信息: {error_msg}")
            self.error_log.append(transmission_record)
    
    def log_checkpoint(self, checkpoint_name: str, data: Dict[str, Any]):
        """记录检查点"""
        timestamp = datetime.now().isoformat()
        self.checkpoints[checkpoint_name] = {
            'timestamp': timestamp,
            'data': self._serialize_parameters(data)
        }
        logger.info(f"📍 检查点记录: {checkpoint_name}")
    
    def _serialize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """序列化参数，处理不可序列化的对象"""
        serialized = {}
        for key, value in params.items():
            try:
                # 尝试JSON序列化测试
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                # 不可序列化的对象转为字符串表示
                serialized[key] = f"<{type(value).__name__}: {str(value)[:100]}>"
        return serialized
    
    def get_transmission_summary(self) -> Dict[str, Any]:
        """获取传输摘要"""
        total_transmissions = len(self.transmission_log)
        successful_transmissions = len([t for t in self.transmission_log if t['status'] == 'success'])
        failed_transmissions = len(self.error_log)
        
        return {
            'total_transmissions': total_transmissions,
            'successful_transmissions': successful_transmissions,
            'failed_transmissions': failed_transmissions,
            'success_rate': successful_transmissions / total_transmissions if total_transmissions > 0 else 0,
            'execution_time': time.time() - self.start_time,
            'checkpoints_count': len(self.checkpoints)
        }
    
    def export_log(self, filename: str = None):
        """导出日志到文件"""
        if filename is None:
            filename = f"parameter_transmission_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_data = {
            'summary': self.get_transmission_summary(),
            'transmissions': self.transmission_log,
            'errors': self.error_log,
            'checkpoints': self.checkpoints
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 日志已导出到: {filename}")
        return filename

# 全局跟踪器实例
tracker = ParameterTracker()

def track_parameter_transmission(source_name: str, target_name: str = None):
    """参数传输跟踪装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            target = target_name or func.__name__
            
            # 记录输入参数
            input_params = {}
            if args:
                input_params['args'] = [str(arg)[:100] for arg in args]
            if kwargs:
                input_params.update(kwargs)
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录成功传输
                tracker.log_transmission(
                    source=source_name,
                    target=target,
                    parameters=input_params,
                    status='success'
                )
                
                return result
                
            except Exception as e:
                # 记录失败传输
                error_msg = f"{type(e).__name__}: {str(e)}"
                tracker.log_transmission(
                    source=source_name,
                    target=target,
                    parameters=input_params,
                    status='error',
                    error_msg=error_msg
                )
                
                # 记录详细错误信息
                logger.error(f"函数 {func.__name__} 执行失败:")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error(f"错误信息: {str(e)}")
                logger.error(f"调用栈: {traceback.format_exc()}")
                
                raise
        
        return wrapper
    return decorator

def log_config_loading(config_file: str, config_data: Dict[str, Any]):
    """记录配置加载"""
    tracker.log_transmission(
        source=f"配置文件: {config_file}",
        target="配置加载器",
        parameters={
            'config_keys': list(config_data.keys()),
            'config_size': len(str(config_data))
        },
        status='success' if config_data else 'error',
        error_msg=None if config_data else "配置为空"
    )

def log_model_creation(model_type: str, model_config: Dict[str, Any], success: bool, error_msg: str = None):
    """记录模型创建"""
    tracker.log_transmission(
        source="模型配置",
        target=f"{model_type}模型创建",
        parameters=model_config,
        status='success' if success else 'error',
        error_msg=error_msg
    )

def log_dataloader_creation(data_config: Dict[str, Any], success: bool, error_msg: str = None):
    """记录数据加载器创建"""
    tracker.log_transmission(
        source="数据配置",
        target="数据加载器创建",
        parameters=data_config,
        status='success' if success else 'error',
        error_msg=error_msg
    )

def log_training_setup(training_config: Dict[str, Any], success: bool, error_msg: str = None):
    """记录训练设置"""
    tracker.log_transmission(
        source="训练配置",
        target="训练循环设置",
        parameters=training_config,
        status='success' if success else 'error',
        error_msg=error_msg
    )

def create_parameter_flow_diagram():
    """创建参数流程图"""
    flow_diagram = """
    参数传输流程图
    ================
    
    配置文件 (YAML)
         ↓
    [1] 配置加载器 (load_unified_config)
         ↓
    [2] 配置验证器 (validate_config)
         ↓
    [3] 配置合并器 (merge_configs)
         ↓
    ┌─────────────────┬─────────────────┬─────────────────┐
    ↓                 ↓                 ↓                 ↓
 [4a] 数据配置      [4b] 模型配置      [4c] 训练配置      [4d] 损失配置
    ↓                 ↓                 ↓                 ↓
 [5a] 数据加载器    [5b] 模型创建      [5c] 训练循环      [5d] 损失函数
    ↓                 ↓                 ↓                 ↓
 [6a] 批次数据      [6b] 模型实例      [6c] 优化器设置    [6d] 损失计算
         ↓                 ↓                 ↓                 ↓
         └─────────────────┴─────────────────┴─────────────────┘
                                    ↓
                            [7] 训练执行
                                    ↓
                            [8] 结果输出
    
    关键传输节点:
    [1] YAML解析 -> Python字典
    [2] 类型验证和转换
    [3] 默认值填充和配置合并
    [4] 配置分发到各个模块
    [5] 参数应用到具体组件
    [6] 运行时参数使用
    [7] 训练过程参数调整
    [8] 结果和日志输出
    """
    
    return flow_diagram

def analyze_parameter_transmission():
    """分析参数传输情况"""
    summary = tracker.get_transmission_summary()
    
    print("\n" + "="*50)
    print("参数传输分析报告")
    print("="*50)
    
    print(f"\n📊 传输统计:")
    print(f"   总传输次数: {summary['total_transmissions']}")
    print(f"   成功传输: {summary['successful_transmissions']}")
    print(f"   失败传输: {summary['failed_transmissions']}")
    print(f"   成功率: {summary['success_rate']:.2%}")
    print(f"   执行时间: {summary['execution_time']:.2f}秒")
    print(f"   检查点数量: {summary['checkpoints_count']}")
    
    if tracker.error_log:
        print(f"\n❌ 错误详情:")
        for i, error in enumerate(tracker.error_log, 1):
            print(f"   {i}. {error['source']} -> {error['target']}")
            print(f"      错误: {error['error_message']}")
    
    print(f"\n📍 检查点:")
    for name, checkpoint in tracker.checkpoints.items():
        print(f"   {name}: {checkpoint['timestamp']}")
    
    # 显示流程图
    print(create_parameter_flow_diagram())
    
    return summary

def test_parameter_transmission_logging():
    """测试参数传输日志功能"""
    print("开始参数传输日志测试")
    
    # 模拟配置加载
    config_data = {
        'training': {'epochs': 50, 'learning_rate': 0.001},
        'data': {'batch_size': 32, 'max_samples': 10000}
    }
    log_config_loading('configs/unified_config.yaml', config_data)
    
    # 模拟模型创建
    model_config = {'input_dim': 1024, 'output_dim': 16384, 'dropout': 0.1}
    log_model_creation('transformer', model_config, True)
    
    # 模拟数据加载器创建
    data_config = {'batch_size': 32, 'shuffle': True, 'num_workers': 0}
    log_dataloader_creation(data_config, True)
    
    # 模拟训练设置
    training_config = {'optimizer': 'adamw', 'scheduler': 'cosine', 'device': 'cuda'}
    log_training_setup(training_config, True)
    
    # 记录检查点
    tracker.log_checkpoint('模型初始化完成', {'model_params': 1000000, 'device': 'cuda'})
    tracker.log_checkpoint('数据加载完成', {'train_samples': 8000, 'val_samples': 2000})
    
    # 模拟一个失败的传输
    log_model_creation('invalid_model', {}, False, "不支持的模型类型")
    
    # 分析结果
    analyze_parameter_transmission()
    
    # 导出日志
    log_file = tracker.export_log()
    print(f"\n📄 详细日志已保存到: {log_file}")

if __name__ == "__main__":
    test_parameter_transmission_logging()