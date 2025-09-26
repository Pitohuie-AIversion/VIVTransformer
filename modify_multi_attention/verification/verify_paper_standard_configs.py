#!/usr/bin/env python3
"""
验证论文标准配置的参数量平衡性和合理性
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型创建函数
from run_crop_model_test import create_simple_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def verify_paper_standard_parameters(config_path: str):
    """验证论文标准配置的参数量"""
    logger.info("🔍 开始验证论文标准配置...")
    
    # 加载配置
    config = load_config(config_path)
    paper_config = config.get('paper_standard_comparison', {})
    
    if not paper_config.get('enabled', False):
        logger.error("❌ 论文标准对比模式未启用")
        return
    
    # 获取配置参数
    scales = paper_config.get('scales', ['small', 'medium', 'large'])
    models = paper_config.get('models', ['transformer', 'unet', 'mlp', 'fno'])
    target_params = paper_config.get('target_parameters', {})
    tolerance = paper_config.get('tolerance', 0.1)
    
    # 检查paper_standard配置是否存在
    if 'paper_standard' not in config.get('models', {}):
        logger.error("❌ 错误: models.paper_standard 配置不存在")
        return
    
    paper_standard_config = config['models']['paper_standard']
    
    logger.info(f"📊 验证规模: {scales}")
    logger.info(f"📊 验证模型: {models}")
    logger.info(f"📊 目标参数: {target_params}")
    logger.info(f"📊 容忍度: {tolerance*100:.1f}%")
    
    # 获取论文标准模型配置
    paper_models = config.get('models', {}).get('paper_standard', {})
    if not paper_models:
        logger.error("❌ 未找到论文标准模型配置")
        return

    # 验证每个规模的参数量
    all_results = {}
    
    for scale in scales:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 验证 {scale.upper()} 规模")
        logger.info(f"{'='*50}")
        
        target = target_params.get(scale, 0)
        logger.info(f"🎯 目标参数量: {target:,}")
        
        scale_results = {}
        scale_params = []
        
        for model_name in models:
            try:
                # 获取模型配置 - 修正配置路径
                model_config = paper_models.get(scale, {}).get(model_name, {})
                if not model_config:
                    logger.warning(f"⚠️  未找到 {model_name} 的 {scale} 规模配置")
                    continue
                
                # 创建模型并计算参数量
                model = create_simple_model(model_name, model_config['input_dim'], model_config['output_dim'], model_config)
                param_count = sum(p.numel() for p in model.parameters())
                
                # 计算偏差
                if target > 0:
                    deviation = abs(param_count - target) / target
                    is_within_tolerance = deviation <= tolerance
                else:
                    deviation = 0
                    is_within_tolerance = True
                
                scale_results[model_name] = {
                    'param_count': param_count,
                    'target': target,
                    'deviation': deviation,
                    'within_tolerance': is_within_tolerance,
                    'config': model_config
                }
                
                scale_params.append(param_count)
                
                # 显示结果
                status = "✅" if is_within_tolerance else "❌"
                logger.info(f"{status} {model_name.upper()}: {param_count:,} 参数 (偏差: {deviation*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ {model_name} 验证失败: {e}")
                scale_results[model_name] = {'error': str(e)}
        
        # 计算规模统计信息
        if scale_params:
            total_params = sum(scale_params)
            avg_params = total_params / len(scale_params)
            min_params = min(scale_params)
            max_params = max(scale_params)
            param_range = max_params - min_params
            balance_ratio = min_params / max_params if max_params > 0 else 0
            
            logger.info(f"\n📈 {scale.upper()} 规模统计:")
            logger.info(f"   总参数量: {total_params:,}")
            logger.info(f"   平均参数量: {avg_params:,.0f}")
            logger.info(f"   参数范围: {min_params:,} - {max_params:,}")
            logger.info(f"   参数跨度: {param_range:,}")
            logger.info(f"   平衡比例: {balance_ratio:.3f}")
            
            # 评估平衡性
            if balance_ratio >= 0.5:
                logger.info("✅ 参数平衡性: 良好")
            elif balance_ratio >= 0.3:
                logger.info("⚠️  参数平衡性: 一般")
            else:
                logger.info("❌ 参数平衡性: 较差")
            
            scale_results['statistics'] = {
                'total_params': total_params,
                'avg_params': avg_params,
                'min_params': min_params,
                'max_params': max_params,
                'param_range': param_range,
                'balance_ratio': balance_ratio
            }
        
        all_results[scale] = scale_results
    
    # 生成总体报告
    generate_verification_report(all_results, scales, models, target_params, tolerance)
    
    return all_results


def generate_verification_report(results: Dict[str, Dict[str, Any]], 
                               scales: List[str], 
                               models: List[str],
                               target_params: Dict[str, int],
                               tolerance: float):
    """生成验证报告"""
    logger.info(f"\n{'='*60}")
    logger.info("📊 生成验证报告")
    logger.info(f"{'='*60}")
    
    # 创建报告目录
    report_dir = "verification_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成详细报告
    report_path = os.path.join(report_dir, "paper_standard_verification.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 论文标准配置验证报告\n\n")
        
        # 使用datetime模块而不是torch.datetime
        from datetime import datetime
        f.write(f"**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 配置概览
        f.write("## 配置概览\n\n")
        f.write(f"- **验证规模**: {', '.join(scales)}\n")
        f.write(f"- **验证模型**: {', '.join([m.upper() for m in models])}\n")
        f.write(f"- **容忍度**: {tolerance*100:.1f}%\n\n")
        
        # 目标参数量
        f.write("## 目标参数量\n\n")
        f.write("| 规模 | 目标参数量 |\n")
        f.write("|------|------------|\n")
        for scale in scales:
            target = target_params.get(scale, 0)
            f.write(f"| {scale.capitalize()} | {target:,} |\n")
        f.write("\n")
        
        # 详细验证结果
        f.write("## 详细验证结果\n\n")
        
        for scale in scales:
            if scale not in results:
                continue
                
            f.write(f"### {scale.upper()} 规模\n\n")
            
            # 模型参数表
            f.write("| 模型 | 实际参数量 | 目标参数量 | 偏差 | 状态 |\n")
            f.write("|------|------------|------------|------|------|\n")
            
            scale_data = results[scale]
            for model in models:
                if model in scale_data and 'error' not in scale_data[model]:
                    data = scale_data[model]
                    status = "✅ 通过" if data['within_tolerance'] else "❌ 超出"
                    f.write(f"| {model.upper()} | {data['param_count']:,} | {data['target']:,} | {data['deviation']*100:.1f}% | {status} |\n")
                elif model in scale_data:
                    f.write(f"| {model.upper()} | - | - | - | ❌ 错误 |\n")
            
            f.write("\n")
            
            # 统计信息
            if 'statistics' in scale_data:
                stats = scale_data['statistics']
                f.write("#### 统计信息\n\n")
                f.write(f"- **总参数量**: {stats['total_params']:,}\n")
                f.write(f"- **平均参数量**: {stats['avg_params']:,.0f}\n")
                f.write(f"- **参数范围**: {stats['min_params']:,} - {stats['max_params']:,}\n")
                f.write(f"- **参数跨度**: {stats['param_range']:,}\n")
                f.write(f"- **平衡比例**: {stats['balance_ratio']:.3f}\n")
                
                # 平衡性评估
                if stats['balance_ratio'] >= 0.5:
                    f.write("- **平衡性评估**: ✅ 良好\n")
                elif stats['balance_ratio'] >= 0.3:
                    f.write("- **平衡性评估**: ⚠️ 一般\n")
                else:
                    f.write("- **平衡性评估**: ❌ 较差\n")
                
                f.write("\n")
        
        # 总体评估
        f.write("## 总体评估\n\n")
        
        # 计算通过率
        total_tests = 0
        passed_tests = 0
        
        for scale in scales:
            if scale in results:
                for model in models:
                    if model in results[scale] and 'error' not in results[scale][model]:
                        total_tests += 1
                        if results[scale][model]['within_tolerance']:
                            passed_tests += 1
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        f.write(f"- **总测试数**: {total_tests}\n")
        f.write(f"- **通过测试数**: {passed_tests}\n")
        f.write(f"- **通过率**: {pass_rate*100:.1f}%\n\n")
        
        if pass_rate >= 0.8:
            f.write("✅ **总体评估**: 配置质量良好，符合论文标准要求\n")
        elif pass_rate >= 0.6:
            f.write("⚠️ **总体评估**: 配置基本合理，建议进行微调\n")
        else:
            f.write("❌ **总体评估**: 配置需要重大调整\n")
        
        # 建议
        f.write("\n## 优化建议\n\n")
        
        for scale in scales:
            if scale in results and 'statistics' in results[scale]:
                stats = results[scale]['statistics']
                if stats['balance_ratio'] < 0.5:
                    f.write(f"- **{scale.capitalize()} 规模**: 参数不平衡，建议调整模型配置以减少参数差异\n")
        
        # 检查是否有模型偏差过大
        for scale in scales:
            if scale in results:
                for model in models:
                    if (model in results[scale] and 'error' not in results[scale][model] and
                        not results[scale][model]['within_tolerance']):
                        deviation = results[scale][model]['deviation']
                        f.write(f"- **{model.upper()} ({scale})**: 偏差 {deviation*100:.1f}%，建议调整配置参数\n")
    
    logger.info(f"📊 验证报告已保存: {report_path}")
    
    # 生成CSV格式的数据
    import pandas as pd
    
    csv_data = []
    for scale in scales:
        if scale in results:
            for model in models:
                if model in results[scale] and 'error' not in results[scale][model]:
                    data = results[scale][model]
                    csv_data.append({
                        'Scale': scale.capitalize(),
                        'Model': model.upper(),
                        'Actual_Parameters': data['param_count'],
                        'Target_Parameters': data['target'],
                        'Deviation_Percent': f"{data['deviation']*100:.1f}%",
                        'Within_Tolerance': data['within_tolerance'],
                        'Status': "Pass" if data['within_tolerance'] else "Fail"
                    })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(report_dir, "verification_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"📊 CSV数据已保存: {csv_path}")


def main():
    """主函数"""
    config_path = "configs/unified_training_config.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return
    
    logger.info("🚀 开始论文标准配置验证")
    
    try:
        results = verify_paper_standard_parameters(config_path)
        logger.info("✅ 验证完成！")
        
    except Exception as e:
        logger.error(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()