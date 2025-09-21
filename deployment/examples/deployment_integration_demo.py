"""
统一部署方案集成演示
展示参数量统一部署和硬件自适应优化的完整工作流程
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入部署模块
from deployment.core.hardware_aware_deployment import AdaptiveDeploymentManager, HardwareProfiler
from deployment.core.unified_model_manager import UnifiedModelManager
from deployment.testing.performance_benchmark import PerformanceBenchmark
from deployment.testing.deployment_validation import DeploymentValidator

class UnifiedDeploymentDemo:
    """统一部署方案演示"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self.hardware_profiler = HardwareProfiler()
        self.deployment_manager = AdaptiveDeploymentManager(config_path)
        self.model_manager = UnifiedModelManager(config_path)
        self.benchmark = PerformanceBenchmark(config_path)
        self.validator = DeploymentValidator(config_path)
        
        self.demo_results = {}
    
    def run_complete_demo(self, run_benchmark: bool = True, run_validation: bool = True):
        """运行完整的部署演示"""
        print("🚀 统一部署方案完整演示开始")
        print("=" * 60)
        
        # 步骤1: 硬件环境分析
        print("\n📊 步骤1: 硬件环境分析")
        self._demo_hardware_analysis()
        
        # 步骤2: 自适应部署配置
        print("\n⚙️ 步骤2: 自适应部署配置")
        self._demo_adaptive_deployment()
        
        # 步骤3: 统一模型管理
        print("\n🎯 步骤3: 统一模型管理")
        self._demo_unified_model_management()
        
        # 步骤4: 性能基准测试 (可选)
        if run_benchmark:
            print("\n🏃 步骤4: 性能基准测试")
            self._demo_performance_benchmark()
        
        # 步骤5: 部署方案验证 (可选)
        if run_validation:
            print("\n🔍 步骤5: 部署方案验证")
            self._demo_deployment_validation()
        
        # 步骤6: 结果总结
        print("\n📋 步骤6: 结果总结")
        self._demo_results_summary()
        
        print("\n✅ 统一部署方案演示完成！")
        print("=" * 60)
    
    def _demo_hardware_analysis(self):
        """演示硬件环境分析"""
        print("  🔍 正在分析硬件环境...")
        
        # 获取硬件信息
        hardware_info = self.hardware_profiler.get_system_info()
        
        print(f"  💻 系统信息:")
        print(f"    - 操作系统: {hardware_info['platform']['system']} {hardware_info['platform']['release']}")
        print(f"    - CPU: {hardware_info['cpu']['cores']} 核心")
        print(f"    - 内存: {hardware_info['memory']['total_mb']:.0f} MB")
        
        if hardware_info['gpu']['available']:
            gpu_device = hardware_info['gpu']['devices'][0]
            print(f"    - GPU: {gpu_device['name']} ({gpu_device['memory_mb']:.0f} MB)")
        else:
            print(f"    - GPU: 未检测到")
        
        # 硬件级别评估
        tier = self.deployment_manager.get_hardware_tier()
        print(f"  🏷️ 硬件级别: {tier}")
        
        self.demo_results['hardware_analysis'] = {
            'hardware_info': hardware_info,
            'tier': tier
        }
    
    def _demo_adaptive_deployment(self):
        """演示自适应部署配置"""
        print("  ⚙️ 正在执行自适应部署配置...")
        
        # 执行自适应部署
        deployment_result = self.deployment_manager.deploy_adaptive_configuration()
        
        if deployment_result['status'] == 'success':
            print(f"  ✅ 部署配置成功")
            print(f"    - 选择级别: {deployment_result['selected_tier']}")
            print(f"    - 配置文件: {deployment_result['config_file']}")
            
            if 'resource_optimization' in deployment_result:
                optimization = deployment_result['resource_optimization']
                print(f"    - 批次大小: {optimization.get('batch_size', 'N/A')}")
                print(f"    - 工作进程: {optimization.get('num_workers', 'N/A')}")
                print(f"    - 内存优化: {'启用' if optimization.get('memory_optimization', False) else '禁用'}")
        else:
            print(f"  ❌ 部署配置失败: {deployment_result.get('error', '未知错误')}")
        
        self.demo_results['adaptive_deployment'] = deployment_result
    
    def _demo_unified_model_management(self):
        """演示统一模型管理"""
        print("  🎯 正在演示统一模型管理...")
        
        # 初始化模型管理器
        deployment_result = self.model_manager.initialize_deployment()
        print(f"  📋 模型管理器初始化: {deployment_result['status']}")
        
        # 获取模型统计信息
        model_stats = self.model_manager.get_model_statistics()
        print(f"  📊 模型统计:")
        print(f"    - 总模型数: {model_stats['total_models']}")
        print(f"    - 当前级别: {model_stats['current_tier']}")
        
        # 显示各级别模型数量
        for tier, models in model_stats['models_by_tier'].items():
            if models:
                param_counts = [m['parameter_count'] for m in models]
                avg_params = sum(param_counts) / len(param_counts)
                print(f"    - {tier}级别: {len(models)}个模型 (平均{avg_params/1e6:.1f}M参数)")
        
        # 演示模型加载和卸载
        available_models = list(model_stats['models'].keys())
        if available_models:
            test_model = available_models[0]
            print(f"  🔄 测试模型操作 ({test_model}):")
            
            # 加载模型
            model = self.model_manager.load_model(test_model)
            if model is not None:
                print(f"    ✅ 模型加载成功")
                
                # 卸载模型
                unload_result = self.model_manager.unload_model(test_model)
                if unload_result:
                    print(f"    ✅ 模型卸载成功")
                else:
                    print(f"    ❌ 模型卸载失败")
            else:
                print(f"    ❌ 模型加载失败")
        
        self.demo_results['unified_model_management'] = {
            'initialization': deployment_result,
            'statistics': model_stats
        }
    
    def _demo_performance_benchmark(self):
        """演示性能基准测试"""
        print("  🏃 正在执行性能基准测试...")
        print("    (这可能需要几分钟时间...)")
        
        try:
            # 运行快速基准测试
            benchmark_report = self.benchmark.run_comprehensive_benchmark(
                num_epochs=2,  # 减少epoch数量以加快演示
                num_samples=10  # 减少样本数量以加快演示
            )
            
            if 'test_summary' in benchmark_report:
                summary = benchmark_report['test_summary']
                print(f"  📊 基准测试结果:")
                print(f"    - 测试模型数: {summary['total_models']}")
                print(f"    - 平均训练时间: {summary['avg_training_time_per_epoch']:.3f}s/epoch")
                print(f"    - 平均推理时间: {summary['avg_inference_time_per_sample']:.6f}s/sample")
                print(f"    - 平均内存使用: {summary['avg_memory_usage_mb']:.1f}MB")
            
            # 显示性能排名
            if 'performance_ranking' in benchmark_report and benchmark_report['performance_ranking']:
                best_model = benchmark_report['performance_ranking'][0]
                print(f"  🏆 最佳性能模型: {best_model['model_name']}")
                print(f"    - 训练时间: {best_model['training_time']:.3f}s")
                print(f"    - 推理时间: {best_model['inference_time']:.6f}s")
            
            self.demo_results['performance_benchmark'] = benchmark_report
            
        except Exception as e:
            print(f"  ❌ 基准测试失败: {e}")
            self.demo_results['performance_benchmark'] = {'error': str(e)}
    
    def _demo_deployment_validation(self):
        """演示部署方案验证"""
        print("  🔍 正在执行部署方案验证...")
        
        try:
            # 运行验证测试
            validation_report = self.validator.run_comprehensive_validation()
            
            if 'validation_summary' in validation_report:
                summary = validation_report['validation_summary']
                print(f"  📋 验证结果:")
                print(f"    - 总体状态: {summary['overall_status']}")
                print(f"    - 通过测试: {summary['passed_tests']}/{summary['total_tests']}")
                print(f"    - 部署就绪: {'是' if summary['deployment_ready'] else '否'}")
                print(f"    - 总体评分: {validation_report.get('overall_validation_score', 0):.1f}/100")
                
                if summary['critical_issues']:
                    print(f"    ⚠️ 关键问题数: {len(summary['critical_issues'])}")
                
                if summary['recommendations']:
                    print(f"    💡 优化建议数: {len(summary['recommendations'])}")
            
            self.demo_results['deployment_validation'] = validation_report
            
        except Exception as e:
            print(f"  ❌ 验证测试失败: {e}")
            self.demo_results['deployment_validation'] = {'error': str(e)}
    
    def _demo_results_summary(self):
        """演示结果总结"""
        print("  📋 统一部署方案演示总结:")
        
        # 硬件适配性
        if 'hardware_analysis' in self.demo_results:
            tier = self.demo_results['hardware_analysis']['tier']
            print(f"    🏷️ 硬件级别: {tier}")
        
        # 部署状态
        if 'adaptive_deployment' in self.demo_results:
            deployment = self.demo_results['adaptive_deployment']
            status = "成功" if deployment['status'] == 'success' else "失败"
            print(f"    ⚙️ 自适应部署: {status}")
        
        # 模型管理
        if 'unified_model_management' in self.demo_results:
            management = self.demo_results['unified_model_management']
            if 'statistics' in management:
                total_models = management['statistics']['total_models']
                print(f"    🎯 模型管理: {total_models}个模型已注册")
        
        # 性能测试
        if 'performance_benchmark' in self.demo_results:
            benchmark = self.demo_results['performance_benchmark']
            if 'error' not in benchmark and 'test_summary' in benchmark:
                tested_models = benchmark['test_summary']['total_models']
                print(f"    🏃 性能测试: {tested_models}个模型已测试")
            else:
                print(f"    🏃 性能测试: 跳过或失败")
        
        # 验证结果
        if 'deployment_validation' in self.demo_results:
            validation = self.demo_results['deployment_validation']
            if 'error' not in validation and 'validation_summary' in validation:
                summary = validation['validation_summary']
                pass_rate = summary['passed_tests'] / summary['total_tests'] * 100 if summary['total_tests'] > 0 else 0
                print(f"    🔍 方案验证: {pass_rate:.0f}%通过率")
            else:
                print(f"    🔍 方案验证: 跳过或失败")
        
        # 整体评估
        print(f"\n  🎯 整体评估:")
        
        success_components = 0
        total_components = 0
        
        # 检查各组件状态
        components = [
            ('硬件分析', 'hardware_analysis'),
            ('自适应部署', 'adaptive_deployment'),
            ('模型管理', 'unified_model_management'),
            ('性能测试', 'performance_benchmark'),
            ('方案验证', 'deployment_validation')
        ]
        
        for name, key in components:
            total_components += 1
            if key in self.demo_results:
                result = self.demo_results[key]
                if 'error' not in result:
                    if key == 'adaptive_deployment':
                        if result.get('status') == 'success':
                            success_components += 1
                    else:
                        success_components += 1
        
        success_rate = (success_components / total_components) * 100 if total_components > 0 else 0
        
        if success_rate >= 80:
            status_emoji = "🟢"
            status_text = "优秀"
        elif success_rate >= 60:
            status_emoji = "🟡"
            status_text = "良好"
        else:
            status_emoji = "🔴"
            status_text = "需要改进"
        
        print(f"    {status_emoji} 系统状态: {status_text} ({success_rate:.0f}%)")
        print(f"    📈 成功组件: {success_components}/{total_components}")
        
        # 保存演示结果
        self._save_demo_results()
    
    def _save_demo_results(self):
        """保存演示结果"""
        try:
            import json
            
            output_dir = "results/demo"
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存详细结果
            output_file = f"{output_dir}/unified_deployment_demo_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.demo_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"    💾 演示结果已保存: {output_file}")
            
        except Exception as e:
            print(f"    ❌ 保存演示结果失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一部署方案演示')
    parser.add_argument('--config', default='configs/unified_adaptive_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='跳过性能基准测试')
    parser.add_argument('--skip-validation', action='store_true',
                       help='跳过部署方案验证')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print("请确保配置文件存在，或使用 --config 参数指定正确的路径")
        return 1
    
    try:
        # 创建演示实例
        demo = UnifiedDeploymentDemo(args.config)
        
        # 运行完整演示
        demo.run_complete_demo(
            run_benchmark=not args.skip_benchmark,
            run_validation=not args.skip_validation
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 演示执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())