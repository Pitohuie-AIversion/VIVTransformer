# 统一部署模块 (Deployment Module)

本模块提供参数量统一部署和硬件自适应优化的完整解决方案。

## 文件夹结构

```
deployment/
├── __init__.py                 # 主模块导入接口
├── README.md                   # 模块说明文档
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── hardware_aware_deployment.py    # 硬件感知部署策略
│   └── unified_model_manager.py        # 统一模型管理器
├── testing/                    # 测试验证模块
│   ├── __init__.py
│   ├── performance_benchmark.py        # 性能基准测试
│   └── deployment_validation.py        # 部署方案验证
├── examples/                   # 示例演示模块
│   ├── __init__.py
│   └── deployment_integration_demo.py  # 集成演示脚本
└── configs/                    # 配置文件模块
    ├── __init__.py
    └── unified_adaptive_config.yaml    # 统一自适应配置
```

## 核心功能

### 1. 硬件感知部署 (core/hardware_aware_deployment.py)
- **HardwareProfiler**: 硬件性能分析器
- **ParameterLevelSelector**: 参数量级别选择器
- **AdaptiveDeploymentManager**: 自适应部署管理器

### 2. 统一模型管理 (core/unified_model_manager.py)
- **ModelInfo**: 模型信息数据类
- **ParameterCalculator**: 参数量计算器
- **ResourceMonitor**: 资源监控器
- **UnifiedModelManager**: 统一模型管理器

### 3. 性能基准测试 (testing/performance_benchmark.py)
- **BenchmarkMetrics**: 基准测试指标
- **HardwareProfile**: 硬件配置档案
- **PerformanceBenchmark**: 性能基准测试器

### 4. 部署验证 (testing/deployment_validation.py)
- **ValidationMetrics**: 验证指标
- **ComparisonResult**: 对比结果
- **DeploymentValidator**: 部署验证器

## 使用方法

### 方法1: 直接导入主模块
```python
from deployment import (
    AdaptiveDeploymentManager,
    UnifiedModelManager,
    PerformanceBenchmark,
    DeploymentValidator
)
```

### 方法2: 导入子模块
```python
from deployment.core import AdaptiveDeploymentManager, UnifiedModelManager
from deployment.testing import PerformanceBenchmark, DeploymentValidator
```

### 方法3: 运行完整演示
```python
from deployment.examples import UnifiedDeploymentDemo

# 创建演示实例
demo = UnifiedDeploymentDemo("deployment/configs/unified_adaptive_config.yaml")

# 运行完整部署流程
demo.run_complete_deployment_workflow()
```

## 配置文件

统一自适应配置文件位于 `configs/unified_adaptive_config.yaml`，包含：
- 硬件感知配置
- 参数量级别定义
- 自适应资源管理
- 数据与训练配置

## 输出结果

运行部署流程后，将生成以下文件：
- `deployment_report_*.json`: 部署配置报告
- `benchmark_results_*.json`: 性能基准测试结果
- `validation_report_*.json`: 验证结果报告
- `*.png`: 可视化图表文件

## 特性优势

1. **参数量统一管理**: 标准化不同模型的参数量级别
2. **硬件自适应优化**: 根据硬件性能自动选择最优配置
3. **性能基准测试**: 全面评估不同硬件配置下的模型性能
4. **部署方案验证**: 验证统一部署方案的有效性和优化效果
5. **模块化设计**: 清晰的文件夹结构，便于维护和扩展