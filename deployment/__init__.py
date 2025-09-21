"""
统一部署模块
提供参数量统一部署和硬件自适应优化的完整解决方案
"""

from .core.hardware_aware_deployment import (
    HardwareProfiler,
    ParameterLevelSelector, 
    AdaptiveDeploymentManager
)

from .core.unified_model_manager import (
    ModelInfo,
    ParameterCalculator,
    ResourceMonitor,
    UnifiedModelManager
)

from .testing.performance_benchmark import (
    BenchmarkMetrics,
    HardwareProfile,
    PerformanceBenchmark
)

from .testing.deployment_validation import (
    ValidationMetrics,
    ComparisonResult,
    DeploymentValidator
)

__version__ = "1.0.0"
__author__ = "VIVTransformer Team"

__all__ = [
    # 核心模块
    "HardwareProfiler",
    "ParameterLevelSelector", 
    "AdaptiveDeploymentManager",
    "ModelInfo",
    "ParameterCalculator",
    "ResourceMonitor",
    "UnifiedModelManager",
    
    # 测试模块
    "BenchmarkMetrics",
    "HardwareProfile", 
    "PerformanceBenchmark",
    "ValidationMetrics",
    "ComparisonResult",
    "DeploymentValidator",
]