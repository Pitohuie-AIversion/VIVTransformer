"""
部署配置模块
包含统一部署和硬件自适应的配置文件
"""

import os
from pathlib import Path

# 配置文件路径
CONFIG_DIR = Path(__file__).parent
UNIFIED_ADAPTIVE_CONFIG = CONFIG_DIR / "unified_adaptive_config.yaml"

__all__ = [
    "CONFIG_DIR",
    "UNIFIED_ADAPTIVE_CONFIG",
]