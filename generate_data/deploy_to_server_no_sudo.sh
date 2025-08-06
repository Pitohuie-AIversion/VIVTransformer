#!/bin/bash
# 🚀 服务器自动部署脚本 (无sudo权限版本)
# 用于在Linux服务器上快速配置和启动训练任务 - 适用于普通用户权限

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置变量
DATA_PATH="/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download/2D_DarcyFlow_beta10.0_Train.hdf5"
CONFIG_FILE="dynamic_config_server_corrected.yaml"
TRAINER_SCRIPT="dynamic_resolution_trainer.py"
LOG_FILE="training.log"

echo "=========================================================="
log_info "🚀 VIVTransformer 服务器部署脚本 (无sudo权限版本)"
echo "=========================================================="

# 1. 检查环境 (无需sudo权限)
log_info "🔍 检查环境..."

# 检查Python (用户环境)
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        log_info "请联系管理员安装Python或使用conda/virtualenv"
        exit 1
    else
        log_warning "使用python3命令"
        alias python=python3
    fi
fi
log_success "Python版本: $(python --version)"

# 检查pip (用户环境)
if ! command -v pip &> /dev/null; then
    if ! command -v pip3 &> /dev/null; then
        log_warning "pip未找到，可能需要手动安装依赖"
    else
        alias pip=pip3
    fi
fi

# 检查PyTorch
if ! python -c "import torch" &> /dev/null; then
    log_error "PyTorch未安装"
    log_info "尝试用户级安装: pip install --user torch torchvision"
    if command -v pip &> /dev/null; then
        log_info "正在尝试安装PyTorch..."
        pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
            log_warning "GPU版本安装失败，尝试CPU版本"
            pip install --user torch torchvision torchaudio
        }
    else
        log_error "请手动安装PyTorch或联系管理员"
        exit 1
    fi
fi
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
log_success "PyTorch版本: $PYTORCH_VERSION"

# 检查其他必要的Python包
log_info "🔍 检查Python依赖..."
REQUIRED_PACKAGES=("yaml" "numpy" "matplotlib" "h5py")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    log_warning "缺少以下Python包: ${MISSING_PACKAGES[*]}"
    if command -v pip &> /dev/null; then
        log_info "正在安装缺少的包..."
        for package in "${MISSING_PACKAGES[@]}"; do
            case $package in
                "yaml")
                    pip install --user PyYAML
                    ;;
                *)
                    pip install --user "$package"
                    ;;
            esac
        done
    else
        log_error "请手动安装缺少的包或联系管理员"
    fi
fi

# 检查CUDA (无需sudo权限)
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    log_success "CUDA可用，GPU数量: $GPU_COUNT"
else
    log_warning "CUDA不可用，将使用CPU训练"
fi

# 2. 检查数据文件
log_info "📁 检查数据文件..."
if [ -f "$DATA_PATH" ]; then
    if [ -r "$DATA_PATH" ]; then
        FILE_SIZE=$(du -h "$DATA_PATH" 2>/dev/null | cut -f1 || echo "未知大小")
        log_success "数据文件存在且可读: $DATA_PATH ($FILE_SIZE)"
    else
        log_error "数据文件存在但无读取权限: $DATA_PATH"
        log_info "请联系管理员修改文件权限"
        exit 1
    fi
else
    log_error "数据文件不存在: $DATA_PATH"
    log_info "请确保数据文件路径正确或联系管理员"
    exit 1
fi

# 3. 检查必要文件
log_info "📋 检查必要文件..."
REQUIRED_FILES=("$CONFIG_FILE" "$TRAINER_SCRIPT")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        if [ -r "$file" ]; then
            log_success "文件存在且可读: $file"
        else
            log_error "文件存在但无读取权限: $file"
            exit 1
        fi
    else
        log_error "文件不存在: $file"
        exit 1
    fi
done

# 4. 创建结果目录 (在用户目录下)
log_info "📂 创建结果目录..."
RESULT_DIR="./results"
mkdir -p "$RESULT_DIR"/{models,logs,loss_logs,sge/{difference_results,visualization_results}} 2>/dev/null || {
    log_warning "无法在当前目录创建results，尝试在用户主目录"
    RESULT_DIR="$HOME/vivtransformer_results"
    mkdir -p "$RESULT_DIR"/{models,logs,loss_logs,sge/{difference_results,visualization_results}}
}
log_success "结果目录已创建: $RESULT_DIR"

# 5. 显示系统资源 (无需sudo权限)
log_info "💻 系统资源信息:"
echo "  CPU核心数: $(nproc 2>/dev/null || echo '未知')"
echo "  内存信息: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $2" 总计, "$7" 可用"}' || echo '无法获取内存信息')"
echo "  当前用户: $(whoami)"
echo "  工作目录: $(pwd)"
echo "  磁盘空间: $(df -h . 2>/dev/null | tail -1 | awk '{print $4" 可用"}' || echo '无法获取磁盘信息')"

if command -v nvidia-smi &> /dev/null; then
    echo "  GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | nl -v0 -s': ' || echo "    无法获取GPU详细信息"
else
    echo "  GPU信息: nvidia-smi命令不可用"
fi

# 6. 检查写入权限
log_info "🔐 检查权限..."
if [ -w "." ]; then
    log_success "当前目录有写入权限"
else
    log_error "当前目录无写入权限，请切换到有权限的目录"
    exit 1
fi

# 7. 提供运行选项
echo ""
log_info "🎯 选择运行模式:"
echo "1. 快速测试 (1轮训练, 100样本)"
echo "2. 完整训练 (后台运行)"
echo "3. 交互式训练 (前台运行)"
echo "4. 仅验证配置"
echo "5. 检查环境依赖"
echo "6. 退出"

read -p "请选择 [1-6]: " choice

case $choice in
    1)
        log_info "🧪 启动快速测试..."
        python "$TRAINER_SCRIPT" --config "$CONFIG_FILE" --epochs 1 --num_samples 100
        ;;
    2)
        log_info "🚀 启动后台训练..."
        nohup python "$TRAINER_SCRIPT" --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
        PID=$!
        log_success "训练已启动，PID: $PID"
        log_info "监控命令: tail -f $LOG_FILE"
        log_info "停止命令: kill $PID"
        echo "$PID" > training.pid
        log_info "PID已保存到 training.pid 文件"
        ;;
    3)
        log_info "🎮 启动交互式训练..."
        python "$TRAINER_SCRIPT" --config "$CONFIG_FILE"
        ;;
    4)
        log_info "✅ 验证配置文件..."
        python -c "
import yaml
import os
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print('✅ 配置文件验证通过')
    print(f'📁 数据路径: {config[\"data\"][\"data_path\"]}')
    print(f'📦 批次大小: {config[\"data\"][\"dataloader\"][\"batch_size\"]}')
    print(f'🔄 训练轮数: {config[\"training\"][\"epochs\"]}')
    print(f'📊 样本数量: {config[\"data\"][\"num_samples\"]}')
    
    # 检查数据文件是否可访问
    data_path = config[\"data\"][\"data_path\"]
    if os.path.exists(data_path) and os.access(data_path, os.R_OK):
        print(f'✅ 数据文件可访问')
    else:
        print(f'❌ 数据文件不可访问: {data_path}')
except Exception as e:
    print(f'❌ 配置文件验证失败: {e}')
"
        ;;
    5)
        log_info "🔍 检查环境依赖..."
        python -c "
import sys
print(f'Python版本: {sys.version}')
print(f'Python路径: {sys.executable}')

try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA版本: {torch.version.cuda}')
        print(f'GPU数量: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch未安装')

try:
    import yaml
    print('PyYAML: 已安装')
except ImportError:
    print('PyYAML: 未安装')

try:
    import numpy
    print(f'NumPy版本: {numpy.__version__}')
except ImportError:
    print('NumPy: 未安装')

try:
    import h5py
    print(f'h5py版本: {h5py.__version__}')
except ImportError:
    print('h5py: 未安装')
"
        ;;
    6)
        log_info "👋 退出部署脚本"
        exit 0
        ;;
    *)
        log_error "无效选择"
        exit 1
        ;;
esac

echo ""
log_success "🎉 部署完成！"

# 显示有用的命令 (无需sudo权限)
echo ""
log_info "📚 常用命令 (无需sudo权限):"
echo "  监控训练: tail -f $LOG_FILE"
echo "  查看进程: ps aux | grep python | grep $USER"
echo "  查看结果: ls -la $RESULT_DIR/"
echo "  停止训练: kill \$(cat training.pid) 或 pkill -f $TRAINER_SCRIPT"
echo "  查看GPU使用: nvidia-smi (如果可用)"
echo "  查看磁盘使用: df -h ."
echo "  查看内存使用: free -h"

echo "=========================================================="
log_success "✨ 祝您训练顺利！(无sudo权限版本)"
log_info "💡 如遇到权限问题，请联系系统管理员"
echo "=========================================================="