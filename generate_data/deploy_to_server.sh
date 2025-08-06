#!/bin/bash
# 🚀 服务器自动部署脚本
# 用于在Linux服务器上快速配置和启动训练任务

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

echo "="*60
log_info "🚀 VIVTransformer 服务器部署脚本"
echo "="*60

# 1. 检查环境
log_info "🔍 检查环境..."

# 检查Python
if ! command -v python &> /dev/null; then
    log_error "Python未安装或不在PATH中"
    exit 1
fi
log_success "Python版本: $(python --version)"

# 检查PyTorch
if ! python -c "import torch" &> /dev/null; then
    log_error "PyTorch未安装"
    exit 1
fi
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
log_success "PyTorch版本: $PYTORCH_VERSION"

# 检查CUDA
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
    FILE_SIZE=$(du -h "$DATA_PATH" | cut -f1)
    log_success "数据文件存在: $DATA_PATH ($FILE_SIZE)"
else
    log_error "数据文件不存在: $DATA_PATH"
    log_info "请确保数据文件路径正确"
    exit 1
fi

# 3. 检查必要文件
log_info "📋 检查必要文件..."
REQUIRED_FILES=("$CONFIG_FILE" "$TRAINER_SCRIPT")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_success "文件存在: $file"
    else
        log_error "文件不存在: $file"
        exit 1
    fi
done

# 4. 创建结果目录
log_info "📂 创建结果目录..."
mkdir -p results/{models,logs,loss_logs,sge/{difference_results,visualization_results}}
log_success "结果目录已创建"

# 5. 显示系统资源
log_info "💻 系统资源信息:"
echo "  CPU核心数: $(nproc)"
echo "  内存信息: $(free -h | grep '^Mem:' | awk '{print $2" 总计, "$7" 可用'}')")"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl -v0 -s': '
fi

# 6. 提供运行选项
echo ""
log_info "🎯 选择运行模式:"
echo "1. 快速测试 (1轮训练, 100样本)"
echo "2. 完整训练 (后台运行)"
echo "3. 交互式训练 (前台运行)"
echo "4. 仅验证配置"
echo "5. 退出"

read -p "请选择 [1-5]: " choice

case $choice in
    1)
        log_info "🧪 启动快速测试..."
        python $TRAINER_SCRIPT --config $CONFIG_FILE --epochs 1 --num_samples 100
        ;;
    2)
        log_info "🚀 启动后台训练..."
        nohup python $TRAINER_SCRIPT --config $CONFIG_FILE > $LOG_FILE 2>&1 &
        PID=$!
        log_success "训练已启动，PID: $PID"
        log_info "监控命令: tail -f $LOG_FILE"
        log_info "停止命令: kill $PID"
        ;;
    3)
        log_info "🎮 启动交互式训练..."
        python $TRAINER_SCRIPT --config $CONFIG_FILE
        ;;
    4)
        log_info "✅ 验证配置文件..."
        python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print('配置文件验证通过')
print(f'数据路径: {config[\"data\"][\"data_path\"]}')
print(f'批次大小: {config[\"data\"][\"dataloader\"][\"batch_size\"]}')
print(f'训练轮数: {config[\"training\"][\"epochs\"]}')
print(f'样本数量: {config[\"data\"][\"num_samples\"]}')
"
        ;;
    5)
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

# 显示有用的命令
echo ""
log_info "📚 常用命令:"
echo "  监控训练: tail -f $LOG_FILE"
echo "  查看GPU: nvidia-smi"
echo "  查看进程: ps aux | grep python"
echo "  查看结果: ls -la results/"
echo "  停止训练: pkill -f $TRAINER_SCRIPT"

echo "="*60
log_success "✨ 祝您训练顺利！"
echo "="*60