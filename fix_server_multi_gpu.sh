#!/bin/bash
# 服务器多GPU错误一键修复脚本
# 使用方法: bash fix_server_multi_gpu.sh

echo "=========================================================="
echo "🔧 服务器多GPU错误一键修复脚本"
echo "=========================================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：打印彩色消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查是否在正确的目录
print_info "检查项目目录..."
if [ ! -d "modify_multi_attention" ]; then
    print_error "未找到 modify_multi_attention 目录"
    print_error "请确保在项目根目录运行此脚本"
    exit 1
fi

if [ ! -f "modify_multi_attention/training/trainer.py" ]; then
    print_error "未找到 trainer.py 文件"
    exit 1
fi

print_success "找到项目文件"

# 备份原文件
print_info "备份原文件..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cp modify_multi_attention/training/trainer.py modify_multi_attention/training/trainer.py.backup_$TIMESTAMP
print_success "已备份到: trainer.py.backup_$TIMESTAMP"

# 检查是否已经修复
print_info "检查是否已经修复..."
if grep -q "if not isinstance(model, torch.nn.DataParallel):" modify_multi_attention/training/trainer.py; then
    print_success "文件已经包含修复代码"
else
    print_info "开始修复 trainer.py 文件..."
    
    # 创建修复后的文件
    python3 << 'EOF'
import re

# 读取文件
with open('modify_multi_attention/training/trainer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找并替换
old_pattern = r'(\s+)model\.to\(device\)'
new_code = '''\g<1># 只有在模型不是DataParallel时才移动到device
\g<1># DataParallel模型已经在主程序中正确设置了设备
\g<1>if not isinstance(model, torch.nn.DataParallel):
\g<1>    model.to(device)'''

# 应用修复
fixed_content = re.sub(old_pattern, new_code, content)

# 写入文件
with open('modify_multi_attention/training/trainer.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("修复完成")
EOF
    
    if [ $? -eq 0 ]; then
        print_success "trainer.py 修复完成"
    else
        print_error "修复失败"
        exit 1
    fi
fi

# 验证修复
print_info "验证修复..."
if grep -q "if not isinstance(model, torch.nn.DataParallel):" modify_multi_attention/training/trainer.py; then
    print_success "修复验证成功"
else
    print_error "修复验证失败"
    exit 1
fi

# 清理Python缓存
print_info "清理Python缓存文件..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
print_success "缓存清理完成"

# 检查环境信息
print_info "检查环境信息..."
echo "Python版本: $(python --version 2>&1)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "CUDA版本: $(nvcc --version 2>/dev/null | grep 'release' || echo '未找到CUDA')"
echo "可用GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"

# 显示修复后的代码
print_info "修复后的关键代码:"
echo "----------------------------------------"
grep -A 3 -B 1 "isinstance.*DataParallel" modify_multi_attention/training/trainer.py
echo "----------------------------------------"

echo ""
echo "=========================================================="
print_success "修复完成！"
echo "=========================================================="
echo ""
print_info "接下来的步骤:"
echo "1. 运行训练脚本:"
echo "   python generate_data/dynamic_resolution_trainer.py --config dynamic_config.yaml"
echo ""
echo "2. 确认多GPU训练正常工作，应该看到:"
echo "   🚀 启用多GPU训练: 检测到 X 张GPU"
echo "   训练正常进行，无设备错误"
echo ""
print_info "如果仍有问题，请检查:"
echo "- Python环境是否正确"
echo "- PyTorch版本是否兼容"
echo "- 是否有权限问题"
echo ""
print_success "修复脚本执行完成！"