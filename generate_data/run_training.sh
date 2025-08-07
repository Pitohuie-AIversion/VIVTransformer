#!/bin/bash

# ================================================================
# 分离式训练启动脚本 (Linux Shell版本)
# ================================================================

# 设置脚本选项
set -e  # 遇到错误时退出
set -u  # 使用未定义变量时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印横幅
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    分离式训练启动器                          ║"
    echo "║                                                              ║"
    echo "║  🚀 自动化数据预处理和模型训练流程                           ║"
    echo "║  💡 解决服务器CPU锁定问题的最佳方案                          ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 打印帮助信息
print_help() {
    echo -e "${YELLOW}分离式训练启动器帮助${NC}"
    echo ""
    echo "使用方法:"
    echo "  $0 <数据文件路径> [选项]"
    echo ""
    echo "选项:"
    echo "  --config <配置文件>        配置文件路径 (默认: dynamic_config_server_optimized_final.yaml)"
    echo "  --output-dir <目录>       输出目录 (默认: preprocessed_data)"
    echo "  --num-samples <数量>      样本数量"
    echo "  --resume <检查点>         恢复训练的检查点"
    echo "  --mode <模式>             运行模式: auto/preprocess/train (默认: auto)"
    echo "  --auto-optimize           自动优化配置"
    echo "  --background              后台运行"
    echo "  --screen                  在screen会话中运行"
    echo "  --help                    显示帮助信息"
    echo ""
    echo "运行模式:"
    echo "  auto        - 自动模式，先预处理再训练 (默认)"
    echo "  preprocess  - 仅执行数据预处理"
    echo "  train       - 仅执行模型训练"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/data.h5"
    echo "  $0 /path/to/data.h5 --num-samples 1000 --auto-optimize"
    echo "  $0 /path/to/data.h5 --mode train --resume checkpoint_epoch_50.pth"
    echo "  $0 /path/to/data.h5 --background  # 后台运行"
    echo "  $0 /path/to/data.h5 --screen      # 在screen会话中运行"
    echo ""
    echo "功能特点:"
    echo "  ✅ 自动检测是否需要重新预处理数据"
    echo "  ✅ 智能资源配置优化"
    echo "  ✅ 支持断点续训"
    echo "  ✅ 实时显示训练进度"
    echo "  ✅ 支持后台运行和screen会话"
    echo ""
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}🔍 检查依赖...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}❌ 错误: 未找到Python解释器${NC}"
        exit 1
    fi
    
    # 确定Python命令
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    echo -e "${GREEN}✅ Python: $(${PYTHON_CMD} --version)${NC}"
    
    # 检查必需的Python包
    local packages=("torch" "numpy" "h5py" "yaml" "tqdm")
    for package in "${packages[@]}"; do
        if ! ${PYTHON_CMD} -c "import ${package}" &> /dev/null; then
            echo -e "${YELLOW}⚠️ 警告: Python包 '${package}' 未安装${NC}"
        fi
    done
}

# 检查系统资源
check_system_resources() {
    echo -e "${BLUE}📊 系统资源检测:${NC}"
    
    # CPU信息
    local cpu_cores=$(nproc)
    echo "  CPU核心数: ${cpu_cores}"
    
    # 内存信息
    if command -v free &> /dev/null; then
        local memory_info=$(free -h | grep '^Mem:')
        echo "  内存信息: ${memory_info}"
    fi
    
    # 磁盘空间
    local disk_space=$(df -h . | tail -1 | awk '{print $4}')
    echo "  可用磁盘空间: ${disk_space}"
    
    # GPU信息
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo "  GPU数量: ${gpu_count}"
        if [ ${gpu_count} -gt 0 ]; then
            echo "  GPU信息:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | \
                awk '{printf "    GPU %d: %s\n", NR-1, $0}'
        fi
    else
        echo "  GPU: 未检测到nvidia-smi"
    fi
    echo ""
}

# 运行命令的函数
run_command() {
    local cmd="$1"
    local mode="$2"
    local log_file="$3"
    
    case "${mode}" in
        "normal")
            echo -e "${GREEN}🚀 执行命令: ${cmd}${NC}"
            eval "${cmd}"
            ;;
        "background")
            echo -e "${GREEN}🚀 后台执行命令: ${cmd}${NC}"
            echo "📝 日志文件: ${log_file}"
            nohup bash -c "${cmd}" > "${log_file}" 2>&1 &
            local pid=$!
            echo "🔢 进程ID: ${pid}"
            echo "💡 使用以下命令监控进度:"
            echo "   tail -f ${log_file}"
            echo "   ps aux | grep ${pid}"
            ;;
        "screen")
            local session_name="training_$(date +%Y%m%d_%H%M%S)"
            echo -e "${GREEN}🚀 在screen会话中执行: ${session_name}${NC}"
            
            if ! command -v screen &> /dev/null; then
                echo -e "${RED}❌ 错误: screen未安装${NC}"
                echo "请安装screen: sudo apt-get install screen 或 sudo yum install screen"
                exit 1
            fi
            
            screen -dmS "${session_name}" bash -c "${cmd}; echo '训练完成，按任意键退出'; read"
            echo "📺 Screen会话已创建: ${session_name}"
            echo "💡 使用以下命令连接到会话:"
            echo "   screen -r ${session_name}"
            echo "💡 分离会话: Ctrl+A+D"
            echo "💡 查看所有会话: screen -ls"
            ;;
    esac
}

# 主函数
main() {
    # 打印横幅
    print_banner
    
    # 默认参数
    local data_path=""
    local config_file="dynamic_config_server_optimized_final.yaml"
    local output_dir="preprocessed_data"
    local num_samples=""
    local resume_checkpoint=""
    local mode="auto"
    local auto_optimize=false
    local run_mode="normal"  # normal, background, screen
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                config_file="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --num-samples)
                num_samples="$2"
                shift 2
                ;;
            --resume)
                resume_checkpoint="$2"
                shift 2
                ;;
            --mode)
                mode="$2"
                shift 2
                ;;
            --auto-optimize)
                auto_optimize=true
                shift
                ;;
            --background)
                run_mode="background"
                shift
                ;;
            --screen)
                run_mode="screen"
                shift
                ;;
            --help)
                print_help
                exit 0
                ;;
            -*)
                echo -e "${RED}❌ 错误: 未知选项 $1${NC}"
                print_help
                exit 1
                ;;
            *)
                if [[ -z "${data_path}" ]]; then
                    data_path="$1"
                else
                    echo -e "${RED}❌ 错误: 多余的参数 $1${NC}"
                    print_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 检查必需参数
    if [[ -z "${data_path}" ]]; then
        echo -e "${RED}❌ 错误: 请提供数据文件路径${NC}"
        print_help
        exit 1
    fi
    
    # 检查文件存在性
    if [[ ! -f "${data_path}" ]]; then
        echo -e "${RED}❌ 错误: 数据文件不存在: ${data_path}${NC}"
        exit 1
    fi
    
    if [[ ! -f "${config_file}" ]]; then
        echo -e "${RED}❌ 错误: 配置文件不存在: ${config_file}${NC}"
        exit 1
    fi
    
    # 检查依赖
    check_dependencies
    
    # 检查系统资源
    check_system_resources
    
    # 显示运行参数
    echo -e "${PURPLE}📋 运行参数:${NC}"
    echo "  数据文件: ${data_path}"
    echo "  配置文件: ${config_file}"
    echo "  输出目录: ${output_dir}"
    [[ -n "${num_samples}" ]] && echo "  样本数量: ${num_samples}"
    [[ -n "${resume_checkpoint}" ]] && echo "  恢复检查点: ${resume_checkpoint}"
    echo "  运行模式: ${mode}"
    echo "  自动优化: ${auto_optimize}"
    echo "  执行方式: ${run_mode}"
    echo ""
    
    # 构建Python命令
    local python_cmd="${PYTHON_CMD} run_separated_training.py"
    python_cmd+="  --data_path '${data_path}'"
    python_cmd+=" --config '${config_file}'"
    python_cmd+=" --output_dir '${output_dir}'"
    python_cmd+=" --mode ${mode}"
    
    [[ -n "${num_samples}" ]] && python_cmd+=" --num_samples ${num_samples}"
    [[ -n "${resume_checkpoint}" ]] && python_cmd+=" --resume '${resume_checkpoint}'"
    [[ "${auto_optimize}" == true ]] && python_cmd+=" --auto_optimize"
    
    # 设置日志文件
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="training_${timestamp}.log"
    
    # 执行命令
    run_command "${python_cmd}" "${run_mode}" "${log_file}"
    
    # 如果是正常模式，显示结果
    if [[ "${run_mode}" == "normal" ]]; then
        local exit_code=$?
        
        if [[ ${exit_code} -eq 0 ]]; then
            echo ""
            echo -e "${GREEN}🎉 分离式训练流程完成！${NC}"
            echo ""
            echo -e "${BLUE}📁 生成的文件:${NC}"
            
            if [[ -d "${output_dir}" ]]; then
                echo "  预处理数据目录: ${output_dir}"
                [[ -f "${output_dir}/train_data.h5" ]] && echo "    ✅ 训练数据: ${output_dir}/train_data.h5"
                [[ -f "${output_dir}/valid_data.h5" ]] && echo "    ✅ 验证数据: ${output_dir}/valid_data.h5"
                [[ -f "${output_dir}/test_data.h5" ]] && echo "    ✅ 测试数据: ${output_dir}/test_data.h5"
                [[ -f "${output_dir}/normalization_info.json" ]] && echo "    ✅ 归一化信息: ${output_dir}/normalization_info.json"
            fi
            
            echo ""
            echo -e "${BLUE}📊 检查点文件:${NC}"
            for checkpoint in *.pth; do
                [[ -f "${checkpoint}" ]] && echo "    ✅ ${checkpoint}"
            done
            
            echo ""
            echo -e "${YELLOW}💡 后续操作建议:${NC}"
            echo "  - 查看训练日志了解详细进度"
            echo "  - 使用 nvidia-smi 监控GPU使用情况"
            echo "  - 检查生成的模型检查点文件"
            echo "  - 可以使用 --resume 参数继续训练"
            echo ""
        else
            echo ""
            echo -e "${RED}❌ 训练流程失败，错误代码: ${exit_code}${NC}"
            echo ""
            echo -e "${YELLOW}🔧 故障排除建议:${NC}"
            echo "  1. 检查数据文件路径是否正确"
            echo "  2. 确认Python环境和依赖包已安装"
            echo "  3. 检查磁盘空间是否充足"
            echo "  4. 查看错误信息进行针对性修复"
            echo "  5. 尝试减小批次大小或模型参数"
            echo ""
        fi
        
        exit ${exit_code}
    fi
}

# 信号处理
trap 'echo -e "\n${YELLOW}⚠️ 程序被中断${NC}"; exit 130' INT TERM

# 执行主函数
main "$@"