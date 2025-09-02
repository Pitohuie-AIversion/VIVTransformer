@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ================================================================
REM 分离式训练启动脚本 (Windows批处理版本)
REM ================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    分离式训练启动器                          ║
echo ║                                                              ║
echo ║  [INFO] 自动化数据预处理和模型训练流程                           ║
echo ║  [TIP] 解决服务器CPU锁定问题的最佳方案                          ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM 设置默认参数
set "DATA_PATH="
set "CONFIG_FILE=dynamic_config_server_optimized_final.yaml"
set "OUTPUT_DIR=preprocessed_data"
set "NUM_SAMPLES="
set "RESUME_CHECKPOINT="
set "MODE=auto"
set "AUTO_OPTIMIZE=false"

REM 检查命令行参数
if "%1"=="" (
    echo [ERROR] 错误: 请提供数据文件路径
    echo.
    echo 使用方法:
    echo   %0 ^<数据文件路径^> [选项]
    echo.
    echo 选项:
    echo   --config ^<配置文件^>        配置文件路径 (默认: %CONFIG_FILE%)
    echo   --output-dir ^<目录^>       输出目录 (默认: %OUTPUT_DIR%)
    echo   --num-samples ^<数量^>      样本数量
    echo   --resume ^<检查点^>         恢复训练的检查点
    echo   --mode ^<模式^>             运行模式: auto/preprocess/train (默认: auto)
    echo   --auto-optimize           自动优化配置
    echo   --help                    显示帮助信息
    echo.
    echo 示例:
    echo   %0 "C:\data\my_data.h5"
    echo   %0 "C:\data\my_data.h5" --num-samples 1000 --auto-optimize
    echo   %0 "C:\data\my_data.h5" --mode train --resume checkpoint_epoch_50.pth
    echo.
    pause
    exit /b 1
)

set "DATA_PATH=%~1"
shift

REM 解析命令行参数
:parse_args
if "%1"=="" goto :args_done
if "%1"=="--config" (
    set "CONFIG_FILE=%~2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--output-dir" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--num-samples" (
    set "NUM_SAMPLES=%~2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--resume" (
    set "RESUME_CHECKPOINT=%~2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--mode" (
    set "MODE=%~2"
    shift
    shift
    goto :parse_args
)
if "%1"=="--auto-optimize" (
    set "AUTO_OPTIMIZE=true"
    shift
    goto :parse_args
)
if "%1"=="--help" (
    echo 分离式训练启动器帮助
    echo.
    echo 这个脚本会自动执行数据预处理和模型训练的完整流程。
    echo.
    echo 运行模式:
    echo   auto        - 自动模式，先预处理再训练 (默认)
    echo   preprocess  - 仅执行数据预处理
    echo   train       - 仅执行模型训练
    echo.
    echo 功能特点:
    echo   [OK] 自动检测是否需要重新预处理数据
    echo   [OK] 智能资源配置优化
    echo   [OK] 支持断点续训
    echo   [OK] 实时显示训练进度
    echo.
    pause
    exit /b 0
)
shift
goto :parse_args

:args_done

REM 检查文件存在性
if not exist "%DATA_PATH%" (
    echo [ERROR] 错误: 数据文件不存在: %DATA_PATH%
    pause
    exit /b 1
)

if not exist "%CONFIG_FILE%" (
    echo [ERROR] 错误: 配置文件不存在: %CONFIG_FILE%
    pause
    exit /b 1
)

echo [INFO] 运行参数:
echo   数据文件: %DATA_PATH%
echo   配置文件: %CONFIG_FILE%
echo   输出目录: %OUTPUT_DIR%
if not "%NUM_SAMPLES%"=="" echo   样本数量: %NUM_SAMPLES%
if not "%RESUME_CHECKPOINT%"=="" echo   恢复检查点: %RESUME_CHECKPOINT%
echo   运行模式: %MODE%
echo   自动优化: %AUTO_OPTIMIZE%
echo.

REM 构建Python命令
set "PYTHON_CMD=python run_separated_training.py"
set "PYTHON_CMD=%PYTHON_CMD% --data_path "%DATA_PATH%""
set "PYTHON_CMD=%PYTHON_CMD% --config "%CONFIG_FILE%""
set "PYTHON_CMD=%PYTHON_CMD% --output_dir "%OUTPUT_DIR%""
set "PYTHON_CMD=%PYTHON_CMD% --mode %MODE%"

if not "%NUM_SAMPLES%"=="" (
    set "PYTHON_CMD=%PYTHON_CMD% --num_samples %NUM_SAMPLES%"
)

if not "%RESUME_CHECKPOINT%"=="" (
    set "PYTHON_CMD=%PYTHON_CMD% --resume "%RESUME_CHECKPOINT%""
)

if "%AUTO_OPTIMIZE%"=="true" (
    set "PYTHON_CMD=%PYTHON_CMD% --auto_optimize"
)

echo [INFO] 执行命令: %PYTHON_CMD%
echo.

REM 执行Python脚本
%PYTHON_CMD%

REM 检查执行结果
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] 分离式训练流程完成！
    echo.
    echo [INFO] 生成的文件:
    if exist "%OUTPUT_DIR%" (
        echo   预处理数据目录: %OUTPUT_DIR%
        if exist "%OUTPUT_DIR%\train_data.h5" echo     [OK] 训练数据: %OUTPUT_DIR%\train_data.h5
        if exist "%OUTPUT_DIR%\valid_data.h5" echo     [OK] 验证数据: %OUTPUT_DIR%\valid_data.h5
        if exist "%OUTPUT_DIR%\test_data.h5" echo     [OK] 测试数据: %OUTPUT_DIR%\test_data.h5
        if exist "%OUTPUT_DIR%\normalization_info.json" echo     [OK] 归一化信息: %OUTPUT_DIR%\normalization_info.json
    )
    
    echo.
    echo [INFO] 检查点文件:
    for %%f in (*.pth) do (
        echo     [OK] %%f
    )
    
    echo.
    echo [TIP] 后续操作建议:
    echo   - 查看训练日志了解详细进度
    echo   - 使用 nvidia-smi 监控GPU使用情况
    echo   - 检查生成的模型检查点文件
    echo   - 可以使用 --resume 参数继续训练
    echo.
) else (
    echo.
    echo [ERROR] 训练流程失败，错误代码: %ERRORLEVEL%
    echo.
    echo 🔧 故障排除建议:
    echo   1. 检查数据文件路径是否正确
    echo   2. 确认Python环境和依赖包已安装
    echo   3. 检查磁盘空间是否充足
    echo   4. 查看错误信息进行针对性修复
    echo.
)

echo 按任意键退出...
pause >nul
exit /b %ERRORLEVEL%