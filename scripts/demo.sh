#!/bin/bash
# 多专家视频理解演示脚本

# 设置工作目录为项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

# 导出PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 查找可用的Python命令
for cmd in python3 python python3.8 python3.9 python3.10; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        echo "找到Python命令: $PYTHON_CMD"
        $PYTHON_CMD --version
        break
    fi
done

# 如果没有找到Python命令，则退出
if [ -z "$PYTHON_CMD" ]; then
    echo "错误: 未找到可用的Python命令，请确保Python已安装并在PATH中"
    echo "您可以通过设置环境变量手动指定Python路径: export PYTHON_CMD=/path/to/python"
    exit 1
fi

# 获取命令行参数
VIDEO_PATH="$1"
QUESTION="$2"

# 检查参数是否齐全
if [ -z "$VIDEO_PATH" ] || [ -z "$QUESTION" ]; then
    echo "用法: $0 <视频路径> <问题>"
    echo "示例: $0 data/videos/example.mp4 \"视频中人物在做什么？\""
    exit 1
fi

# 检查视频文件是否存在
if [ ! -f "$VIDEO_PATH" ]; then
    echo "错误: 视频文件不存在: $VIDEO_PATH"
    exit 1
fi

# 检查是否为选择题格式
IS_CHOICE=false
if [[ "$QUESTION" == *"选项"* ]] || [[ "$QUESTION" == *"Option"* ]]; then
    IS_CHOICE=true
    echo "检测到选择题格式"
fi

# 设置配置文件路径
CONFIG_PATH="configs/funqa_config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 创建输出和日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/$TIMESTAMP"
LOG_DIR="logs/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 设置输出文件名
VIDEO_BASENAME=$(basename "$VIDEO_PATH")
OUTPUT_FILE="$OUTPUT_DIR/${VIDEO_BASENAME%.*}_result.json"
LOG_FILE="$LOG_DIR/${VIDEO_BASENAME%.*}_log.txt"

echo "=========================================="
echo "多专家视频理解演示"
echo "=========================================="
echo "视频路径: $VIDEO_PATH"
echo "问题: $QUESTION"
echo "结果将保存至: $OUTPUT_FILE"
echo "日志将保存至: $LOG_FILE"
echo "=========================================="

# 打印处理信息
echo "开始处理视频问答..."

# 运行预测脚本
$PYTHON_CMD scripts/predict_video.py \
    --video "$VIDEO_PATH" \
    --question "$QUESTION" \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_FILE" \
    --num_frames 8 2>&1 | tee "$LOG_FILE"

# 检查结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "演示完成!"
    echo "结果已保存至: $OUTPUT_FILE"
    echo "日志已保存至: $LOG_FILE"
    echo "=========================================="
    
    # 如果是选择题，提取选择编号并提示用户
    if [ "$IS_CHOICE" = true ]; then
        if [ -f "$OUTPUT_FILE" ]; then
            # 使用jq提取选择编号（如果安装了jq）
            if command -v jq &> /dev/null; then
                CHOICE_NUMBER=$(jq -r '.choice_number // ""' "$OUTPUT_FILE")
                if [ ! -z "$CHOICE_NUMBER" ]; then
                    echo "最终选择: 选项 $CHOICE_NUMBER"
                fi
            else
                echo "提示: 如果安装jq工具，可以更方便地解析JSON结果"
                echo "输出已保存到: $OUTPUT_FILE"
            fi
        fi
    fi
else
    echo "=========================================="
    echo "处理过程中发生错误，查看日志了解详情: $LOG_FILE"
    echo "=========================================="
    exit 1
fi 