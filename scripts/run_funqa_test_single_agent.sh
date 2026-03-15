#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# 设置PYTHONPATH
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 检测GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $NUM_GPUS 个 GPU 设备"

# 预处理数据
echo "===== 步骤1: 预处理FunQA数据集 ====="
PREPROCESSED_FILE="$ROOT_DIR/data/processed/funqa_train_processed.json"

# 创建processed目录（如果不存在）
mkdir -p "$ROOT_DIR/data/processed"

# 运行预处理脚本
python "$ROOT_DIR/data/preprocess.py" \
  --input "$ROOT_DIR/data/FunQA/train_small.json" \
  --output "$PREPROCESSED_FILE"

# 确认预处理完成
if [ ! -f "$PREPROCESSED_FILE" ]; then
  echo "错误: 预处理失败，无法找到输出文件: $PREPROCESSED_FILE"
  exit 1
fi

# 更新配置文件中的数据路径
echo "===== 步骤2: 更新配置文件 ====="
CONFIG_FILE="$ROOT_DIR/configs/funqa_config_Choice.yaml"

# 备份原始配置
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

# 更新配置文件中的JSON文件路径
sed -i "s|json_file:.*|json_file: \"${PREPROCESSED_FILE}\"|g" "$CONFIG_FILE"

echo "配置文件已更新，路径: $CONFIG_FILE"

# 运行训练
echo "===== 步骤3: 开始训练 ====="
python "$ROOT_DIR/scripts/test_single_agent.py" \
  --config "$CONFIG_FILE" \
  --mode train

# # 评估模型
# echo "===== 步骤4: 评估模型 ====="
# python "$ROOT_DIR/scripts/train.py" \
#   --config "$CONFIG_FILE" \
#   --mode eval

echo "===== 训练和评估完成 ====="