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
echo "===== 步骤1: 预处理MVBench数据集 ====="
PREPROCESSED_FILE="$ROOT_DIR/data/processed/mvbench_processed.json"

# 创建processed目录（如果不存在）
mkdir -p "$ROOT_DIR/data/processed"

# 运行预处理脚本
python "$ROOT_DIR/data/preprocess.py" \
  --input "$ROOT_DIR/data/MVBench/mvbench_data.json" \
  --output "$PREPROCESSED_FILE"

# 确认预处理完成
if [ ! -f "$PREPROCESSED_FILE" ]; then
  echo "错误: 预处理失败，无法找到输出文件: $PREPROCESSED_FILE"
  exit 1
fi

# 更新配置文件中的数据路径
echo "===== 步骤2: 更新配置文件 ====="
CONFIG_FILE="$ROOT_DIR/configs/mvbench_config.yaml"

# 备份原始配置
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak" 2>/dev/null || echo "创建新配置文件"

# 更新配置文件中的JSON文件路径
sed -i "s|json_file:.*|json_file: \"${PREPROCESSED_FILE}\"|g" "$CONFIG_FILE" 2>/dev/null || \
cat > "$CONFIG_FILE" << EOF
# MVBench配置文件
dataset:
  name: "mvbench"
  json_file: "${PREPROCESSED_FILE}"
  
model:
  name: "your_model_name"  # 替换为你的模型名称
  
test:
  batch_size: 1
  num_workers: 4
EOF

echo "配置文件已更新，路径: $CONFIG_FILE"

# 运行测试
echo "===== 步骤3: 开始测试 ====="
python "$ROOT_DIR/scripts/test_single_agent.py" \
  --config "$CONFIG_FILE" \
  --mode test

echo "===== 测试完成 =====" 