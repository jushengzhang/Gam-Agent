#!/bin/bash

# 基础配置模板路径
base_config="configs/ExpertConfig_local_images_enhanced_debate.yaml"
model_name="ExpertAgentModelWrapperEnhancedDebateQwen"
dataset="MMBench_TEST_EN_V11"
output_base="./outputs_run/qwen_7B_dr"

# 遍历 debate_rounds = 2 到 6
for rounds in {2..6}
do
    # 创建临时配置文件
    temp_config="configs/temp_debate_rounds_${rounds}.yaml"
    
    # 替换 debate_rounds 的值
    sed "s/debate_rounds: [0-9]\+/debate_rounds: ${rounds}/" "$base_config" > "$temp_config"
    
    echo "Running with debate_rounds=$rounds using config: $temp_config"
    
    # 运行测试命令
    python VLMEvalKit/run_new.py \
      --work-dir "${output_base}${rounds}" \
      --config "$temp_config" \
      --sample-num -2
done