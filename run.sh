# 使用该脚本中的指令运行MMbench测试
conda activate vlmevalkit
cd .
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891
# data 可选项：MMBench_DEV_EN MMBench_TEST_EN MMBench_DEV_CN MMBench_TEST_CN MMBench_TEST_EN_V11 MMBench_DEV_EN_V11
# ExpertAgentModelWrapper为我们的local版，ExpertAgentModelWrapperAPI为api版
# 使用 --sample-num 500 来指定某个集上的抽样\
# 注意同一模型跑同一数据集会调用相同目录存储结果，如果修改了模型内部定义建议用--work-dir指定一个新的
torchrun --nproc-per-node=4 VLMEvalKit/run.py --data MMBench_TEST_EN_V11 --model ExpertAgentModelWrapper --work-dir ./outputs_new --reuse
#python VLMEvalKit/run.py --data MMBench_DEV_EN_V11 --model ExpertAgentModelWrapper --work-dir ./outputs_compare


# 这里快速测试更改后的模型，使用本地模型
# 使用 --sample-num 500 来指定某个集上的抽样\\
# ExpertAgentModelWrapperEnhancedDebate为我们的更新后的local版
# 注意同一模型跑同一数据集会调用相同目录存储结果，如果修改了模型内部定义建议用--work-dir指定一个新的
# python VLMEvalKit/run.py --data MMBench_DEV_EN_V11 --model ExpertAgentModelWrapperEnhancedDebate --work-dir ./outputs_enhanced_debate



