# 使用该脚本中的指令运行MMbench测试
conda activate vlmevalkit
cd .
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export HF_ENDPOINT=https://hf-mirror.com
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export http_proxy=http://127.0.0.1:7890
#export https_proxy=http://127.0.0.1:7890
#export all_proxy=socks5://127.0.0.1:7891
# data 可选项：MMBench_DEV_EN MMBench_TEST_EN MMBench_DEV_CN MMBench_TEST_CN MMBench_TEST_EN_V11 MMBench_DEV_EN_V11
# ExpertAgentModelWrapper为我们的local版，ExpertAgentModelWrapperAPI为api版
# 使用 --sample-num 500 来指定某个集上的抽样\
# 注意同一模型跑同一数据集会调用相同目录存储结果，如果修改了模型内部定义建议用--work-dir指定一个新的
# torchrun --nproc-per-node=4 VLMEvalKit/run.py --data MMBench_TEST_EN_V11 --model ExpertAgentModelWrapper --work-dir ./outputs_new
#python VLMEvalKit/run.py --data MMBench_DEV_EN_V11 --model ExpertAgentModelWrapper --work-dir ./outputs_compare


# 这里快速测试更改后的模型，使用本地模型
# 使用 --sample-num 500 来指定某个集上的抽样\\
# ExpertAgentModelWrapperEnhancedDebate为我们的更新后的local版
# 注意同一模型跑同一数据集会调用相同目录存储结果，如果修改了模型内部定义建议用--work-dir指定一个新的
# python VLMEvalKit/run_new.py --data MVBench_128frame --model ExpertAgentModelWrapperEnhancedDebateQwen --work-dir ./outputs_mvbench/qwen_7B --sample-num -2

# 分布式dds

# 使用 --sample-num -2 来指定某个集上的按分类1/10均匀抽样
torchrun --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29544 VLMEvalKit/run_new.py \
                              --data MMMU_TEST \
                              --model ExpertAgentModelWrapperEnhancedDebateQwen \
                              --work-dir ./outputs_Muir/qwen_7B \
#                              --sample-num -2 \
#                              --reuse \
#                              --sample-split 2 --sample-index 0
#./PATH_TO_LOCAL_RESOURCE 44 240 0,1,2,3
#python VLMEvalKit/run_new.py --data MMBench_TEST_EN_V11 --model ExpertAgentModelWrapperEnhancedDebateQwen --work-dir ./outputs_run/qwen_32B --sample-num -2


