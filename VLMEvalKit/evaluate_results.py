'''
独立评估脚本，用于对指定的预测结果文件进行打分。
'''
import argparse
import json
import os
import sys

import pandas as pd

# 将 VLMEvalKit 的根目录添加到 Python 路径中
# 假设此脚本位于 VLMEvalKit 目录下
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_root = current_dir # 或者根据实际情况调整
if kit_root not in sys.path:
    sys.path.append(kit_root)

from vlmeval.dataset import build_dataset
from vlmeval.smp import load_env, get_logger, listinstr, proxy_set, tabulate
from vlmeval.utils.result_transfer import MMBenchOfficialServer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate predictions for a given dataset.')
    parser.add_argument('prediction_file', type=str, help='Path to the prediction Excel file.')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Name of the dataset (e.g., MMBench_DEV_EN_V11). Will try to infer from filename if not provided.')
    parser.add_argument('--judge', type=str, default=None, help='Specify the judge model.')
    parser.add_argument('--retry', type=int, default=3, help='Retry numbers for API calls.')
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    args = parser.parse_args()
    return args

def infer_dataset_name(filename):
    # 尝试从文件名推断数据集名称
    # 例如：ExpertAgentModelWrapperEnhancedDebate_MMBench_DEV_EN_V11_aggregated.xlsx -> MMBench_DEV_EN_V11
    basename = os.path.basename(filename)
    parts = basename.split('_')
    # 查找可能的 dataset 名称部分
    # 这是一个简单的启发式方法，可能需要根据实际文件名格式调整
    potential_datasets = [
        'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11',
        'MME', 'SEEDBench_IMG', 'CCBench', 'ScienceQA_IMG', 'AI2D_TEST', # ... 添加更多已知数据集
        'MMBench_Video', 'Video-MME' # ... 视频数据集
    ]
    for name in potential_datasets:
        if name in basename:
            return name
    return None

def main():
    load_env()
    logger = get_logger('EVAL')
    args = parse_args()

    # 获取数据集名称
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = infer_dataset_name(args.prediction_file)
        if dataset_name is None:
            logger.error(f"Could not infer dataset name from filename: {args.prediction_file}. Please specify using --dataset-name.")
            sys.exit(1)
        else:
            logger.info(f"Inferred dataset name: {dataset_name}")
    else:
        logger.info(f"Using specified dataset name: {dataset_name}")


    # 加载数据集对象
    try:
        # MMBench 需要特殊处理官方服务器检查
        if listinstr(['MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN', 'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'], dataset_name) \
           and not MMBenchOfficialServer(dataset_name):
            logger.error(
                f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
            sys.exit(1)
        dataset = build_dataset(dataset_name)
        if dataset is None:
             logger.error(f'Failed to build dataset: {dataset_name}')
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        sys.exit(1)

    # 准备 judge 参数 (参考 run.py)
    judge_kwargs = {
        'nproc': args.api_nproc,
        'verbose': args.verbose,
        'retry': args.retry,
        **(json.loads(args.judge_args) if args.judge_args else {}),
    }

    if args.judge is not None:
        judge_kwargs['model'] = args.judge
    else:
        # 使用 run.py 中的默认 judge 逻辑
        if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr([
                'moviechat1k'
            ], dataset_name.lower()):
            if listinstr(['WeMath'], dataset_name):
                judge_kwargs['model'] = 'gpt-4o-mini'
            else:
                judge_kwargs['model'] = 'chatgpt-0125' # MMBench 默认
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
            judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['VDC'], dataset_name):
            judge_kwargs['model'] = 'llama31-8b'
        elif listinstr(['VideoMMLU_QA', 'VideoMMLU_CAP'], dataset_name):
            judge_kwargs['model'] = 'qwen-72b'
        else:
            logger.warning(f"No default judge model specified for dataset type {dataset.TYPE} and name {dataset_name}. Evaluation might fail if judge is required.")

    logger.info(f"Using Judge Arguments: {judge_kwargs}")

    # 检查预测文件是否存在
    if not os.path.exists(args.prediction_file):
        logger.error(f"Prediction file not found: {args.prediction_file}")
        sys.exit(1)

    # 设置代理 (参考 run.py)
    eval_proxy = os.environ.get('EVAL_PROXY', None)
    old_proxy = os.environ.get('HTTP_PROXY', '')
    if eval_proxy is not None:
        proxy_set(eval_proxy)

    # 执行评估
    try:
        eval_results = dataset.evaluate(args.prediction_file, **judge_kwargs)
    except Exception as e:
        logger.error(f"Evaluation failed for {args.prediction_file} on dataset {dataset_name}:")
        logger.exception(e)
        sys.exit(1)
    finally:
        # 恢复代理
        if eval_proxy is not None:
            proxy_set(old_proxy)

    # 显示结果
    if eval_results is not None:
        logger.info(f'Evaluation finished for: {args.prediction_file}')
        logger.info('Evaluation Results:')
        if isinstance(eval_results, dict):
            logger.info('\n' + json.dumps(eval_results, indent=4))
        elif isinstance(eval_results, pd.DataFrame):
            if len(eval_results) < len(eval_results.columns):
                eval_results = eval_results.T
            logger.info('\n' + tabulate(eval_results))
        else:
            logger.info(eval_results)
    else:
        logger.warning("Evaluation function did not return any results.")

if __name__ == '__main__':
    main() 