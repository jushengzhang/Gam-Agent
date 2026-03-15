import os
import glob
import argparse
import pandas as pd
import json
from tqdm import tqdm
import sys
import logging
# 尝试导入 vlmeval 相关库
try:
    from vlmeval.smp import load, dump, MMBenchOfficialServer
    from vlmeval.dataset import build_dataset
    from vlmeval.config import supported_VLM # 可能需要用于设置 judge model
    from tabulate import tabulate # 用于美观打印 DataFrame 结果
except ImportError:
    print("错误：无法导入 vlmeval 或其依赖库 (pandas, tqdm, tabulate)。")
    print("请确保已在正确的环境中安装 vlmeval 及其依赖： pip install vlmeval pandas tqdm tabulate")
    sys.exit(1)
    
logger_initialized = {}

def proxy_set(s):
    import os
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s
    
def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    except ImportError:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger
    
def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="聚合 VLM Eval Kit 的 pkl 中间文件并进行评估")
    parser.add_argument('--pkl-dir', type=str, default='./PATH_TO_LOCAL_RESOURCE',
                        help='包含 pkl 中间文件的目录路径 (例如 T20250428_Gac1beb8b)')
    parser.add_argument('--model-name', type=str, default='ExpertAgentModelWrapperEnhancedDebate',
                        help='模型名称 (例如 ExpertAgentModelWrapperEnhancedDebate)')
    parser.add_argument('--dataset-name', type=str, default='MMBench_DEV_EN',
                        help='数据集名称 (例如 MMBench_DEV_EN_V11)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='聚合结果输出文件路径 (.xlsx 或 .tsv)。如果未指定，将自动生成。')
    parser.add_argument('--judge', type=str, default=None,
                        help='指定评估用的 Judge 模型 (例如 chatgpt-0125, gpt-4-turbo)。如果未指定，将尝试根据数据集自动选择。')
    parser.add_argument('--judge-args', type=str, default='{}',
                        help='传递给 Judge 的额外参数 (JSON 格式字符串，例如 {"temperature": 0.1})')
    parser.add_argument('--api-nproc', type=int, default=4, help='评估时并行 API 调用数')
    parser.add_argument('--retry', type=int, default=3, help='评估时 API 调用重试次数')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    # 添加比较文件参数
    parser.add_argument('--compare-file', type=str, default='./PATH_TO_LOCAL_RESOURCE',
                        help='用于比较的结果文件路径')
    args = parser.parse_args()

    # 如果未指定输出文件，自动生成路径和名称
    if args.output_file is None:
        try:
            dataset_for_check = build_dataset(args.dataset_name)
            if dataset_for_check is None:
                 print(f"错误：无法构建数据集 {args.dataset_name} 来确定输出文件类型。请检查数据集名称是否正确。")
                 sys.exit(1)
            ext = '.tsv' if dataset_for_check.TYPE == 'MT' else '.xlsx'
            del dataset_for_check
            args.output_file = os.path.join(args.pkl_dir, f"{args.model_name}_{args.dataset_name}_aggregated{ext}")
        except Exception as e:
            print(f"错误：构建数据集 {args.dataset_name} 时出错: {e}。无法自动确定输出文件类型。")
            print("请使用 --output-file 手动指定输出文件路径。")
            sys.exit(1)

    return args

def compare_results(dataset, result_file1, result_file2, logger):
    """比较两个结果文件在共同子集上的准确率"""
    try:
        # 加载两个结果文件
        df1 = load(result_file1)
        df2 = load(result_file2)
        
        # 确保两个文件都有必要的列
        required_cols = ['index', 'prediction']
        for df, file in [(df1, result_file1), (df2, result_file2)]:
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"文件 {file} 缺少必要的列: {required_cols}")
        
        # 获取共同索引
        indices1 = set(df1['index'].unique())
        indices2 = set(df2['index'].unique())
        common_indices = list(indices1.intersection(indices2))
        
        if not common_indices:
            logger.error("两个结果文件之间没有共同的索引，无法进行比较。")
            return None
            
        print(f"找到 {len(common_indices)} 个共同索引用于比较。")
        
        # 过滤出共同子集
        df1_subset = df1[df1['index'].isin(common_indices)]
        df2_subset = df2[df2['index'].isin(common_indices)]
        
        # 创建临时文件用于评估
        temp_dir = os.path.join(os.path.dirname(result_file1), 'temp_compare')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file1 = os.path.join(temp_dir, 'subset1.xlsx')
        temp_file2 = os.path.join(temp_dir, 'subset2.xlsx')
        
        # 保存子集到临时文件
        dump(df1_subset, temp_file1)
        dump(df2_subset, temp_file2)
        
        # 对两个子集进行评估
        logger.info("评估第一个结果文件...")
        results1 = dataset.evaluate(temp_file1)
        
        logger.info("评估第二个结果文件...")
        results2 = dataset.evaluate(temp_file2)
        
        # 打印比较结果
        logger.info("\n=== 比较结果 ===")
        logger.info(f"文件1 ({os.path.basename(result_file1)}) 在共同子集上的结果:")
        if isinstance(results1, dict):
            logger.info(json.dumps(results1, indent=4, ensure_ascii=False))
        elif isinstance(results1, pd.DataFrame):
            logger.info(f"\n{tabulate(results1, headers='keys', tablefmt='psql')}")
        else:
            logger.info(str(results1))
            
        logger.info(f"\n文件2 ({os.path.basename(result_file2)}) 在共同子集上的结果:")
        if isinstance(results2, dict):
            logger.info(json.dumps(results2, indent=4, ensure_ascii=False))
        elif isinstance(results2, pd.DataFrame):
            logger.info(f"\n{tabulate(results2, headers='keys', tablefmt='psql')}")
        else:
            logger.info(str(results2))
            
        # 清理临时文件
        try:
            os.remove(temp_file1)
            os.remove(temp_file2)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")
            
        return results1, results2
        
    except Exception as e:
        logger.error(f"比较结果时发生错误: {e}", exc_info=True)
        return None

def main():
    args = parse_args()
    logger = get_logger('AggregateEvaluate')

    # 初始化聚合数据结构
    all_data = {}
    logger.info("初始化聚合数据结构为字典。")

    # 1. 查找 pkl 文件
    pkl_pattern = os.path.join(args.pkl_dir, f'[0-9]*_{args.dataset_name}.pkl')
    pkl_files = glob.glob(pkl_pattern)

    # 兼容另一种可能的命名方式
    if not pkl_files:
        pkl_pattern_alt = os.path.join(args.pkl_dir, f'[0-9]*.pkl')
        pkl_files_alt = glob.glob(pkl_pattern_alt)
        pkl_files = [f for f in pkl_files_alt if 'aggregated' not in f.lower() and 'prev' not in f.lower()]

    # 如果在当前目录没找到，尝试在子目录中查找
    if not pkl_files:
        # 查找所有子目录
        subdirs = [d for d in os.listdir(args.pkl_dir) if os.path.isdir(os.path.join(args.pkl_dir, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(args.pkl_dir, subdir)
            # 在子目录中查找pkl文件
            subdir_pattern = os.path.join(subdir_path, f'[0-9]*_{args.dataset_name}.pkl')
            subdir_files = glob.glob(subdir_pattern)
            if subdir_files:
                pkl_files.extend(subdir_files)
                logger.info(f"在子目录 {subdir} 中找到 {len(subdir_files)} 个pkl文件")
                break  # 找到文件后退出循环

        # 如果还是没找到，尝试在子目录中查找任意pkl文件
        if not pkl_files:
            for subdir in subdirs:
                subdir_path = os.path.join(args.pkl_dir, subdir)
                subdir_pattern = os.path.join(subdir_path, f'[0-9]*.pkl')
                subdir_files = glob.glob(subdir_pattern)
                subdir_files = [f for f in subdir_files if 'aggregated' not in f.lower() and 'prev' not in f.lower()]
                if subdir_files:
                    pkl_files.extend(subdir_files)
                    logger.info(f"在子目录 {subdir} 中找到 {len(subdir_files)} 个pkl文件")
                    break  # 找到文件后退出循环

    if not pkl_files:
        logger.error(f"在目录 {args.pkl_dir} 及其子目录中未找到匹配模式的 pkl 文件。")
        logger.error(f"尝试的模式: '{os.path.basename(pkl_pattern)}' 或 '{os.path.basename(pkl_pattern_alt)}' (已过滤)")
        return

    pkl_files.sort()
    logger.info(f"找到 {len(pkl_files)} 个待聚合的 pkl 文件: {', '.join(map(os.path.basename, pkl_files))}")

    # 2. 加载并合并数据
    logger.info("开始加载和合并 pkl 文件...")
    successful_loads = 0
    for pkl_file in tqdm(pkl_files, desc="聚合 pkl 文件"):
        try:
            data = load(pkl_file)
            if successful_loads == 0 and isinstance(data, dict):
                 logger.info(f"第一个 pkl 文件 ({os.path.basename(pkl_file)}) 中字典包含的键: {list(data.keys())}")

            if isinstance(data, dict):
                logger.info(f"文件 {os.path.basename(pkl_file)} 加载成功，包含 {len(data)} 条记录。")
                all_data.update(data)
                successful_loads += 1
            else:
                logger.warning(f"文件 {os.path.basename(pkl_file)} 包含未预期的数据类型 {type(data)}，将被忽略。预期类型为 dict。")

        except Exception as e:
            logger.error(f"加载文件 {os.path.basename(pkl_file)} 失败: {e}", exc_info=True)

    if not all_data:
         logger.error("未能成功加载任何数据，无法继续。")
         return

    logger.info(f"成功合并来自 {successful_loads} 个文件的数据，总记录数: {len(all_data)}")

    # 3. 保存聚合结果
    logger.info(f"将聚合结果保存到: {args.output_file}")

    # 构建数据集对象
    logger.info(f"构建数据集对象: {args.dataset_name}")
    try:
        dataset = build_dataset(args.dataset_name)
        if dataset is None:
             logger.error(f"无法构建数据集 {args.dataset_name}，无法进行后续合并和评估。")
             return
    except Exception as e:
        logger.error(f"构建数据集 {args.dataset_name} 时出错: {e}", exc_info=True)
        return

    try:
        output_ext = os.path.splitext(args.output_file)[1].lower()
        if output_ext in ['.xlsx', '.tsv'] and isinstance(all_data, dict):
            logger.info("检测到数据为字典，尝试转换为 vlmeval 格式的 Pandas DataFrame。")
            try:
                records_for_df = [{'index': k, 'prediction': v} for k, v in all_data.items()]
                if not records_for_df:
                     raise ValueError("转换后的记录列表为空。")
                     
                all_data_df = pd.DataFrame(records_for_df)
                all_data_df['index'] = all_data_df['index'].astype(int)
                all_data_df = all_data_df.sort_values(by='index').reset_index(drop=True)
                
                logger.info(f"预测 DataFrame 创建成功，包含列: {all_data_df.columns.tolist()}，行数: {len(all_data_df)}")

                # 合并预测与原始数据集信息
                logger.info("开始合并预测结果与原始数据集信息...")
                try:
                    original_data_df = dataset.data.sort_values(by='index').reset_index(drop=True)
                    if 'index' not in original_data_df.columns:
                         raise ValueError("原始数据 DataFrame 中缺少 'index' 列，无法合并。")

                    merged_df = pd.merge(original_data_df, all_data_df[['index', 'prediction']], on='index', how='left')

                    logger.info(f"合并完成。最终 DataFrame 包含列: {merged_df.columns.tolist()}，行数: {len(merged_df)}")
                    if merged_df['prediction'].isnull().any():
                        missing_count = merged_df['prediction'].isnull().sum()
                        logger.warning(f"合并后发现 {missing_count} 行缺少预测值 (prediction is NaN)。")

                    dump(merged_df, args.output_file)
                    logger.info(f"合并后的聚合结果已成功保存为 DataFrame 到 {args.output_file}。")

                    # 创建并保存用于评估的文件
                    filtered_df = merged_df.dropna(subset=['prediction'])
                    if not filtered_df.empty:
                        eval_file_path = os.path.splitext(args.output_file)[0] + '_foreval' + output_ext
                        try:
                            dump(filtered_df, eval_file_path)
                            logger.info(f"已将包含 {len(filtered_df)} 条有效预测的数据保存到临时评估文件: {eval_file_path}")
                        except Exception as dump_filtered_e:
                            logger.error(f"保存过滤后的评估文件失败: {dump_filtered_e}", exc_info=True)
                            eval_file_path = args.output_file
                            logger.warning(f"回退到使用完整文件进行评估: {eval_file_path}")
                    else:
                        logger.warning("过滤后没有发现任何有效的预测数据，无法生成用于评估的文件。评估将使用原始合并文件。")
                        eval_file_path = args.output_file

                except Exception as merge_e:
                    logger.error(f"合并预测与原始数据失败: {merge_e}", exc_info=True)
                    logger.warning("将仅保存原始聚合的预测。")
                    dump(all_data_df, args.output_file)
                    logger.info(f"仅包含预测的聚合结果已保存到 {args.output_file}。")
                    eval_file_path = args.output_file

            except Exception as convert_e:
                logger.error(f"将聚合字典转换为 DataFrame 或保存失败: {convert_e}", exc_info=True)
                return
        else:
             logger.warning(f"输出格式为 {output_ext} 或数据不是字典，尝试按原样保存 all_data (类型: {type(all_data)})...")
             dump(all_data, args.output_file)
             logger.info(f"聚合结果已尝试按原格式保存到 {args.output_file}。")
             eval_file_path = args.output_file

    except Exception as e:
        logger.error(f"保存聚合结果时发生意外错误: {e}", exc_info=True)
        return

    # 4. 设置评估参数
    try:
        judge_args_dict = json.loads(args.judge_args)
    except json.JSONDecodeError as e:
        logger.error(f"解析 --judge-args 失败，请确保是有效的 JSON 格式: {e}")
        return

    judge_kwargs = {
        'nproc': args.api_nproc,
        'verbose': args.verbose,
        'retry': args.retry,
        **judge_args_dict,
    }

    # 自动选择 Judge 模型
    if args.judge:
        judge_kwargs['model'] = args.judge
        logger.info(f"使用用户指定的 Judge 模型: {args.judge}")
    else:
        dataset_name_lower = args.dataset_name.lower()
        if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(['moviechat1k'], dataset_name_lower):
             if listinstr(['wemath'], dataset_name_lower):
                 judge_kwargs['model'] = 'gpt-4o-mini'
             else:
                 judge_kwargs['model'] = 'chatgpt-0125'
        elif listinstr(['mmvet', 'llavabench', 'mmbench_video'], dataset_name_lower):
             judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['mathvista', 'mathverse', 'mathvision', 'dynamath', 'vl-rewardbench', 'logicvista', 'moat'], dataset_name_lower):
             judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['mmlongbench', 'mmdu', 'dude', 'slidevqa', 'mia-bench', 'wildvision', 'mmalignbench'], dataset_name_lower):
             judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['vdc'], dataset_name_lower):
            judge_kwargs['model'] = 'llama31-8b'
        elif listinstr(['videommlu_qa', 'videommlu_cap'], dataset_name_lower):
            judge_kwargs['model'] = 'qwen-72b'

        if 'model' in judge_kwargs:
            logger.info(f"根据数据集 {args.dataset_name} 自动选择 Judge 模型: {judge_kwargs['model']}")
        else:
             logger.warning(f"无法为数据集 {args.dataset_name} 自动确定 Judge 模型。")
             logger.warning("评估可能会失败或使用默认 Judge。建议使用 --judge 参数指定。")

    logger.info(f"最终评估参数 (judge_kwargs): {judge_kwargs}")

    # 5. 执行评估
    logger.info(f"开始使用数据集 {args.dataset_name} 评估文件: {eval_file_path}")
    try:
        # 检查特殊数据集的评估限制
        if 'MMBench_TEST' in args.dataset_name and not MMBenchOfficialServer(args.dataset_name):
            logger.error(f"无法在非官方服务器上评估 {args.dataset_name}，跳过评估步骤。")
            logger.info("聚合文件已生成，您可以手动提交到官方服务器进行评估。")
            return
        elif 'MLLMGuard_DS' in args.dataset_name:
            logger.info('MLLMGuard_DS 的评估当前可能不受支持，跳过评估步骤。聚合文件已生成。')
            return
        elif 'AesBench_TEST' == args.dataset_name:
            logger.info(f'聚合结果已保存至 {args.output_file}。请将此文件发送给 AesBench 团队进行评估。跳过本地评估。')
            return
        elif args.dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
            logger.info(f'{args.dataset_name} 是没有公开 Ground Truth 的测试集，无法进行本地评估，跳过评估步骤。聚合文件已生成。')
            return

        # 设置代理
        eval_proxy = os.environ.get('EVAL_PROXY', None)
        old_proxy = os.environ.get('HTTP_PROXY', None)
        if eval_proxy:
            logger.info(f"检测到 EVAL_PROXY 环境变量，设置代理为: {eval_proxy}")
            proxy_set(eval_proxy)

        # 执行评估
        eval_results = dataset.evaluate(eval_file_path, **judge_kwargs)

        # 恢复代理设置
        if eval_proxy:
            logger.info(f"恢复原始代理设置: {old_proxy if old_proxy else '无'}")
            proxy_set(old_proxy if old_proxy else '')

        # 打印评估结果
        logger.info("--- 评估结果 ---")
        if eval_results is not None:
            if isinstance(eval_results, dict):
                formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
                logger.info(f'\n{formatted_results}')
            elif isinstance(eval_results, pd.DataFrame):
                if len(eval_results.columns) > 10 and len(eval_results) == 1:
                    logger.info(f'\n{tabulate(eval_results.T, headers="keys", tablefmt="psql")}')
                else:
                    logger.info(f'\n{tabulate(eval_results, headers="keys", tablefmt="psql", showindex="always")}')
            else:
                logger.info(str(eval_results))
        else:
            logger.info("评估函数没有返回明确的结果。")

    except Exception as e:
        logger.error(f"评估过程中发生严重错误: {e}", exc_info=True)
        if eval_proxy:
            logger.info(f"尝试恢复原始代理设置: {old_proxy if old_proxy else '无'}")
            proxy_set(old_proxy if old_proxy else '')

    # 在评估完成后，添加比较逻辑
    if args.compare_file and os.path.exists(args.compare_file):
        logger.info(f"\n开始比较结果文件: {args.compare_file}")
        compare_results(dataset, eval_file_path, args.compare_file, logger)
    else:
        logger.warning(f"未找到比较文件: {args.compare_file}，跳过比较步骤。")

if __name__ == '__main__':
    main() 