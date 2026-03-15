import os
import torch
import numpy as np
import json
import logging
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from datetime import datetime
from ..models.agent_model import ExpertAgentModel
from ..datasets.video_dataset import create_dataloader
import sys
import time

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)  # 将 numpy.int64 转换为 int
    elif isinstance(obj, np.floating):
        return float(obj)  # 将 numpy.float64 转换为 float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 将数组转换为列表
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

class SingleAgentTest:
    """单个专家测试"""
    
    def __init__(self, config_loader):
        """
        初始化训练器
        
        Args:
            config_loader: 配置加载器
        """
        self.config_loader = config_loader
        self.model_config = config_loader.get_model_config()
        self.experts_config = config_loader.get_experts_config()
        self.dataset_config = config_loader.get_dataset_config()
        self.training_config = config_loader.get_training_config()
        self.metrics_config = config_loader.get_metrics_config()
        self.api_config = config_loader.get_api_config()  # 获取API配置
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型
        self.model = ExpertAgentModel(self.model_config, self.experts_config, self.api_config)
        
        # 初始化数据加载器
        self.dataloader = create_dataloader(self.dataset_config)
        
        # 初始化Rouge计算器
        self.rouge_calculator = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        
    def _setup_logging(self):
        """设置日志"""
        log_dir = self.training_config.get('log_dir', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("AgentTrainer")
        
    def calculate_reward(self, reference, candidate, is_choice=False, choice_number=0):
        """
        计算奖励（基于参考答案和候选答案的相似度）
        
        Args:
            reference: 参考答案
            candidate: 候选答案
            is_choice: 是否为选择题
            choice_number: 选择题编号
            
        Returns:
            float: 奖励值
        """
        reward = 0.0
        reward_breakdown = {}
        
        # 记录输入内容长度
        self.logger.info(f"计算奖励 - 参考答案长度: {len(reference)}, 候选答案长度: {len(candidate)}")
        self.logger.info(f"参考答案(截取): {reference[:100]}...")
        self.logger.info(f"候选答案(截取): {candidate[:100]}...")
        
        # 检查输入是否为空或过短
        if len(reference) < 5 or len(candidate) < 5:
            self.logger.warning(f"参考答案或候选答案过短，无法计算有效奖励")
            return 0.0
        
        # 计算BLEU
        if self.metrics_config.get('use_bleu', True):
            try:
                reference_tokens = reference.split()
                candidate_tokens = candidate.split()
                bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
                reward += bleu_score
                reward_breakdown['bleu'] = bleu_score
                self.logger.info(f"BLEU得分: {bleu_score:.4f}")
            except Exception as e:
                self.logger.warning(f"计算BLEU出错: {e}")
                reward_breakdown['bleu'] = 0.0
        
        # 计算ROUGE
        if self.metrics_config.get('use_rouge', True):
            try:
                rouge_scores = self.rouge_calculator.score(reference, candidate)
                rouge1_score = rouge_scores['rouge1'].fmeasure
                reward += rouge1_score
                reward_breakdown['rouge1'] = rouge1_score
                self.logger.info(f"ROUGE-1得分: {rouge1_score:.4f}")
            except Exception as e:
                self.logger.warning(f"计算ROUGE出错: {e}")
                reward_breakdown['rouge1'] = 0.0
        
        # 使用BERTScore（如果需要，这部分需要设置bert_model）
        if self.metrics_config.get('use_bert_score', False):
            try:
                # 使用模型的get_text_features计算相似度
                ref_features = self.model.get_text_features(reference)
                cand_features = self.model.get_text_features(candidate)
                
                # 计算余弦相似度
                cosine_sim = torch.nn.functional.cosine_similarity(ref_features, cand_features).item()
                reward += cosine_sim
                reward_breakdown['bert_score'] = cosine_sim
                self.logger.info(f"BERT得分: {cosine_sim:.4f}")
            except Exception as e:
                self.logger.warning(f"计算BERT得分出错: {e}")
                reward_breakdown['bert_score'] = 0.0
        
        # 计算使用的指标数量
        active_metrics = sum([
            self.metrics_config.get('use_bleu', True),
            self.metrics_config.get('use_rouge', True),
            self.metrics_config.get('use_bert_score', False)
        ])
        
        # 归一化奖励，使其在[0, 1]范围内
        if active_metrics > 0:
            reward = reward / active_metrics
        
        # 特殊处理：检查选择题的选项匹配
        if is_choice and choice_number is not None:
            # 检查候选答案中是否包含正确选项的标记（如"B"、"B."、"B："等）
            correct_choice_markers = [f"{chr(65+choice_number)}", f"{chr(65+choice_number)}.", f"{chr(65+choice_number)}："]
            for marker in correct_choice_markers:
                if marker in candidate:
                    choice_bonus = 0.5  # 选择题正确奖励
                    reward = max(reward, choice_bonus)  # 取较大值
                    self.logger.info(f"选择题匹配成功，选项 {marker} 奖励: {choice_bonus:.4f}")
                    break
        
        self.logger.info(f"最终奖励: {reward:.4f}, 明细: {reward_breakdown}")
        return reward
    
    def distribute_rewards(self, reward, selected_arm, agent_responses, final_response):
        """
        分配奖励给各个臂
        
        Args:
            reward: 整体奖励
            selected_arm: 选择的臂
            agent_responses: 各专家回答
            final_response: 最终回答
        """
        # 更新选择的臂的奖励
        if selected_arm < len(self.model.arms):
            self.model.arms[selected_arm]['total_reward'] += reward
            self.model.arms[selected_arm]['count'] += 1
            
        # 如果是组合专家臂，还需要更新各个单独专家的贡献
        if selected_arm >= 3:  # 组合专家
            # 简单实现：根据各专家回答与最终回答的相似度分配奖励
            for i, agent_response in enumerate(agent_responses):
                if i < 3:  # 只考虑前三个单独专家
                    # 计算回答相似度
                    ref_features = self.model.get_text_features(final_response)
                    agent_features = self.model.get_text_features(agent_response['response'])
                    similarity = torch.nn.functional.cosine_similarity(ref_features, agent_features).item()
                    
                    # 根据相似度分配奖励
                    partial_reward = reward * similarity
                    
                    # 更新专家臂
                    self.model.arms[i]['total_reward'] += partial_reward
                    self.model.arms[i]['count'] += 1
    
    def save_results(self, epoch, results):
        """
        保存评估结果到文件
        
        Args:
            epoch: 当前轮次
            results: 评估结果列表
        """
        results_dir = self.training_config.get('results_dir', 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # 保存结果
        results_path = os.path.join(results_dir, f"evaluation_results_epoch_{epoch}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results), f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"评估结果已保存至 {results_path}")
    
    def save_checkpoint(self, epoch, rl_history=None):
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            rl_history: 强化学习训练历史记录（可选）
        """
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # 保存MAB状态
        mab_path = os.path.join(checkpoint_dir, f"mab_state_epoch_{epoch}.json")
        with open(mab_path, 'w') as f:
            json.dump(convert_numpy_types(self.model.arms), f, indent=2)
            
        # 保存其他配置
        config_path = os.path.join(checkpoint_dir, f"config_epoch_{epoch}.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model_config": self.model_config,
                "experts_config": self.experts_config,
                "training_config": self.training_config
            }, f, indent=2)
            
        # 保存强化学习训练历史记录
        if rl_history:
            history_path = os.path.join(checkpoint_dir, f"rl_history_epoch_{epoch}.json")
            with open(history_path, 'w') as f:
                json.dump(convert_numpy_types(rl_history), f, indent=2)
            
        self.logger.info(f"检查点已保存至 {checkpoint_dir}")
        
    def train(self):
        """训练模型"""
        max_epochs = self.training_config.get('max_epochs', 10)
        save_interval = self.training_config.get('save_interval', 1)
        samples_save_interval = 2  # 每20个样本保存一次
        
        # 获取是否启用辩论功能的配置
        use_debate = self.training_config.get('use_debate', False)
        
        self.logger.info(f"开始训练，共{max_epochs}轮")
        self.logger.info(f"辩论功能：{'已启用' if use_debate else '未启用'}")
        
        # 添加记录强化学习训练变化的记录器
        rl_training_history = {
            'arm_weights': [],
            'arm_selections': [],
            'rewards': [],
            'uncertainties': [],
            'task_complexities': [],
            'debate_improvements': []  # 添加辩论改进记录
        }
        
        # 创建此次训练的resultsjson文件 
        results_file = f'results/Single_Agent_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        # 初始化为空列表
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        
        for epoch in range(1, max_epochs + 1):
            self.logger.info(f"===== 第 {epoch}/{max_epochs} 轮 =====")
            
            # 训练一个轮次
            total_reward = 0.0
            processed_items = 0
            epoch_results = []  # 存储本轮结果
            arm_selections = [0] * self.model.num_arms  # 记录每个臂的选择次数
            uncertainties = []  # 存储不确定性评分
            task_complexities = []  # 存储任务复杂度
            
            for batch_idx, batch in enumerate(self.dataloader):
                # 记录当前处理的视频ID
                video_ids = batch['video_id']
                self.logger.info(f"处理批次 {batch_idx+1}，视频ID: {video_ids}")
                
                for i in range(len(video_ids)):
                    video_id = video_ids[i]
                    instruction = batch['instruction'][i]
                    self.logger.info(f"完成视频 {video_id} 处理，问题: {instruction[:50]}...")
                    
                    video_frame = batch['video_frames'][i]  # 从数据中获取真实的视频帧
                    output = batch['output'][i]
                    options = batch['options'][i]
                    choice_answer = batch['choice_answer'][i]

                    print(f"options: {options}")
                    
                    # 添加日志以便调试
                    if i == 0 and processed_items == 0 and self.dataset_config.get('use_original_video', False) is False:  # 只对第一个样本记录详细信息
                        self.logger.info(f"视频帧类型: {type(video_frame)}")
                        if isinstance(video_frame, torch.Tensor):
                            self.logger.info(f"视频帧形状: {video_frame.shape}, 数据类型: {video_frame.dtype}")
                        elif hasattr(video_frame, 'shape'):
                            self.logger.info(f"视频帧形状: {video_frame.shape}, 数据类型: {getattr(video_frame, 'dtype', 'unknown')}")
                    
                    whether_use_original_video = self.dataset_config.get('use_original_video', False)
                    result = self.model.process_with_single_agent(video_frame, instruction, output , options , choice_answer, whether_use_original_video)
                    
                    # # 如果启用辩论功能，执行辩论流程
                    # if  use_debate:
                    #     # 获取视频关键帧
                    #     key_frames = self.model.base_model.process_video(video_frame)
                        
                    #     # 记录初始回答
                    #     initial_response = result['final_response']
                        
                    #     # 执行辩论流程，传入关键帧
                    #     debate_result = self.model.run_debate_process(
                    #         key_frames,
                    #         instruction, 
                    #         initial_response
                    #     )
                        
                    #     # 记录辩论改进指标
                    #     debate_improvement = debate_result.get('final_quality', 0) - self.model.evaluate_response_quality(initial_response, instruction)
                        
                    #     # 更新结果
                    #     result['initial_response'] = initial_response
                    #     result['debate_result'] = debate_result
                    #     result['final_response'] = debate_result['final_response']
                    #     result['debate_improvement'] = debate_improvement
                        
                        
                    #     self.logger.info(f"辩论完成，轮次：{debate_result['rounds_completed']}，质量提升：{debate_improvement:.4f}")
                    # else:
                    #     debate_improvement = 0.0
                        
                    #添加output和videoid到result中
                    result['output'] = output
                    result['video_id'] = video_id
                        
                    # 立即保存单个结果到文件
                    # 读取当前的JSON文件
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            existing_results = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        # 如果文件不存在或JSON解析错误，就创建一个新的空列表
                        existing_results = []
                    
                    # 添加新结果
                    existing_results.append(convert_numpy_types(result))
                    
                    # 写回文件（使用临时文件确保原子性写入）
                    temp_file = f"{results_file}.temp"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_results, f, ensure_ascii=False, indent=2)
                    
                    # 重命名临时文件为目标文件
                    os.replace(temp_file, results_file)
                    
                    self.logger.info(f"已保存样本 #{len(existing_results)} 的结果")
                    
                    # # 获取结果中的不确定性和任务复杂度
                    # agent_uncertainties = result.get('agent_uncertainties', [])
                    # task_complexity = result.get('task_complexity', 0.0)
                    # selected_arm = result.get('selected_arm', 0)
                    
                    # # 记录不确定性和任务复杂度
                    # if agent_uncertainties:
                    #     uncertainties.append(sum(agent_uncertainties) / len(agent_uncertainties))
                    # if task_complexity > 0:
                    #     task_complexities.append(task_complexity)
                    
                    # # 更新臂选择计数
                    # arm_selections[selected_arm] += 1
                    
                    # # 计算奖励
                    # reward = self.calculate_reward(
                    #     output, 
                    #     result['final_response'],
                    #     is_choice=result.get('is_choice', False),
                    #     choice_number=result.get('choice_number', None)
                    # )
                    
                    # # 分配奖励
                    # self.distribute_rewards(
                    #     reward, 
                    #     result['selected_arm'], 
                    #     result['agent_responses'], 
                    #     result['final_response']
                    # )
                    
                    # total_reward += reward
                    # processed_items += 1
                    
                    # print(f"已处理 {processed_items} 个样本，平均奖励: {total_reward/processed_items:.4f}")
                    
                    # # 记录强化学习训练变化
                    # rl_training_history['arm_weights'].append([arm['total_reward']/max(1, arm['count']) for arm in self.model.arms])
                    # rl_training_history['arm_selections'].append(selected_arm)
                    # rl_training_history['rewards'].append(reward)
                    # if agent_uncertainties:
                    #     rl_training_history['uncertainties'].append(sum(agent_uncertainties) / len(agent_uncertainties))
                    # else:
                    #     rl_training_history['uncertainties'].append(0.0)
                    # rl_training_history['task_complexities'].append(task_complexity)
                    # rl_training_history['debate_improvements'].append(result.get('debate_improvement', 0.0))
                    
                    # # 保存结果
                    # result_entry = {
                    #     'video_id': video_id,
                    #     'instruction': instruction,
                    #     'reference': output,
                    #     'final_response': result['final_response'],
                    # }
                    
                    # # # # 添加辩论相关结果
                    # # # if use_debate:
                    # # #     result_entry.update({
                    # # #         'initial_response': result.get('initial_response', ''),
                    # # #         'debate_rounds': result.get('debate_result', {}).get('rounds_completed', 0),
                    # # #         'debate_improvement': result.get('debate_improvement', 0.0),
                    # # #         'debate_critiques': [critique['critique'] for critique in 
                    # # #                              result.get('debate_result', {}).get('debate_history', [{}])[0].get('critiques', [])]
                    # # #                              if result.get('debate_result', {}).get('debate_history') else []
                    # # #     })
                    
                    # epoch_results.append(result_entry)
                    
                    #暂缓1秒来缓解api压力
                    time.sleep(1)
                    # # 每隔20个样本保存一次检查点和评估结果
                    # if processed_items % samples_save_interval == 0:
                    #     # 保存评估结果
                    #     results_dir = self.training_config.get('results_dir', 'results')
                    #     if not os.path.exists(results_dir):
                    #         os.makedirs(results_dir)
                        
                    #     # 保存中间结果
                    #     interim_results_path = os.path.join(results_dir, f"interim_results_epoch_{epoch}_samples_{processed_items}.json")
                    #     epoch_results_converted = convert_numpy_types(epoch_results)
                        
                    #     with open(interim_results_path, 'w', encoding='utf-8') as f:
                    #         json.dump(epoch_results_converted, f, ensure_ascii=False, indent=2)
                        
                    #     self.logger.info(f"已处理 {processed_items} 个样本，平均奖励: {total_reward/processed_items:.4f}")
                    #     self.logger.info(f"中间结果已保存至 {interim_results_path}")
                        
                    #     self.logger.info("当前MAB权重状态:")
                    #     for arm_idx, arm in enumerate(self.model.arms):
                    #         avg_reward = arm['total_reward'] / max(1, arm['count'])
                    #         self.logger.info(f"臂 {arm_idx}: 平均奖励 = {avg_reward:.4f}, 选择次数 = {arm['count']}")
                        
                    #     # 保存中间检查点
                    #     checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
                    #     if not os.path.exists(checkpoint_dir):
                    #         os.makedirs(checkpoint_dir)
                        
                    #     mab_path = os.path.join(checkpoint_dir, f"mab_state_epoch_{epoch}_samples_{processed_items}.json")
                    #     with open(mab_path, 'w') as f:
                    #         json.dump(convert_numpy_types(self.model.arms), f, indent=2)
                        
                    #     # 保存强化学习训练变化记录
                    #     history_path = os.path.join(checkpoint_dir, f"rl_history_epoch_{epoch}_samples_{processed_items}.json")
                    #     with open(history_path, 'w') as f:
                    #         json.dump(convert_numpy_types(rl_training_history), f, indent=2)
                        
                    #     self.logger.info(f"中间检查点已保存至 {mab_path}")
            
            # 计算平均奖励
            avg_reward = total_reward / max(1, processed_items)
            self.logger.info(f"第 {epoch} 轮完成，平均奖励: {avg_reward:.4f}")
            
            # 记录臂选择统计
            self.logger.info("臂选择统计:")
            for i, count in enumerate(arm_selections):
                percentage = 100 * count / max(1, processed_items)
                self.logger.info(f"臂 {i}: 选择次数 {count} ({percentage:.2f}%)")
            
            # 记录不确定性和任务复杂度统计
            if uncertainties:
                avg_uncertainty = sum(uncertainties) / len(uncertainties)
                self.logger.info(f"平均不确定性: {avg_uncertainty:.4f}")
            if task_complexities:
                avg_complexity = sum(task_complexities) / len(task_complexities)
                self.logger.info(f"平均任务复杂度: {avg_complexity:.4f}")
            
            # 保存评估结果
            self.save_results(epoch, epoch_results)
            
            # 保存检查点
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, rl_history=rl_training_history)
                
        self.logger.info("训练完成")
        
    def evaluate(self, test_dataloader=None):
        pass
        #这里train逻辑复用过来加载视频用于单个视频测试就行了，不需要evaluate