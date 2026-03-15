import os
import torch
import numpy as np
import json
import logging
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

class AgentTrainer:
    """专家代理训练器"""
    
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
            
        # 保存其他配置
        config_path = os.path.join(checkpoint_dir, f"config_epoch_{epoch}.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model_config": self.model_config,
                "experts_config": self.experts_config,
                "training_config": self.training_config
            }, f, indent=2)
            
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
            'uncertainties': [],
            'task_complexities': [],
            'debate_improvements': []  # 添加辩论改进记录
        }
        
        # 创建此次训练的resultsjson文件 
        results_file = f'results/result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        # 初始化为空列表
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        
        for epoch in range(1, max_epochs + 1):
            self.logger.info(f"===== 第 {epoch}/{max_epochs} 轮 =====")
            
            # 训练一个轮次
            processed_items = 0
            epoch_results = []  # 存储本轮结果
            uncertainties = []  # 存储不确定性评分
            task_complexities = []  # 存储任务复杂度
            
            for batch_idx, batch in enumerate(self.dataloader):
                # 记录当前处理的视频ID
                video_ids = batch['video_id']
                # 这里我选出来了一个子集，只跑这里面的数据，其他跳过
                test_video_ids = ['M_C_168_0000_1012.mp4', 'M_Z_23_8249_8575.mp4', 'M_Z_163_0000_0311.mp4', 'H_T_688_0587_0854.mp4', 'H_H_92_1724_1779.mp4', 'M_Z_162_2177_2294.mp4', 'M_Z_238_3800_3970.mp4', 'H_H_4_1596_1710.mp4', 'M_Z_113_0000_0178.mp4', 'M_C_278_0110_1086.mp4', 'M_Z_79_3735_3952.mp4', 'C_KT_26_1220_1236.mp4', 'H_H_141_2294_2410.mp4', 'M_Z_23_7918_8090.mp4', 'H_A_63_1790_1991.mp4', 'H_H_262_0271_0334.mp4', 'H_A_27_0299_0696.mp4', 'C_KT_28_6420_6435.mp4', 'M_Z_98_0000_0355.mp4', 'H_T_672_0000_0338.mp4', 'H_T_758_0000_0311.mp4', 'C_KT_30_1541_1615.mp4', 'H_T_269_0178_0248.mp4', 'H_T_765_0356_0533.mp4', 'H_T_842_0000_0268.mp4', 'M_C_613_0000_0604.mp4', 'C_KT_30_1541_1615.mp4', 'C_KT_8_10102_10137.mp4', 'H_H_118_1070_1194.mp4', 'H_H_256_0311_0405.mp4', 'M_C_18_0000_0502.mp4', 'H_T_155_0000_0350.mp4', 'M_Z_320_10325_10810.mp4', 'H_H_135_2178_2308.mp4', 'H_A_187_2503_2780.mp4', 'H_A_33_3542_3981.mp4', 'H_A_137_4531_4660.mp4', 'C_KT_18_3129_3146.mp4', 'M_Z_197_0000_0156.mp4', 'H_T_1199_0010_0270.mp4', 'H_T_367_0211_0379.mp4', 'H_H_7_2138_2239.mp4', 'M_Z_12_0000_0285.mp4', 'H_H_105_2061_2140.mp4', 'C_KT_6_1238_1312.mp4', 'H_A_130_0416_0563.mp4', 'C_KT_6_0640_0714.mp4', 'M_Z_303_2327_2388.mp4', 'H_T_716_0000_0285.mp4', 'M_Z_302_1980_2939.mp4', 'M_Z_267_1069_1304.mp4', 'M_Z_21_5750_5910.mp4', 'C_KT_6_0640_0714.mp4', 'H_H_109_2393_2513.mp4', 'C_KT_12_4655_4724.mp4', 'M_Z_213_3050_3280.mp4', 'M_Z_163_0000_0311.mp4', 'H_H_21_1914_2140.mp4', 'H_A_150_0225_0373.mp4', 'C_KT_9_2640_2710.mp4', 'H_H_126_0765_0855.mp4', 'C_KT_6_1238_1312.mp4', 'H_H_128_2381_2523.mp4', 'M_C_113_0196_0365.mp4', 'H_H_198_2002_2102.mp4', 'H_H_160_0500_0667.mp4', 'H_T_1239_0010_0200.mp4', 'H_H_124_1539_1618.mp4', 'H_H_41_2033_2229.mp4', 'H_T_764_0000_0382.mp4', 'M_Z_207_4950_5125.mp4', 'H_T_413_0000_0283.mp4', 'H_T_415_0000_0243.mp4', 'M_Z_10_3680_4000.mp4', 'M_Z_322_2837_2920.mp4', 'M_Z_210_8550_8800.mp4', 'H_H_157_2738_2826.mp4', 'M_Z_36_7911_8366.mp4', 'C_KT_16_2645_2715.mp4', 'C_KT_16_6559_6620.mp4', 'M_Z_237_0850_1200.mp4', 'M_Z_207_8230_8550.mp4', 'H_T_516_0000_0220.mp4', 'M_C_613_0000_0604.mp4', 'H_T_1296_0010_0735.mp4', 'M_Z_239_6264_6409.mp4', 'H_T_672_0000_0338.mp4', 'M_Z_310_0000_0176.mp4', 'H_T_970_0000_0248.mp4', 'M_Z_303_18930_19032.mp4', 'H_A_27_2680_2950.mp4', 'H_A_57_2800_3137.mp4', 'H_A_106_0000_0191.mp4', 'H_H_192_1077_1216.mp4', 'M_Z_32_0000_0681.mp4', 'M_C_681_0184_1028.mp4', 'H_A_134_0571_0703.mp4', 'M_C_2_0421_0669.mp4', 'H_T_147_0000_0296.mp4', 'M_Z_41_0852_1075.mp4', 'C_KT_10_1827_1901.mp4', 'H_H_198_2002_2102.mp4', 'M_C_369_0276_1041.mp4', 'H_A_29_3793_3963.mp4', 'M_C_520_0000_0484.mp4', 'M_Z_33_0483_0814.mp4', 'H_H_39_0510_0629.mp4', 'H_H_258_3727_3845.mp4', 'M_Z_206_12530_13013.mp4', 'M_Z_195_0000_0184.mp4', 'H_H_40_1807_1909.mp4', 'H_H_260_0960_1115.mp4', 'M_Z_252_1487_2538.mp4', 'M_Z_266_5111_5421.mp4', 'H_T_799_0000_0841.mp4', 'M_Z_301_0000_0138.mp4', 'M_Z_266_5819_5948.mp4', 'M_Z_152_0000_0285.mp4', 'H_H_275_2170_2348.mp4', 'H_A_7_3840_4295.mp4', 'H_H_258_3727_3845.mp4', 'H_T_1051_0000_0629.mp4', 'H_H_114_0558_0658.mp4', 'C_KT_21_8513_8528.mp4', 'H_T_426_0000_0335.mp4', 'M_Z_132_0207_0379.mp4', 'H_H_251_2257_2351.mp4', 'H_H_261_2496_2621.mp4', 'M_Z_210_4860_5100.mp4', 'M_Z_258_5046_5170.mp4']
                # test_video_ids = ['H_H_256_0311_0405.mp4', 'M_C_18_0000_0502.mp4', 'H_T_155_0000_0350.mp4', 'M_Z_320_10325_10810.mp4', 'H_H_135_2178_2308.mp4', 'H_A_187_2503_2780.mp4', 'H_A_33_3542_3981.mp4', 'H_A_137_4531_4660.mp4', 'C_KT_18_3129_3146.mp4', 'M_Z_197_0000_0156.mp4', 'H_T_1199_0010_0270.mp4', 'H_T_367_0211_0379.mp4', 'H_H_7_2138_2239.mp4', 'M_Z_12_0000_0285.mp4', 'H_H_105_2061_2140.mp4', 'C_KT_6_1238_1312.mp4', 'H_A_130_0416_0563.mp4', 'C_KT_6_0640_0714.mp4', 'M_Z_303_2327_2388.mp4', 'H_T_716_0000_0285.mp4', 'M_Z_302_1980_2939.mp4', 'M_Z_267_1069_1304.mp4', 'M_Z_21_5750_5910.mp4', 'C_KT_6_0640_0714.mp4', 'H_H_109_2393_2513.mp4', 'C_KT_12_4655_4724.mp4', 'M_Z_213_3050_3280.mp4', 'M_Z_163_0000_0311.mp4', 'H_H_21_1914_2140.mp4', 'H_A_150_0225_0373.mp4', 'C_KT_9_2640_2710.mp4', 'H_H_126_0765_0855.mp4', 'C_KT_6_1238_1312.mp4', 'H_H_128_2381_2523.mp4', 'M_C_113_0196_0365.mp4', 'H_H_198_2002_2102.mp4', 'H_H_160_0500_0667.mp4', 'H_T_1239_0010_0200.mp4', 'H_H_124_1539_1618.mp4', 'H_H_41_2033_2229.mp4', 'H_T_764_0000_0382.mp4', 'M_Z_207_4950_5125.mp4', 'H_T_413_0000_0283.mp4', 'H_T_415_0000_0243.mp4', 'M_Z_10_3680_4000.mp4', 'M_Z_322_2837_2920.mp4', 'M_Z_210_8550_8800.mp4', 'H_H_157_2738_2826.mp4', 'M_Z_36_7911_8366.mp4', 'C_KT_16_2645_2715.mp4', 'C_KT_16_6559_6620.mp4', 'M_Z_237_0850_1200.mp4', 'M_Z_207_8230_8550.mp4', 'H_T_516_0000_0220.mp4', 'M_C_613_0000_0604.mp4', 'H_T_1296_0010_0735.mp4', 'M_Z_239_6264_6409.mp4', 'H_T_672_0000_0338.mp4', 'M_Z_310_0000_0176.mp4', 'H_T_970_0000_0248.mp4', 'M_Z_303_18930_19032.mp4', 'H_A_27_2680_2950.mp4', 'H_A_57_2800_3137.mp4', 'H_A_106_0000_0191.mp4', 'H_H_192_1077_1216.mp4', 'M_Z_32_0000_0681.mp4', 'M_C_681_0184_1028.mp4', 'H_A_134_0571_0703.mp4', 'M_C_2_0421_0669.mp4', 'H_T_147_0000_0296.mp4', 'M_Z_41_0852_1075.mp4', 'C_KT_10_1827_1901.mp4', 'H_H_198_2002_2102.mp4', 'M_C_369_0276_1041.mp4', 'H_A_29_3793_3963.mp4', 'M_C_520_0000_0484.mp4', 'M_Z_33_0483_0814.mp4', 'H_H_39_0510_0629.mp4', 'H_H_258_3727_3845.mp4', 'M_Z_206_12530_13013.mp4', 'M_Z_195_0000_0184.mp4', 'H_H_40_1807_1909.mp4', 'H_H_260_0960_1115.mp4', 'M_Z_252_1487_2538.mp4', 'M_Z_266_5111_5421.mp4', 'H_T_799_0000_0841.mp4', 'M_Z_301_0000_0138.mp4', 'M_Z_266_5819_5948.mp4', 'M_Z_152_0000_0285.mp4', 'H_H_275_2170_2348.mp4', 'H_A_7_3840_4295.mp4', 'H_H_258_3727_3845.mp4', 'H_T_1051_0000_0629.mp4', 'H_H_114_0558_0658.mp4', 'C_KT_21_8513_8528.mp4', 'H_T_426_0000_0335.mp4', 'M_Z_132_0207_0379.mp4', 'H_H_251_2257_2351.mp4', 'H_H_261_2496_2621.mp4', 'M_Z_210_4860_5100.mp4', 'M_Z_258_5046_5170.mp4']
                self.logger.info(f"处理批次 {batch_idx+1}，视频ID: {video_ids}")
                
                for i in range(len(video_ids)):
                    video_id = video_ids[i]
                    instruction = batch['instruction'][i]
                    self.logger.info(f"完成视频 {video_id} 处理，问题: {instruction[:50]}...")
                    
                    video_frame = batch['video_frames'][i]  # 从数据中获取真实的视频帧
                    output = batch['output'][i]
                    options = batch['options'][i]
                    choice_answer = batch['choice_answer'][i]
                    video_or_image_path = batch['video_path'][i]
                    
                    if video_id not in test_video_ids:
                        print(f"跳过视频 {video_id[i]}")
                        continue
                    
                    # 添加日志以便调试
                    if i == 0 and processed_items == 0 and self.dataset_config.get('use_original_video', False) is False:  # 只对第一个样本记录详细信息
                        self.logger.info(f"视频帧类型: {type(video_frame)}")
                        if isinstance(video_frame, torch.Tensor):
                            self.logger.info(f"视频帧形状: {video_frame.shape}, 数据类型: {video_frame.dtype}")
                        elif hasattr(video_frame, 'shape'):
                            self.logger.info(f"视频帧形状: {video_frame.shape}, 数据类型: {getattr(video_frame, 'dtype', 'unknown')}")
                    
                    whether_use_original_video = self.dataset_config.get('use_original_video', False)
                    
                    config = self.model_config
                    self.model_name = config.get("api", {}).get("model_name", "qwen/qwen-2.5-vl-72b-instruct") or config.get("model_name") or os.environ.get("OPENROUTER_MODEL_NAME", "qwen/qwen-2.5-vl-72b-instruct")
                    
                    result = self.model.process_with_experts(video_frame, instruction, output , options , choice_answer, whether_use_original_video , video_or_image_path)
                    
                    # 如果启用辩论功能，执行辩论流程
                    if  use_debate:
                        # 获取视频关键帧 - 注意：这里如果原始视频未处理，可能需要调整
                        # key_frames = self.model.base_model.process_video(video_frame) # 这行可能重复或需要条件判断
                        # 检查是否需要处理视频帧
                        if whether_use_original_video is False:
                            key_frames = self.model.base_model.process_video(video_frame)
                        else:
                            # 如果使用原始视频(路径或base64)，辩论可能需要调整或传递原始标识
                            key_frames = video_frame # 传递原始视频标识符

                        # 记录初始回答
                        initial_response = result['final_response']
                        
                        # 执行辩论流程，传入关键帧或标识符
                        debate_result = self.model.run_debate_process_new( # 假设使用新的辩论函数
                            key_frames,
                            instruction,
                            options, # 传递选项给辩论
                            initial_response,
                            output, # 传递参考答案给辩论（如果需要）
                            whether_use_original_video, # 传递原始视频使用标志
                            video_or_image_path # 传递路径
                        )
                        
                        # 记录辩论改进指标 - 注意：run_debate_process_new可能不直接返回质量，需要评估
                        # 这里假设辩论后的结果直接作为最终结果
                        final_response_after_debate = debate_result # run_debate_process_new直接返回最终回答
                        debate_improvement = 0.0 # 暂时无法计算提升，除非有评估函数
                        # initial_quality = self.model.evaluate_response_quality(initial_response, instruction)
                        # final_quality = self.model.evaluate_response_quality(final_response_after_debate, instruction)
                        # debate_improvement = final_quality - initial_quality

                        # 更新结果
                        result['initial_response'] = initial_response
                        # result['debate_result'] = debate_result # debate_result现在是最终回答
                        result['final_response'] = final_response_after_debate # 更新为辩论后的结果
                        result['debate_improvement'] = debate_improvement
                        
                        self.logger.info(f"辩论完成，质量提升估算：{debate_improvement:.4f}") # 日志可能需要调整
                    else:
                        debate_improvement = 0.0
                        
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
                    
                    # 获取结果中的不确定性和任务复杂度
                    agent_uncertainties = result.get('agent_uncertainties', [])
                    task_complexity = result.get('task_complexity', 0.0)
                    
                    # 记录不确定性和任务复杂度
                    if agent_uncertainties:
                        uncertainties.append(sum(agent_uncertainties) / len(agent_uncertainties))
                    if task_complexity > 0:
                        task_complexities.append(task_complexity)
                    
                    processed_items += 1
                    
                    print(f"已处理 {processed_items} 个样本")
                    
                    # 记录强化学习训练变化
                    if agent_uncertainties:
                        rl_training_history['uncertainties'].append(sum(agent_uncertainties) / len(agent_uncertainties))
                    else:
                        rl_training_history['uncertainties'].append(0.0)
                    rl_training_history['task_complexities'].append(task_complexity)
                    rl_training_history['debate_improvements'].append(result.get('debate_improvement', 0.0))
                    
                    # 保存结果
                    result_entry = {
                        'video_id': video_id,
                        'instruction': instruction,
                        'reference': output,
                        'final_response': result['final_response'],
                        'agent_uncertainties': agent_uncertainties,
                        'task_complexity': task_complexity
                    }
                    
                    # 添加辩论相关结果
                    if use_debate:
                        result_entry.update({
                            'initial_response': result.get('initial_response', ''),
                            'debate_improvement': result.get('debate_improvement', 0.0),
                            'debate_critiques': [critique['critique'] for critique in 
                                                 result.get('debate_result', {}).get('debate_history', [{}])[0].get('critiques', [])]
                                                 if result.get('debate_result', {}).get('debate_history') else []
                        })
                    
                    epoch_results.append(result_entry)
                    
                    #暂缓1秒来缓解api压力
                    time.sleep(1)
                    # 每隔20个样本保存一次检查点和评估结果
                    if processed_items % samples_save_interval == 0:
                        # 保存评估结果
                        results_dir = self.training_config.get('results_dir', 'results')
                        if not os.path.exists(results_dir):
                            os.makedirs(results_dir)
                        
                        # 保存中间结果
                        interim_results_path = os.path.join(results_dir, f"interim_results_epoch_{epoch}_samples_{processed_items}.json")
                        epoch_results_converted = convert_numpy_types(epoch_results)
                        
                        with open(interim_results_path, 'w', encoding='utf-8') as f:
                            json.dump(epoch_results_converted, f, ensure_ascii=False, indent=2)
                        
                        self.logger.info(f"已处理 {processed_items} 个样本")
                        self.logger.info(f"中间结果已保存至 {interim_results_path}")
                        
                        # 保存中间检查点
                        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir)
                        
                        history_path = os.path.join(checkpoint_dir, f"rl_history_epoch_{epoch}_samples_{processed_items}.json")
                        with open(history_path, 'w') as f:
                            json.dump(convert_numpy_types(rl_training_history), f, indent=2)
                        
                        self.logger.info(f"中间检查点(不含MAB)已保存至 {checkpoint_dir}")
            
            self.logger.info(f"第 {epoch} 轮完成")
            
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
                
        self.logger.info("训练（推理模式）完成")
        
    def evaluate(self, test_dataloader=None):
        pass
        # """
        # 评估模型
        
        # Args:
        #     test_dataloader: 测试数据加载器（可选）
        # """
        # if test_dataloader is None:
        #     # 使用训练数据加载器
        #     test_dataloader = self.dataloader
            
        # # 获取是否启用辩论功能的配置
        # use_debate = self.training_config.get('use_debate', False)
        
        # self.logger.info("开始评估")
        # self.logger.info(f"辩论功能：{'已启用' if use_debate else '未启用'}")
        
        # total_reward = 0.0
        # processed_items = 0
        # arm_selections = [0] * self.model.num_arms
        # evaluation_results = []
        # uncertainties = []
        # task_complexities = []
        
        # for batch in test_dataloader:
        #     instructions = batch['instruction']
        #     video_frames = batch['video_frames']  # 从数据加载器获取视频帧
        #     video_ids = batch['video_id']
        #     outputs = batch['output']
            
        #     for i in range(len(instructions)):
        #         instruction = instructions[i]
        #         video_frame = video_frames[i]  # 获取真实视频帧
        #         video_id = video_ids[i]
        #         output = outputs[i]
                
        #         # 添加日志以便调试
        #         if i == 0 and processed_items == 0 and self.dataset_config.get('use_original_video', False) is False:  # 只对第一个样本记录详细信息
        #             self.logger.info(f"评估视频帧类型: {type(video_frame)}")
        #             if isinstance(video_frame, torch.Tensor):
        #                 self.logger.info(f"评估视频帧形状: {video_frame.shape}, 数据类型: {video_frame.dtype}")
        #             elif hasattr(video_frame, 'shape'):
        #                 self.logger.info(f"评估视频帧形状: {video_frame.shape}, 数据类型: {getattr(video_frame, 'dtype', 'unknown')}")
                
        #         # 使用专家处理
        #         try:
        #             result = self.model.process_with_experts(video_frame, instruction)
                    
        #             # 如果启用辩论功能，执行辩论流程
        #             if use_debate:
        #                 # 获取视频关键帧
        #                 key_frames = self.model.base_model.process_video(video_frame)
                        
        #                 # 记录初始回答
        #                 initial_response = result['final_response']
                        
        #                 # 执行辩论流程，传入关键帧
        #                 debate_result = self.model.run_debate_process(
        #                     key_frames,
        #                     instruction, 
        #                     initial_response
        #                 )
                        
        #                 # 记录辩论改进指标
        #                 debate_improvement = debate_result.get('final_quality', 0) - self.model.evaluate_response_quality(initial_response, instruction)
                        
        #                 # 更新结果
        #                 result['initial_response'] = initial_response
        #                 result['debate_result'] = debate_result
        #                 result['final_response'] = debate_result['final_response']
        #                 result['debate_improvement'] = debate_improvement
                        
        #                 self.logger.info(f"评估辩论完成，轮次：{debate_result['rounds_completed']}，质量提升：{debate_improvement:.4f}")
        #         except Exception as e:
        #             self.logger.error(f"评估过程中处理视频帧时出错: {e}")
        #             continue
                
        #         # 计算奖励
        #         reward = self.calculate_reward(
        #             output, 
        #             result['final_response'],
        #             is_choice=result.get('is_choice', False),
        #             choice_number=result.get('choice_number', None)
        #         )
                
        #         # 获取结果中的不确定性和任务复杂度
        #         agent_uncertainties = result.get('agent_uncertainties', [])
        #         task_complexity = result.get('task_complexity', 0.0)
        #         selected_arm = result.get('selected_arm', 0)
                
        #         # 记录不确定性和任务复杂度
        #         if agent_uncertainties:
        #             uncertainties.append(sum(agent_uncertainties) / len(agent_uncertainties))
        #         if task_complexity > 0:
        #             task_complexities.append(task_complexity)
                
        #         # 记录选择的臂
        #         arm_selections[result['selected_arm']] += 1
                
        #         total_reward += reward
        #         processed_items += 1
                
        #         # 保存评估结果
        #         result_entry = {
        #             'video_id': video_id,
        #             'instruction': instruction,
        #             'reference': output,
        #             'final_response': result['final_response'],
        #             'selected_arm': selected_arm,
        #             'selected_expert': result.get('selected_expert_name', ''),
        #             'reward': reward,
        #             'agent_uncertainties': agent_uncertainties,
        #             'task_complexity': task_complexity
        #         }
                
        #         # 添加辩论相关结果
        #         if use_debate:
        #             result_entry.update({
        #                 'initial_response': result.get('initial_response', ''),
        #                 'debate_rounds': result.get('debate_result', {}).get('rounds_completed', 0),
        #                 'debate_improvement': result.get('debate_improvement', 0.0),
        #                 'debate_critiques': [critique['critique'] for critique in 
        #                                      result.get('debate_result', {}).get('debate_history', [{}])[0].get('critiques', [])]
        #                                      if result.get('debate_result', {}).get('debate_history') else []
        #             })
                
        #         evaluation_results.append(result_entry)
        
        # # 计算平均奖励
        # avg_reward = total_reward / max(1, processed_items)
        # self.logger.info(f"评估完成，平均奖励: {avg_reward:.4f}")
        
        # # 打印臂选择统计
        # self.logger.info("臂选择统计:")
        # for i, count in enumerate(arm_selections):
        #     percentage = 100 * count / max(1, processed_items)
        #     self.logger.info(f"臂 {i}: 选择次数 {count} ({percentage:.2f}%)")
            
        # # 记录不确定性和任务复杂度统计
        # if uncertainties:
        #     avg_uncertainty = sum(uncertainties) / len(uncertainties)
        #     self.logger.info(f"平均不确定性: {avg_uncertainty:.4f}")
        # if task_complexities:
        #     avg_complexity = sum(task_complexities) / len(task_complexities)
        #     self.logger.info(f"平均任务复杂度: {avg_complexity:.4f}")
        
        # # 保存评估结果
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.save_results(f"evaluation_{timestamp}", evaluation_results)
            
        # return avg_reward, evaluation_results 