import os
import sys
import torch
import numpy as np
from PIL import Image
import yaml
import logging
# import cv2 # No longer needed here

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Make sure the class name matches the potentially renamed file
# If the file is still named _未完成.py, use that name.
# If you renamed it, change the import accordingly.
from src.models.agent_model_local_video import ExpertAgentModel 

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    # 指向新的配置文件
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'configs', 'multi_agent_local_video.yaml') # <--- 修改这里

    if not os.path.exists(config_path):
        # 可以保留回退逻辑，或者直接报错
        logging.error(f"配置文件未找到: {config_path}")
        # return {} # 或者引发错误
        # Fallback for safety?
        logging.warning(f"回退到 MMbench_config_Choice.yaml")
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'configs', 'MMbench_config_Choice.yaml')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        # 记录加载的实际文件路径，方便调试
        config['config_file_path'] = config_path
    return config

# Removed the old process_video function which used OpenCV
# def process_video(...):
#    ...

def main():
    # 设置日志
    logger = setup_logging()
    logger.info("开始多智能体视频处理演示 (使用 InternVideo2.5)")
    
    # 加载配置 (Ensure config has model_path, num_segments etc. for the new model)
    config = load_config()
    logger.info(f"配置加载完成 from {config.get('config_file_path', '默认路径')}") # Add path logging if available
    
    # ----- Configuration Check & Defaults ----- 
    # Ensure the loaded config contains necessary keys for the new AgentModel
    # Provide defaults if they are missing in the YAML
    model_config_dict = config.get('local_model', {}) # Assuming model settings are under 'local_model' key
    if 'model_path' not in model_config_dict:
        model_config_dict['model_path'] = 'OpenGVLab/InternVideo2_5_Chat_8B'
        logger.warning(f"配置文件中未找到 'model_path'，使用默认值: {model_config_dict['model_path']}")
    if 'num_segments' not in model_config_dict:
        model_config_dict['num_segments'] = 32
        logger.warning(f"配置文件中未找到 'num_segments'，使用默认值: {model_config_dict['num_segments']}")
    if 'input_size' not in model_config_dict:
        model_config_dict['input_size'] = 448
        logger.warning(f"配置文件中未找到 'input_size'，使用默认值: {model_config_dict['input_size']}")
    if 'max_num_patches_per_frame' not in model_config_dict:
        model_config_dict['max_num_patches_per_frame'] = 1
        logger.warning(f"配置文件中未找到 'max_num_patches_per_frame'，使用默认值: {model_config_dict['max_num_patches_per_frame']}")
    # Update the config dictionary with defaults if needed
    config['local_model'] = model_config_dict
    # ----- End Configuration Check -----

    # ----- Initialize Model ----- 
    logger.info("正在初始化专家代理模型...")
    expert_agent = ExpertAgentModel(
        model_config=config.get('local_model', {}), # Pass the model sub-dictionary
        experts_config=config.get('experts', {}),
    )
    logger.info("专家代理模型初始化完成")
    # ----- End Initialize Model -----

    # ----- Set Logger Level to WARNING to reduce verbosity from model ----- 
    logging.getLogger().setLevel(logging.WARNING) # Set root logger level
    logger.warning("日志级别已设置为 WARNING，将只显示重要消息和错误。") # Use warning to ensure this message is shown
    # ----- End Set Logger Level -----

    # ----- Get Video Path ----- 
    # Use a default path, but make it easily changeable
    default_video_path = "./PATH_TO_LOCAL_RESOURCE" 
    video_path = config.get("demo_video_path", default_video_path) # Allow overriding via config
    if not os.path.exists(video_path):
        logger.error(f"视频文件未找到: {video_path}")
        logger.error("请确保配置文件中的 'demo_video_path' 或默认路径指向一个有效的视频文件。")
        return # Exit if video is not found
    logger.info(f"将要处理的视频文件: {video_path}")
    # ----- End Get Video Path -----
    
    # 测试问题 (更新为英文问题和选项)
    instruction = "Where was this video filmed?"
    options = [
        "A. United Kingdom", 
        "B. China", 
        "C. Japan", 
        "D. Vietnam", 
        "E. United States"
    ]

    # 对这个特定问题进行测试
    logger.info(f"\n===== 测试问题: {instruction} =====")
    logger.info(f"选项: {', '.join(options)}")
        
    try:
        # 使用专家代理模型处理 (传入问题和选项)
        result = expert_agent.process_with_experts(
            video_path=video_path, 
            instruction=instruction,
            options=options # 传入选项列表
        )

        # 打印结果 (Check if keys in result dict have changed)
        if "error" in result:
            logger.error(f"处理时发生错误: {result['error']}")
        else:
            # Ensure logger level is INFO temporarily to print results clearly
            original_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
            
            logger.info("--- 专家回答 --- ") # Clearer header
            if result.get("agent_responses"):
                 for expert_resp in result["agent_responses"]:
                     # Limit long outputs for clarity
                     response_text = expert_resp.get('response', '[无回应]')
                     logger.info(f" - {expert_resp.get('role', '未知专家')}: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            else:
                logger.warning("结果中未找到 'agent_responses'。")
            
            # Log uncertainties
            if result.get("agent_uncertainties"):
                uncertainties_str = ", ".join([f"{u:.3f}" for u in result["agent_uncertainties"]])
                logger.info(f"专家不确定性: [{uncertainties_str}]")

            # Log initial and final responses
            initial_resp = result.get('initial_response', '[无初始回应]')
            final_resp = result.get('final_response', '[无最终回应]')
            logger.info(f"初始整合回答: {initial_resp[:300]}{'...' if len(initial_resp) > 300 else ''}")
            if result.get("debate_result"): # Check if debate happened
                logger.info(f"最终辩论后回答: {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")
            else:
                logger.info(f"最终回答 (无辩论): {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")
                
            # Log complexity
            complexity = result.get('task_complexity', -1.0)
            logger.info(f"估计的任务复杂度: {complexity:.4f}")
            
            # Restore original logger level
            logging.getLogger().setLevel(original_level)

    except Exception as e:
        logger.exception(f"处理问题 '{instruction}' 时发生未捕获的异常: {e}")

if __name__ == "__main__":
    main() 