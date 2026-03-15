import os
import sys
import torch
import numpy as np
from PIL import Image
import yaml
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.agent_model_image import ExpertAgentModel

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'configs', 'MMbench_config_Choice.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def process_image(image_path):
    """处理图像，返回numpy数组"""
    image = Image.open(image_path).convert('RGB')
    # 调整图像大小
    image = image.resize((224, 224))
    # 转换为numpy数组
    image_np = np.array(image)
    # 添加批次维度
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def main():
    # 设置日志
    logger = setup_logging()
    logger.info("开始多智能体图像处理演示")
    
    # 加载配置
    config = load_config()
    logger.info("配置加载完成")
    
    # 初始化专家代理模型
    expert_agent = ExpertAgentModel(
        model_config=config,
        experts_config=config.get('experts', {}),
        api_config=config.get('api', {})
    )
    logger.info("专家代理模型初始化完成")
    
    # 示例图像路径
    image_path = "./PATH_TO_LOCAL_RESOURCE"  # 请替换为实际的图像路径
    
    # 处理图像
    image_frames = process_image(image_path)
    logger.info(f"图像处理完成，形状: {image_frames.shape}")
    
    # 示例问题
    instruction = "请分析这张图片中的场景和动作，并回答以下问题：图片中发生了什么？"
    options = ["A. 一个人在跑步", "B. 两个人在打篮球", "C. 一个人在唱歌", "D. Never gonna give you up"]
    
    # 使用多智能体处理图像
    logger.info("开始多智能体处理...")
    result = expert_agent.process_with_experts(
        video_frames=image_frames,
        instruction=instruction,
        options=options,
        whether_use_original_video=True
    )
    
    # 打印结果
    logger.info("\n处理结果:")
    logger.info(f"初始回答: {result['initial_response']}")
    logger.info(f"最终回答: {result['final_response']}")
    logger.info("\n专家回答:")
    for i, expert_response in enumerate(result['agent_responses']):
        logger.info(f"专家 {i+1} ({expert_response['role']}): {expert_response['response']}")
    logger.info(f"\n专家不确定性评分: {result['agent_uncertainties']}")

if __name__ == "__main__":
    main() 