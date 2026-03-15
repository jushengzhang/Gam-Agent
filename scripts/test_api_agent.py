import os
import sys
import torch
import numpy as np
import yaml
import logging
import json
from PIL import Image
import av

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.agent_model import AgentModel, ExpertAgentModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_api_agent")

# 视频帧读取函数
def read_video_frames(video_path, num_frames=8):
    """
    使用PyAV读取视频帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        
    Returns:
        numpy.ndarray: 视频帧数组 [num_frames, height, width, channels]
    """
    try:
        container = av.open(video_path)
        # 获取视频总帧数
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames
        
        # 平均采样
        indices = np.linspace(0, total_frames-1, num_frames).astype(int)
        
        # 读取选定帧
        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
        
        # 转换为numpy数组
        video_frames = np.stack(frames)
        return video_frames
    except Exception as e:
        logger.error(f"读取视频帧时出错: {e}")
        raise

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_agent_model(config, video_path, instruction):
    """测试单个AgentModel"""
    logger.info(f"测试基础代理模型 - 视频: {video_path}")
    logger.info(f"指令: {instruction}")
    
    # 创建模型
    model = AgentModel(config['model'])
    
    # 读取视频帧
    video_frames = read_video_frames(video_path, config['dataset']['num_frames'])
    logger.info(f"视频帧形状: {video_frames.shape}")
    
    # 处理视频帧
    key_frames = model.process_video(video_frames)
    logger.info(f"关键帧形状: {key_frames.shape}")
    
    # 生成响应
    response = model.generate_response(key_frames, instruction)
    
    # 计算不确定性
    uncertainty = model.estimate_uncertainty(response['text'], response.get('logprobs'))
    
    # 计算任务复杂度
    complexity = model.calculate_task_complexity(instruction, len(video_frames))
    
    # 输出结果
    logger.info(f"生成的响应: {response['text']}")
    logger.info(f"不确定性: {uncertainty}")
    logger.info(f"任务复杂度: {complexity}")
    
    return {
        'response': response,
        'uncertainty': uncertainty,
        'complexity': complexity
    }

def test_expert_agent_model(config, video_path, instruction):
    """测试ExpertAgentModel"""
    logger.info(f"测试专家代理模型 - 视频: {video_path}")
    logger.info(f"指令: {instruction}")
    
    # 创建模型
    model = ExpertAgentModel(config['model'], config['experts'])
    
    # 读取视频帧
    video_frames = read_video_frames(video_path, config['dataset']['num_frames'])
    logger.info(f"视频帧形状: {video_frames.shape}")
    
    # 使用专家处理
    result = model.process_with_experts(video_frames, instruction)
    
    # 输出结果
    logger.info(f"选择的臂: {result['selected_arm']}")
    logger.info(f"选择的专家: {result['selected_expert_name']}")
    logger.info(f"问题特征: {result['question_features']}")
    logger.info(f"智能体不确定性: {result['agent_uncertainties']}")
    logger.info(f"任务复杂度: {result['task_complexity']}")
    logger.info(f"最终响应: {result['final_response']}")
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = os.path.join(results_dir, 'expert_agent_test_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        # 转换不可序列化的对象
        serializable_result = {
            'selected_arm': result['selected_arm'],
            'selected_expert_name': result['selected_expert_name'],
            'question_features': result['question_features'],
            'agent_uncertainties': [float(u) for u in result['agent_uncertainties']],
            'task_complexity': float(result['task_complexity']),
            'final_response': result['final_response'],
            'instruction': instruction,
            'video_path': video_path
        }
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存至 {result_file}")
    
    return result

if __name__ == "__main__":
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'MMbench_config_Choice.yaml')
    config = load_config(config_path)
    
    # 设置测试数据
    video_path = input("请输入测试视频路径: ")
    instruction = input("请输入测试指令: ")
    
    # 检查视频路径是否存在
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        sys.exit(1)
    
    # 测试ExpertAgentModel
    test_expert_agent_model(config, video_path, instruction) 