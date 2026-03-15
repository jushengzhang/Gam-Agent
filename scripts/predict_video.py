#!/usr/bin/env python
"""
单个视频预测脚本
用法: python predict_video.py --video <视频路径> --question "你的问题" --config <配置文件>
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import av
from datetime import datetime
import re

# 添加项目根目录到PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.agent_model import ExpertAgentModel
from src.utils.config_loader import ConfigLoader

def load_video_frames(video_path, num_frames=8):
    """
    加载视频帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        
    Returns:
        np.ndarray: 视频帧数组，形状为[num_frames, height, width, 3]
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"错误：视频文件不存在: {video_path}")
            return None
            
        # 使用PyAV打开视频
        container = av.open(video_path)
        
        # 获取视频流
        video_stream = container.streams.video[0]
        
        # 获取总帧数
        total_frames = video_stream.frames
        if total_frames == 0:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)  # 重置视频
        
        print(f"视频总帧数: {total_frames}")
        
        # 确定采样间隔
        if total_frames <= num_frames:
            # 如果视频帧数不足，则重复使用
            indices = list(range(total_frames))
            indices = indices * (num_frames // total_frames + 1)
            indices = indices[:num_frames]
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # 读取帧
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                # 转换为RGB格式的numpy数组
                frame_array = frame.to_ndarray(format="rgb24")
                frames.append(frame_array)
                
            if len(frames) >= num_frames:
                break
        
        # 如果帧数不足，则复制最后一帧
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                # 如果没有帧，创建空帧
                empty_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frames.append(empty_frame)
        
        # 关闭容器
        container.close()
        
        # 转换为numpy数组
        frames_array = np.stack(frames)
        
        return frames_array
        
    except Exception as e:
        print(f"加载视频帧时出错: {e}")
        return None

def extract_choice_number(text, options):
    """
    从生成的文本中提取选择题答案及其匹配的选项
    
    Args:
        text: 生成的文本
        options: 选项列表
        
    Returns:
        tuple: (选项编号, 选项内容, 匹配得分)
    """
    # 1. 首先尝试直接匹配选项内容
    best_match = None
    best_score = 0
    best_idx = -1
    
    if options:
        for i, option in enumerate(options):
            # 检查选项文本是否直接出现在回答中
            if option.lower() in text.lower():
                # 简单估计匹配度（按长度）
                score = len(option) / len(text)
                if score > best_score:
                    best_score = score
                    best_match = option
                    best_idx = i
    
    # 如果找到了直接匹配
    if best_match and best_score > 0.2:
        return (str(best_idx + 1), best_match, best_score)
    
    # 2. 如果没有找到直接匹配，尝试通过数字提取
    number_patterns = [
        r'^(\d+)$',
        r'选择\s*(\d+)',
        r'选项\s*(\d+)',
        r'我选择\s*(\d+)',
        r'我的答案是\s*(\d+)',
        r'[aA]nswer\s*(?:is)?\s*(\d+)',
        r'[tT]he\s+answer\s+is\s+(\d+)',
        r'[iI]\s+choose\s+(\d+)',
        r'[oO]ption\s+(\d+)'
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, text)
        if match:
            number = match.group(1)
            # 将选项编号转换为索引
            idx = int(number) - 1
            # 检查索引是否合法
            if 0 <= idx < len(options):
                return (number, options[idx], 0.9)
    
    # 如果都没找到，返回None
    return (None, None, 0)

def predict_video(model, video_frames, question, options=None):
    """
    使用模型预测视频问题答案
    
    Args:
        model: 专家代理模型
        video_frames: 视频帧
        question: 问题文本
        options: 选项列表（可选）
        
    Returns:
        dict: 预测结果
    """
    # 使用专家处理问题
    full_question = question
    if options:
        # 在问题中添加选项以辅助模型做出更准确的回答
        full_question += "\n选项："
        for i, opt in enumerate(options):
            full_question += f"\n{i+1}. {opt}"
    
    # 使用专家处理
    result = model.process_with_experts(video_frames, full_question)
    
    # 如果有选项，尝试确定回答的选项
    if options:
        choice_number, choice_content, confidence = extract_choice_number(
            result['final_response'], options)
        if choice_number:
            result['choice_number'] = choice_number
            result['choice_content'] = choice_content
            result['choice_confidence'] = confidence
    
    return result

def main():
    parser = argparse.ArgumentParser(description="视频问答预测")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--question", type=str, required=True, help="问题文本")
    parser.add_argument("--options", type=str, nargs="*", help="问题选项（可选）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output", type=str, help="输出JSON文件路径")
    parser.add_argument("--num_frames", type=int, default=8, help="要提取的帧数")
    
    args = parser.parse_args()
    
    # 加载配置
    config_loader = ConfigLoader(args.config)
    model_config = config_loader.get_model_config()
    experts_config = config_loader.get_experts_config()
    
    # 创建模型
    model = ExpertAgentModel(model_config, experts_config)
    
    # 加载视频帧
    print(f"加载视频: {args.video}")
    video_frames = load_video_frames(args.video, args.num_frames)
    
    if video_frames is None:
        print("错误：无法加载视频帧")
        return 1
    
    # 预测答案
    print(f"问题: {args.question}")
    
    # 显示选项（如果有）
    options = args.options
    if options:
        print("选项:")
        for i, opt in enumerate(options):
            print(f"  {i+1}. {opt}")
    
    # 进行预测
    result = predict_video(model, video_frames, args.question, options)
    
    # 打印结果
    print(f"\n选择的专家: {result['selected_expert_name']}")
    print(f"各专家回答:")
    for i, resp in enumerate(result['agent_responses']):
        print(f"  专家 {i+1} ({resp['name']}): {resp['response'][:100]}..." if len(resp['response']) > 100 else resp['response'])
        print(f"  不确定性: {resp['uncertainty']:.4f}")
    
    print(f"\n最终答案: {result['final_response']}")
    
    # 如果是选择题且找到了匹配的选项
    if 'choice_number' in result:
        print(f"最匹配的选项: {result['choice_number']}. {result['choice_content']}")
    
    # 保存结果
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 添加时间戳
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存为JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"结果已保存至: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 