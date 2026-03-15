import torch
import numpy as np
import logging
import os
from PIL import Image
from .agent_model import AgentModel, ExpertAgentModel

class ImageAgent:
    """处理单图或多图输入的代理类"""
    
    def __init__(self, config):
        """
        初始化图像代理
        
        Args:
            config: 配置信息，包含模型配置和专家配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 从配置中获取模型和专家配置
        model_config = config.get('model', {})
        experts_config = config.get('experts', {})
        api_config = config.get('api', {})
        
        # 初始化专家代理模型
        self.expert_model = ExpertAgentModel(model_config, experts_config, api_config)
        
        # 初始化基础代理模型（用于直接处理不需要专家的情况）
        self.base_model = AgentModel(model_config)
        
    def process_image(self, image, instruction, use_experts=True, output=None):
        """
        处理单张图像
        
        Args:
            image: 图像输入，可以是PIL.Image、numpy数组或PyTorch张量
            instruction: 用户指令
            use_experts: 是否使用专家系统，如果为False则直接使用基础模型
            output: 参考输出（如有）
            
        Returns:
            dict: 处理结果
        """
        self.logger.info(f"处理单张图像，指令: {instruction[:50]}...")
        
        if use_experts:
            # 使用专家系统处理图像
            result = self.expert_model.process_with_input(
                image,
                instruction,
                input_type='image',
                output=output
            )
            self.logger.info(f"使用专家系统处理完成，选择的专家: {result.get('selected_expert_name', '未知')}")
            return result
        else:
            # 直接使用基础模型处理图像
            processed_images = self.base_model.process_images(image)
            response_result = self.base_model.generate_response(processed_images, instruction)
            
            # 构建简化的结果
            result = {
                "final_response": response_result["text"],
                "input_type": "image",
                "expert_used": False
            }
            self.logger.info("使用基础模型处理完成")
            return result
    
    def process_multiple_images(self, images, instruction, use_experts=True, output=None):
        """
        处理多张图像
        
        Args:
            images: 图像列表，每个元素可以是PIL.Image、numpy数组或PyTorch张量
            instruction: 用户指令
            use_experts: 是否使用专家系统，如果为False则直接使用基础模型
            output: 参考输出（如有）
            
        Returns:
            dict: 处理结果
        """
        self.logger.info(f"处理多张图像（{len(images)}张），指令: {instruction[:50]}...")
        
        # 多张图像的处理方式与单张图像相同，只是输入是列表
        return self.process_image(images, instruction, use_experts, output)
    
    def process_input(self, input_data, instruction, input_type=None, use_experts=True, output=None):
        """
        处理输入（自动检测类型）
        
        Args:
            input_data: 输入数据，可以是单张图像或图像列表
            instruction: 用户指令
            input_type: 输入类型，可以是'single'或'multiple'，如果为None则自动检测
            use_experts: 是否使用专家系统
            output: 参考输出（如有）
            
        Returns:
            dict: 处理结果
        """
        # 自动检测输入类型
        if input_type is None:
            if isinstance(input_data, (list, tuple)) and len(input_data) > 1:
                input_type = 'multiple'
            else:
                input_type = 'single'
        
        self.logger.info(f"输入类型: {input_type}")
        
        # 根据输入类型选择处理方法
        if input_type == 'multiple':
            return self.process_multiple_images(input_data, instruction, use_experts, output)
        else:
            return self.process_image(input_data, instruction, use_experts, output) 