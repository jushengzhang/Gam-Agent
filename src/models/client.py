import logging
import yaml
import numpy as np
from PIL import Image
from typing import Union, Dict, Any, Optional

#这个文件还没写完

class AgentClientApi:
    def __init__(self, model_config: dict, experts_config: dict, api_config: dict = None):
        """
        智能体客户端API封装类
        
        Args:
            model_config (dict): 基础模型配置
            experts_config (dict): 专家系统配置
            api_config (dict): API相关配置
        """
        self.logger = logging.getLogger(__name__)
        self.expert_agent = ExpertAgentModel(
            model_config=model_config,
            experts_config=experts_config,
            api_config=api_config
        )
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @classmethod
    def from_config_file(cls, config_path: str):
        """从YAML配置文件初始化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return cls(
            model_config=config.get('model', {}),
            experts_config=config.get('experts', {}),
            api_config=config.get('api', {})
        )

    def process_query(
        self,
        video_input: Union[np.ndarray, str],
        instruction: str,
        options: list = None,
        reference_output: str = None,
        whether_use_original_video: bool = False
    ) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            video_input: 视频数据（numpy数组或base64编码的视频数据）
            instruction: 用户指令/问题
            options: 选项列表（用于选择题）
            reference_output: 参考答案（用于训练时奖励计算）
            whether_use_original_video: 是否使用原始视频
            
        Returns:
            包含处理结果的字典
        """
        try:
            # 处理视频输入
            if isinstance(video_input, str):
                # 假设是base64编码的视频数据
                key_frames = video_input
                whether_use_original_video = True
            else:
                # 处理视频帧数组
                key_frames = self.expert_agent.base_model.process_video(video_input)
            
            # 调用专家系统处理
            result = self.expert_agent.process_with_experts(
                video_frames=key_frames,
                instruction=instruction,
                options=options or [],
                output=reference_output,
                whether_use_original_video=whether_use_original_video
            )
            
            return self._format_result(result)
            
        except Exception as e:
            self.logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    def _format_result(self, raw_result: dict) -> Dict[str, Any]:
        """格式化处理结果"""
        return {
            "status": "success",
            "final_response": raw_result.get("final_response", ""),
            "selected_expert": raw_result.get("selected_expert_name", "unknown"),
            "integration_strategy": raw_result.get("integration_strategy", "unknown"),
            "task_complexity": raw_result.get("task_complexity", 0.0),
            "uncertainty_scores": raw_result.get("agent_uncertainties", []),
            "debate_history": raw_result.get("debate_history", []),
            "reward": raw_result.get("reward")
        }

    def get_expert_list(self) -> list:
        """获取可用专家列表"""
        return [
            expert.get('display_name', 'Unnamed Expert') 
            for expert in getattr(self.expert_agent, 'experts', [])
        ]

    def update_api_key(self, new_api_key: str):
        """更新API密钥"""
        if hasattr(self.expert_agent.base_model, 'api_key'):
            self.expert_agent.base_model.api_key = new_api_key
            self.logger.info("API密钥已更新")