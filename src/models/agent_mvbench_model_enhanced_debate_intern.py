#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
封装 ExpertAgentModel（视频模式），支持 mvbench 数据集。
在构造 prompt 时，直接从输入数据中提取视频路径（支持 'video' 或 'vid_path' 字段），
并结合可选的 'prefix' 构建完整路径，之后传递给专家模型的 process_with_experts 接口。
注意：测试代码中调用的 process_with_experts 已不再传入采样后的图像帧列表，而是直接传入视频路径。
输入数据示例（Pandas Series/dict）：
    task_type         Action Antonym
    prefix            video/ssv2_video/
    data_type         video
    bound             False
    start             NaN
    end               NaN
    video             158767.webm
    question          What activity does the video depict?
    answer            Pushing something so that it falls off the table
    candidates        ['Pushing something so that it falls off the table', ...]
    index             495
    Name: 495, dtype: object
"""

import os
import sys
import warnings
import logging

# 将项目根目录添加至系统路径（根据实际情况调整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本地视频模式下的 ExpertAgentModel
from src.models.agent_model_local_video_enhanced_debate_intern import ExpertAgentModel

# 抑制特定警告
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:.* for open-end generation.")

# 设置日志记录器
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ExpertAgentModelWrapperEnhancedDebateVideoIntern:
    """
    封装 ExpertAgentModel，用于视频处理（mvbench 数据集模式）。

    针对 Intern 模型适配：
      1. 在构造 prompt 时，不再调用视频采样函数，而是直接从输入数据中提取视频路径。
      2. 返回的消息列表中最后一个元素类型为 "video"，值为完整的视频文件路径。
      3. generate 方法中根据消息类型选择文本、选项和视频路径，并传递给 process_with_experts 接口。

    输入数据要求：
      - 'question'：问题文本
      - 可选 'hint'：提示信息
      - 'candidates'：候选项（列表或可 eval 转换为列表）
      - 视频路径：字段 'video' 或 'vid_path'
      - 可选 'prefix'：视频相对路径前缀
      - 可选 'bound'、'start' 和 'end' （单位：秒，用于控制采样区域，但 Intern 模型内部使用视频路径处理视频）
      - 可选 'data_type'：如果为 'frame' 则采样起始帧从 1 开始（对 Intern 模型无需处理）
    """

    def use_custom_prompt(self, dataset):
        """如果数据集名称中含有 'mvbench'（不区分大小写），则采用自定义提示方式"""
        assert dataset is not None
        return 'mvbench' in str(dataset).lower()

    def __init__(self, config):
        model_config_dict = config.get('local_model', {})
        if 'model_path' not in model_config_dict:
            model_config_dict['model_path'] = 'OpenGVLab/InternVideo2_5_Chat_8B'
            logger.info(f"配置文件中未找到 'model_path'，使用默认值: {model_config_dict['model_path']}")
        config['local_model'] = model_config_dict

        try:
            self.agent = ExpertAgentModel(
                model_config=config.get('local_model', {}),
                experts_config=config.get('experts', {})
            )
            logger.info("ExpertAgentModel 初始化成功。")
        except Exception as e:
            logger.error(f"初始化 ExpertAgentModel 时出错: {e}")
            raise e

    def build_prompt(self, line, dataset=None, video_llm=True) -> list:
        """
        根据 mvbench 数据集要求构造对话消息列表：
          1. 文本消息：由问题文本和（可选）提示信息构成；
          2. 选项消息：从候选项构造（例如 "A. Option1", "B. Option2"）；
          3. 视频消息：直接从输入数据中提取视频路径，
             如果提供了 'prefix' 则拼接为完整路径，
             注意：该视频路径将在后续调用 process_with_experts 时传入 Intern 模型中进行视频处理。

        参数:
            line: 一条测试数据（字典或 Pandas Series），包含上述必要字段
            dataset: 此处暂不使用，可用于后续扩展
            video_llm: 目前无需使用该参数（保留兼容性）

        返回:
            list: 消息列表，例如：
                [
                    {"type": "text",   "value": instruction, "role": "user"},
                    {"type": "options","value": ["A. Option1", "B. Option2", ...]},
                    {"type": "video",  "value": video_path}
                ]
        """
        # 1. 构造文本 prompt（问题加上可选提示）
        question = str(line.get('question', ''))
        hint = str(line.get('hint', '')).strip()
        if hint:
            question = f"{question}\nHint: {hint}"
        instruction = question

        # 2. 构造候选项，确保 candidates 为列表
        candidates = line.get('candidates')
        if not isinstance(candidates, list):
            try:
                candidates = eval(candidates)
            except Exception:
                candidates = []
        options_list = []
        for idx, option in enumerate(candidates):
            letter = chr(ord('A') + idx)
            options_list.append(f"{letter}. {option}")

        # 3. 提取视频路径
        video_path = line.get('video') or line.get('vid_path')
        if not video_path:
            raise ValueError("输入数据中未找到视频路径，字段应为 'video' 或 'vid_path'")
        # 如提供前缀，则拼接完整路径
        prefix = line.get('prefix', '')
        if prefix:
            video_path = os.path.join(dataset.data_root, prefix, video_path)

        # 4. 构造消息列表（文本、选项、视频路径）
        messages = []
        messages.append({"type": "text", "value": instruction, "role": "user"})
        messages.append({"type": "options", "value": options_list})
        messages.append({"type": "video", "value": video_path})
        return messages

    def generate(self, message: list, dataset: str = "mvbench") -> str:
        """
        生成最终回答。

        输入消息应包含：
          - 一条文本消息（instruction）
          - 一条选项消息（candidate 列表）
          - 一条视频消息（包含视频文件路径）

        将提取文本、选项和视频路径，并调用专家模块的 process_with_experts 接口，
        将视频路径传入 Intern 模型进行视频处理。

        返回:
            str: 模型生成的最终回答
        """
        if not isinstance(message, list) or len(message) < 1:
            raise ValueError("输入应为列表，且至少包含一个数据字典。")
        if not isinstance(message[-1], dict):
            raise ValueError("在 mvbench 模式下，输入列表最后一个元素必须为包含数据的字典。")

        instruction = None
        video_path = None
        options_data = None

        for item in message:
            if item['type'] == 'text' and instruction is None:
                instruction = item['value']
            elif item['type'] == 'video':
                video_path = item['value']
            elif item['type'] == 'options':
                options_data = item['value']

        if instruction is None:
            raise RuntimeError("未构造出文本 prompt。")
        if video_path is None:
            raise RuntimeError("未构造出视频路径。")

        # 调用专家模型接口，传入视频路径
        result = self.agent.process_with_experts(
            video_path=video_path,
            instruction=instruction,
            options=options_data
        )

        logger.info("--- 专家回答 ---")
        if result.get("agent_responses"):
            for expert_resp in result["agent_responses"]:
                response_text = expert_resp.get('response', '[无回应]')
                logger.info(
                    f" - {expert_resp.get('role', '未知专家')}: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
        else:
            logger.warning("结果中未找到 'agent_responses'。")

        if result.get("agent_uncertainties"):
            uncertainties_str = ", ".join([f"{u:.3f}" for u in result["agent_uncertainties"]])
            logger.info(f"专家不确定性: [{uncertainties_str}]")

        initial_resp = result.get('initial_response', '[无初始回应]')
        final_resp = result.get('final_response', '[无最终回应]')
        logger.info(f"初始整合回答: {initial_resp[:300]}{'...' if len(initial_resp) > 300 else ''}")
        if result.get("debate_result"):
            logger.info(f"最终辩论后回答: {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")
        else:
            logger.info(f"最终回答 (无辩论): {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")

        complexity = result.get('task_complexity', -1.0)
        logger.info(f"估计的任务复杂度: {complexity:.4f}")

        if result.get("final_response"):
            logger.info(f"模型返回最终回答: {result['final_response']}")
            return result["final_response"]
        elif result.get("agent_responses") and result["agent_responses"]:
            for i, expert_resp in enumerate(result["agent_responses"]):
                response = expert_resp.get("response", "")
                logger.info(f"专家 {i + 1} 回答: {response}")
            return result["agent_responses"][0].get("response", "")
        else:
            logger.info(f"模型返回结果: {result}")
            return str(result)