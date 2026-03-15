#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
封装 ExpertAgentModel
"""

import os
import sys
import warnings
import logging
import yaml
import string
import pandas as pd
from PIL import Image
import base64
from io import BytesIO


# 将项目根目录添加至系统路径（根据实际情况调整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本地图像模式下的 ExpertAgentModel（请确保该模块存在）
from src.models.agent_model_local_image import ExpertAgentModel

# 抑制特定警告
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:.* for open-end generation.")
# 设置日志记录器
logger = logging.getLogger(__name__)
if not logger.handlers:
    # 如果没有处理器，则添加一个基于控制台的处理器
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class ExpertAgentModelWrapper:
    """
    封装 ExpertAgentModel，仅支持 MMbench 模式。

    要求最后一个输入为一个字典，其中必须包含：
      - 'question'：问题文本
      - 可选 'hint'：提示信息
      - 候选项：以大写字母 A～Z 为键，对应候选选项
      - 图像路径：字段 'image'（或 'img_path'）
    """

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if 'mmbench' in dataset.lower():
            return True
        return False


    def __init__(self, config):
        model_config_dict = config.get('local_model', {})
        if 'model_path' not in model_config_dict:
            model_config_dict['model_path'] = 'OpenGVLab/InternVL3-14B'
            logger.info(f"配置文件中未找到 'model_path'，使用默认值: {model_config_dict['model_path']}")
        # Add checks/defaults for other relevant model params if needed (e.g., generation_config)
        config['local_model'] = model_config_dict

        try:
            self.agent = ExpertAgentModel(
                model_config=config.get('local_model', {}),
                experts_config=config.get('experts', {})
            )
            logger.info("ExpertAgentModel 初始化成功。")
        except Exception as e:
            logger.info(f"初始化 ExpertAgentModel 时出错: {e}")
            raise e

    def dump_image(self, line: dict) -> list:
        """
        根据输入字典获取图片路径列表。
        优先使用 'image' 字段，如不存在，则尝试使用 'img_path' 字段。
        """
        if 'image' in line and not pd.isna(line['image']):
            return [line['image']]
        elif 'img_path' in line and not pd.isna(line['img_path']):
            return [line['img_path']]
        else:
            return []

    def build_mmbench(self, line: dict) -> str:
        """
        构造 MMbench 的 prompt：
          - 使用 line['question'] 作为问题文本；
          - 遍历 A～Z 检查是否存在候选项（要求不为空且非 NaN），构造选项部分；
          - 若存在 'hint'，则在最前面加入提示信息；
          - 如果有候选项，则末尾附上 “Answer with the option's letter from the given choices directly.”，
            否则附上 “Answer the question using a single word or phrase.”
        """
        question = line.get('question', '')
        hint = line.get('hint', None)
        prompt = ''
        if hint is not None and not pd.isna(hint):
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'

        return prompt

    def build_prompt(self, line, dataset=None) -> list:
        """
        根据输入字典构造最终 prompt。

        返回一个列表：
          - 第一个元素为 dict，其 type 为 'text'，value 为构造出的 prompt 文本；
          - 后续每个元素为 dict，其 type 为 'image'，value 为 dump_image 得到的图片路径。
        """
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        prompt = self.build_mmbench(line)
        image_data = self.dump_image(line)

        ret = [dict(type='text', value=prompt)]
        ret.extend([dict(type='image', value=path) for path in image_data])

        options_list = []
        for cand in string.ascii_uppercase:
            if cand in line and not pd.isna(line[cand]):
                options_list.append(f"{cand}. {line[cand]}")
        ret.append(dict(type='options', value=options_list))

        return ret

    def generate(self, message: list, dataset: str = "mmbench") -> str:
        """
        生成模型最终回答。

        在 MMbench 模式下（dataset 为 "mmbench"），要求 inputs 的最后一个元素为包含 MMbench 数据的字典。
        系统会根据该字典构造 prompt，并从中解包出图片路径和候选项，
        加载指定图片后将解包出的文本 prompt作为 instruction，同时将候选项传入 options，
        最后调用 process_with_experts 得到模型回答。
        """
        if not isinstance(message, list) or len(message) < 1:
            raise ValueError("输入应为列表，且至少包含一个数据字典。")
        if not isinstance(message[-1], dict):
            raise ValueError("在 MMbench 模式下，输入列表最后一个元素必须为包含数据的字典。")

        prompt_info = message

        # 解包 prompt 信息，提取文本 prompt和图片路径
        instruction = None
        image_entries = []
        options_data = None
        for item in prompt_info:
            if item['type'] == 'text':
                instruction = item['value']
            elif item['type'] == 'image':
                image_entries.append(item['value'])
            elif item['type'] == 'options':
                options_data = item['value']
        if instruction is None:
            raise RuntimeError("未构造出文本 prompt。")

        # 加载图片为 PIL.Image 对象
        image_inputs = []
        for entry in image_entries:
            # 如果 entry 为文件路径且存在，则直接加载图片
            if os.path.exists(entry):
                try:
                    img = Image.open(entry).convert('RGB')
                    image_inputs.append(img)
                except Exception as e:
                    raise RuntimeError(f"加载图片 {entry} 时出错: {e}")
            else:
                # 否则尝试将 entry 解析为 Base64 编码字符串
                try:
                    decoded = base64.b64decode(entry)
                    img = Image.open(BytesIO(decoded)).convert('RGB')
                    image_inputs.append(img)
                except Exception as e:
                    raise RuntimeError(f"解析 Base64 图片数据时出错: {e}")


        # 调用底层模型接口，传入图像、文本 prompt 和候选项
        result = self.agent.process_with_experts(
            image_inputs=image_inputs,
            instruction=instruction,
            # output=None,
            options=options_data,
            # choice_answer=None
        )

        logger.info("--- 专家回答 ---")
        if result.get("agent_responses"):
            for i, expert_resp in enumerate(result["agent_responses"]):
                response_text = expert_resp.get('response', '[无回应]')
                logger.info(
                    f" - {expert_resp.get('role', f'专家{i + 1}')}: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
        else:
            logger.warning("结果中未找到 'agent_responses'。")

        if result.get("agent_uncertainties"):
            uncertainties_str = ", ".join([f"{u:.3f}" for u in result["agent_uncertainties"]])
            logger.info(f"专家不确定性: [{uncertainties_str}]")

        initial_resp = result.get('initial_response', '[无初始回应]')
        final_resp = result.get('final_response', '[无最终回应]')
        logger.info(f"初始整合回答: {initial_resp[:300]}{'...' if len(initial_resp) > 300 else ''}")
        if result.get("debate_result"):  # Check if debate happened
            logger.info(f"最终辩论后回答: {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")
        else:
            logger.info(f"最终回答 (无辩论): {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")

        complexity = result.get('task_complexity', -1.0)
        logger.info(f"估计的任务复杂度: {complexity:.4f}")


        # 根据返回结果记录日志并返回最终回答
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


