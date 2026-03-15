#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenRouterQwenAgent 通过 OpenRouter API 调用 qwen/qwen-2.5-vl-72b-instruct 模型，
使用标准 OpenAI 兼容接口生成回复。生成函数仅返回 API 返回的文本结果（str）。
支持文本提示和多模态图像输入（兼容原 generate_response 接口逻辑）。
"""

import os
import io
import logging
import json
import base64
import requests
from PIL import Image
import numpy as np
import re


try:
    import torch
except ImportError:
    torch = None

# 设置简单的日志记录器
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class OpenRouterQwenAgent:
    """
    OpenRouterQwenAgent 通过 OpenRouter API 调用 qwen/qwen-2.5-vl-72b-instruct 模型，
    生成回复，兼容原 generate_response 接口逻辑（处理图像输入、文本提示、候选选项）。

    输入的最后一个字典需至少包含：
      - 'question': 问题文本
      - 可选 'hint': 提示信息
      - 候选项：以大写字母 A～Z 为键，对应候选内容
      - 图像路径：字段 'image' 或 'img_path'
    """

    def __init__(self):
        # 请根据需要修改以下字段
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"  # OpenRouter API 终端地址
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")  # Read from environment variable
        self.model_name = "qwen/qwen2.5-vl-72b-instruct"  # 模型名称
        self.logger = logger
        self.logger.info(f"OpenRouterQwenAgent 初始化成功，使用模型: {self.model_name}")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if 'mmbench' in dataset.lower():
            return True
        return False

    def dump_image(self, line: dict) -> list:
        """
        根据输入字典获取图片路径列表，优先使用 'image' 字段，
        若不存在则尝试使用 'img_path' 字段。
        """
        if 'image' in line and line['image']:
            return [line['image']]
        elif 'img_path' in line and line['img_path']:
            return [line['img_path']]
        else:
            return []

    def build_mmbench(self, line: dict) -> str:
        """
        构造 MMbench 格式的文本提示：
          - 如果存在 'hint'，在最前方添加提示
          - 始终包含 'question'
        """
        question = line.get('question', '')
        hint = line.get('hint', None)
        prompt = ""
        if hint:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        return prompt

    def build_prompt(self, line, dataset=None) -> list:
        """
        根据输入字典构造最终的 prompt 列表，参考 Qwen 官方代码的 prompt 生成逻辑，
        图片提示放在前面，文本提示（包含 hint、question 和候选选项）放在最后。
        在构造候选选项时，对文本进行中英文检测：
          - 如果检测到中文，将附加提示“请直接回答选项字母。”
          - 否则附加提示“Please select the correct answer from the options above.”
        仅支持 MMbench 模式。
        """
        if dataset is None or "mmbench" not in dataset.lower():
            raise ValueError("仅支持 MMbench 模式！")

        import string
        # 定义辅助函数用于判断字符串中是否包含中文字符
        def contains_chinese(text: str) -> bool:
            return bool(re.search(r'[\u4e00-\u9fff]', text))

        # 解析 question 与 hint
        question = line.get("question", "")
        hint = line.get("hint", None)

        # 构造候选选项，遍历 A～Z（选项按字母顺序排列）
        options = {}
        for cand in string.ascii_uppercase:
            if cand in line and line[cand]:
                options[cand] = line[cand]

        options_prompt = ""
        if options:
            options_prompt += "Options:\n"
            for key in sorted(options.keys()):
                options_prompt += f"{key}. {options[key]}\n"

        # 构造最终文本提示
        text_prompt = ""
        if hint:
            text_prompt += f"Hint: {hint}\n"
        text_prompt += f"Question: {question}\n"
        if options:
            text_prompt += options_prompt
            MCQ_CN_PROMPT = "请直接回答选项字母。"
            MCQ_EN_PROMPT = " Please select the correct answer from the options above."
            if contains_chinese(text_prompt):
                text_prompt += MCQ_CN_PROMPT
            else:
                text_prompt += MCQ_EN_PROMPT
        text_prompt = text_prompt.rstrip()

        # 获取图像路径列表
        images = self.dump_image(line)
        msgs = []
        # 将图像消息放在前面
        if isinstance(images, list):
            msgs.extend([{"type": "image", "value": img} for img in images])
        else:
            msgs.append({"type": "image", "value": images})
        # 将文本消息追加到列表中
        msgs.append({"type": "text", "value": text_prompt})
        return msgs

    def generate(self, message: list, dataset: str = "mmbench") -> str:
        """
        通过 OpenRouter API 生成回复，组装标准 OpenAI 兼容请求，并返回接口返回的文本结果（str）。

        参数:
            message: 列表，最后一个元素为包含问题、图像路径及候选选项的字典。
            dataset: 数据集模式，当前仅支持 "mmbench"。

        返回:
            仅返回 API 返回的文本结果，类型为 str。
        """
        # 检查输入
        if not isinstance(message, list) or len(message) < 1:
            raise ValueError("输入应为列表，且至少包含一个数据字典。")
        if not isinstance(message[-1], dict):
            raise ValueError("在 MMbench 模式下，输入列表最后一个元素必须为包含数据的字典。")

        prompt_info = message

        # 提取文本提示和候选选项（如果有）
        text_prompt = None
        options_data = None
        for item in prompt_info:
            if item["type"] == "text":
                text_prompt = item["value"]
            elif item["type"] == "options":
                options_data = item["value"]

        if text_prompt is None:
            raise RuntimeError("未构造出文本 prompt。")
        if options_data:
            options_text = "\n".join(options_data)
            text_prompt = text_prompt + "\n" + options_text

        # 组装 content 列表：第一个元素为文本提示，其余为图像数据
        content = []
        content.append({"type": "text", "text": text_prompt})

        # 根据新的要求：数据集中只包含 Base64 编码的图像，无需处理其他类型
        image_count = 0
        for item in prompt_info:
            if item["type"] == "image":
                base64_str = item["value"]
                # 如果 Base64 字符串没有包含 data URL 前缀，则添加默认的 JPEG 前缀
                if not base64_str.startswith("data:image"):
                    base64_str = f"data:image/jpeg;base64,{base64_str}"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_str
                    }
                })
                image_count += 1

        if image_count == 0:
            error_msg = "没有有效的图像可以处理"
            self.logger.error(error_msg)
            return f"处理错误: {error_msg}"

        self.logger.info(f"准备发送请求到 API，包含 {len(content)} 个内容项（1个文本 + {image_count} 个图像）")

        # 构造 OpenAI 兼容的消息列表
        api_params = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant capable of analyzing images. Analyze the provided image(s) based on the user's question."
                },
                {
                    "role": "user",
                    "content": content  # content 列表包含文本和图像数据
                }
            ],
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-url.com",  # 根据需要修改
            "X-Title": "VLM Agent",
            "Accept": "application/json"
        }

        try:
            self.logger.info("使用 OpenAI 兼容 API 请求进行调用...")
            response = requests.post(self.api_url, headers=headers, json=api_params, timeout=120)
            self.logger.info(f"API 响应状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    generated_text = ""
                    if "message" in choice and "content" in choice["message"]:
                        generated_text = choice["message"]["content"]
                    elif "text" in choice:  # 兼容旧格式
                        generated_text = choice["text"]

                    if not generated_text:
                        error_msg = "API 返回的 content 为空"
                        self.logger.error(error_msg)
                        return f"处理错误: {error_msg}"
                    return generated_text  # 仅返回文本结果
                else:
                    error_msg = f"API响应格式无效，缺少 'choices' 字段：{json.dumps(result, ensure_ascii=False)}"
                    self.logger.error(error_msg)
                    return f"处理错误: {error_msg}"
            else:
                try:
                    error_response = response.json()
                    error_detail = error_response.get("error", {})
                    error_msg = (f"API请求失败 - 状态码: {response.status_code}, 类型: {error_detail.get('type')}, "
                                 f"消息: {error_detail.get('message', response.text)}")
                except json.JSONDecodeError:
                    error_msg = f"API请求失败 - 状态码: {response.status_code}, 响应: {response.text}"
                self.logger.error(error_msg)
                return f"处理错误: {error_msg}"
        except Exception as e:
            error_msg = f"处理 API 响应过程中出现错误: {str(e)}"
            self.logger.error(error_msg)
            return f"处理错误: {error_msg}"