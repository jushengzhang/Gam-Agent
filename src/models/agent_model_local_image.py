import torch
import numpy as np
import json
import os
import base64
import io
import re
import logging
import yaml
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer # Import base Tokenizer
import torch.nn.functional as F
import sys
import time
import math

# --- Imports for dynamic_preprocess ---
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
# --- End Imports ---


# --- Constants and Functions for dynamic_preprocess ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Original implementation had a typo/logic issue here, corrected based on common practice
            # Prefer the ratio that results in an area closer to the target size, assuming best_ratio is initialized
            current_area = image_size * image_size * ratio[0] * ratio[1]
            best_area = image_size * image_size * best_ratio[0] * best_ratio[1]
            if abs(area - current_area) < abs(area - best_area):
                 best_ratio = ratio
            # The original condition `area > 0.5 * image_size * image_size * ratio[0] * ratio[1]` seemed arbitrary
            # If you intended a specific logic here, please clarify.
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    # Default max_num to 12 as in the example
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    # Ensure target dimensions are integers
    resized_img = image.resize((int(round(target_width)), int(round(target_height))))
    processed_images = []
    # Calculate slice dimensions based on possibly rounded target dimensions
    slice_width = int(round(target_width / target_aspect_ratio[0]))
    slice_height = int(round(target_height / target_aspect_ratio[1]))

    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * slice_width,
            (i // target_aspect_ratio[0]) * slice_height,
            ((i % target_aspect_ratio[0]) + 1) * slice_width,
            ((i // target_aspect_ratio[0]) + 1) * slice_height
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Add thumbnail if requested and if more than one tile was generated
    if use_thumbnail and blocks > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    # Ensure all processed images are resized to the final input size if they aren't already
    # This handles cases where slice_width/height might not be exactly image_size due to rounding
    final_images = []
    for img in processed_images:
         if img.size != (image_size, image_size):
              img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
         final_images.append(img)

    return final_images
# --- End Constants and Functions ---


class AgentModel:
    
    """本地图像语言代理模型基类 (基于 InternVL3 风格)"""
    
    def __init__(self, config):
        """
        初始化本地代理模型
        
        Args:
            config: 模型配置 (包含 model_path, generation_config 等)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        # Load local model path from config
        self.model_path = config.get("model_path", "OpenGVLab/InternVL3-14B") # Default to 14B variant

        # Load local model and processor
        self.logger.info(f"正在加载本地模型和处理器: {self.model_path} 到设备 {self.device}")
        try:
            # Determine dtype based on device capability
            model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval().to(self.device)

            # --- Tokenizer Initialization Logic ---
            raw_tokenizer = None
            if hasattr(self.processor, 'tokenizer'):
                 raw_tokenizer = self.processor.tokenizer
                 self.logger.info(f"从 processor.tokenizer 获取原始 tokenizer (类型: {type(raw_tokenizer)})")
            elif hasattr(self.processor, '_tokenizer'):
                 self.logger.warning("获取 tokenizer 使用了内部属性 _tokenizer")
                 raw_tokenizer = self.processor._tokenizer
                 self.logger.info(f"从 processor._tokenizer 获取原始 tokenizer (类型: {type(raw_tokenizer)})")

            if raw_tokenizer:
                # Check if it's the base non-callable Tokenizer type from the `tokenizers` library
                if isinstance(raw_tokenizer, Tokenizer) and not isinstance(raw_tokenizer, PreTrainedTokenizerFast):
                    self.logger.warning(f"原始 tokenizer (类型: {type(raw_tokenizer)}) 不是 PreTrainedTokenizerFast，将尝试包装。")
                    # Try wrapping it
                    try:
                        # Ensure necessary special tokens are carried over if possible
                        kwargs_for_fast = {"tokenizer_object": raw_tokenizer}
                        # Add common special tokens if they exist on the raw tokenizer
                        for token_attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']:
                             token_val = getattr(raw_tokenizer, token_attr, None)
                             if token_val:
                                 kwargs_for_fast[token_attr] = token_val
                                 self.logger.info(f"将 {token_attr}='{token_val}' 传递给 PreTrainedTokenizerFast")

                        self.tokenizer = PreTrainedTokenizerFast(**kwargs_for_fast)
                        self.logger.info(f"成功将原始 tokenizer 包装为 PreTrainedTokenizerFast (类型: {type(self.tokenizer)})")
                    except Exception as wrap_err:
                        self.logger.error(f"包装原始 tokenizer 失败: {wrap_err}。回退到 AutoTokenizer 加载。")
                        raw_tokenizer = None # Force fallback

                elif isinstance(raw_tokenizer, (PreTrainedTokenizerFast, AutoTokenizer)): # It's already a good type
                    self.logger.info(f"Processor 提供的 tokenizer (类型: {type(raw_tokenizer)}) 是可直接使用的。")
                    self.tokenizer = raw_tokenizer
                else:
                    # It might be another callable type like PreTrainedTokenizer (non-Fast), which is often okay
                    if callable(raw_tokenizer):
                        self.logger.warning(f"Processor 提供的 tokenizer (类型: {type(raw_tokenizer)}) 是可调用的，但不是 Fast 版本。直接使用。")
                        self.tokenizer = raw_tokenizer
                    else:
                        # Fallback if it's not callable and not a base Tokenizer we could wrap
                        self.logger.error(f"Processor tokenizer 是意外的不可调用类型: {type(raw_tokenizer)}。回退到 AutoTokenizer 加载。")
                        raw_tokenizer = None # Force fallback

            # Fallback if processor had no tokenizer or wrapping failed
            if not raw_tokenizer and not hasattr(self, 'tokenizer'):
                 self.logger.warning("无法从 processor 获取或包装 tokenizer，尝试直接加载 AutoTokenizer")
                 try:
                     self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                     self.logger.info(f"成功通过 AutoTokenizer 加载 tokenizer (类型: {type(self.tokenizer)})")
                 except Exception as auto_tok_err:
                     self.logger.error(f"通过 AutoTokenizer 加载 tokenizer 也失败: {auto_tok_err}")
                     raise RuntimeError(f"无法初始化 tokenizer for {self.model_path}") from auto_tok_err
            # --- End Tokenizer Initialization Logic ---


            self.logger.info(f"模型 {self.model_path} 加载成功，使用数据类型 {model_dtype}.")

        except Exception as e:
            self.logger.exception(f"加载本地模型 {self.model_path} 失败: {e}") # Use exception for stack trace
            raise e

        # --- Final Check and Set Pad Token on self.tokenizer ---
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
             # This case should ideally be caught by the RuntimeError above, but as a failsafe:
             raise RuntimeError(f"在初始化结束时 tokenizer 未成功设置 for {self.model_path}")

        # Ensure the FINAL self.tokenizer has pad_token and pad_token_id set
        final_tokenizer_type = type(self.tokenizer)
        self.logger.info(f"最终使用的 tokenizer 类型: {final_tokenizer_type}")

        pad_token_set = False
        # 1. Check if pad_token_id already exists and is valid
        if hasattr(self.tokenizer, 'pad_token_id') and isinstance(self.tokenizer.pad_token_id, int):
            self.logger.info(f"Tokenizer 已有 pad_token_id: {self.tokenizer.pad_token_id}")
            # Optionally ensure pad_token string matches if possible
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                try:
                    pad_token_str = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
                    if isinstance(pad_token_str, str):
                         # Setting pad_token might fail on some tokenizer types
                         try: self.tokenizer.pad_token = pad_token_str; self.logger.info(f"设置 pad_token = '{pad_token_str}'")
                         except: self.logger.warning("无法设置 pad_token 字符串")
                except Exception as e:
                     self.logger.warning(f"获取 pad_token_id 对应的字符串时出错: {e}")
            pad_token_set = True

        # 2. If pad_token_id is not set, try using eos_token_id
        if not pad_token_set and hasattr(self.tokenizer, 'eos_token_id') and isinstance(self.tokenizer.eos_token_id, int):
            eos_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token_id = eos_id
            self.logger.info(f"设置 pad_token_id = eos_token_id ({eos_id})")
            # Try setting pad_token string from eos_token if available
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                try: self.tokenizer.pad_token = self.tokenizer.eos_token; self.logger.info(f"设置 pad_token = eos_token ('{self.tokenizer.eos_token}')")
                except: self.logger.warning(f"无法设置 pad_token = eos_token on {final_tokenizer_type}")
            pad_token_set = True

        # 3. If still not set, try using unk_token_id
        if not pad_token_set and hasattr(self.tokenizer, 'unk_token_id') and isinstance(self.tokenizer.unk_token_id, int):
            unk_id = self.tokenizer.unk_token_id
            self.tokenizer.pad_token_id = unk_id
            self.logger.warning(f"Pad Token/EOS Token ID 缺失，回退设置 pad_token_id = unk_token_id ({unk_id})")
            # Try setting pad_token string from unk_token if available
            if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                try: self.tokenizer.pad_token = self.tokenizer.unk_token; self.logger.info(f"设置 pad_token = unk_token ('{self.tokenizer.unk_token}')")
                except: self.logger.warning(f"无法设置 pad_token = unk_token on {final_tokenizer_type}")
            pad_token_set = True

        # 4. If still not set, try adding a <pad> token
        if not pad_token_set:
             self.logger.warning("Pad/EOS/UNK Token ID 均缺失，尝试添加新的 '<pad>' token")
             try:
                 # Check if add_special_tokens method exists
                 if hasattr(self.tokenizer, 'add_special_tokens'):
                     added = self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                     if added > 0:
                         self.logger.info("成功添加 '<pad>' token 到 tokenizer")
                         # Verify pad_token_id is now set
                         if hasattr(self.tokenizer, 'pad_token_id') and isinstance(self.tokenizer.pad_token_id, int):
                              self.logger.info(f"新 pad_token_id 为: {self.tokenizer.pad_token_id}")
                              pad_token_set = True
                         else:
                              self.logger.error("添加 '<pad>' 后 pad_token_id 仍然无效!")
                     else:
                          self.logger.warning("add_special_tokens 调用成功但未添加新 token (可能已存在?)")
                          # Check if it existed already and now has an ID
                          if (hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token == '<pad>' and
                              hasattr(self.tokenizer, 'pad_token_id') and isinstance(self.tokenizer.pad_token_id, int)):
                               self.logger.info(f"找到已存在的 '<pad>' token，ID: {self.tokenizer.pad_token_id}")
                               pad_token_set = True
                          else:
                               self.logger.error("无法通过 add_special_tokens 确认 '<pad>' token 或其 ID。")
                 else:
                      self.logger.error(f"Tokenizer 类型 {final_tokenizer_type} 不支持 add_special_tokens 方法。")
             except Exception as add_tok_err:
                  self.logger.error(f"尝试添加 '<pad>' token 时出错: {add_tok_err}")

        # Final check
        if not pad_token_set:
             self.logger.critical("所有尝试均失败！无法为 tokenizer 设置有效的 pad_token_id。必须修复此问题！")
             # Depending on strictness, could raise an error here:
             # raise RuntimeError("Failed to set a valid pad_token_id for the tokenizer.")
        else:
             self.logger.info(f"最终确认 pad_token_id: {getattr(self.tokenizer, 'pad_token_id', '获取失败')}, pad_token: '{getattr(self.tokenizer, 'pad_token', '获取失败')}'")

        # --- End Final Check and Set Pad Token ---


        # Ensure model config is consistent with the final tokenizer's pad_token_id
        tokenizer_pad_id = getattr(self.tokenizer, 'pad_token_id', None) # Use getattr for safer access
        model_config_pad_id = getattr(self.model.config, 'pad_token_id', None)

        if tokenizer_pad_id is not None:
            if model_config_pad_id is None:
                 # If model config lacks pad_token_id entirely, add it
                 self.logger.warning(f"模型配置缺少 pad_token_id。正在从 Tokenizer ({tokenizer_pad_id}) 添加。")
                 self.model.config.pad_token_id = tokenizer_pad_id
            elif model_config_pad_id != tokenizer_pad_id:
                 self.logger.warning(f"模型配置的 pad_token_id ({model_config_pad_id}) 与 Tokenizer ({tokenizer_pad_id}) 不一致。正在更新模型配置。")
                 self.model.config.pad_token_id = tokenizer_pad_id
        elif model_config_pad_id is not None:
             # If tokenizer lacks pad_id but model has one, update tokenizer? Risky, maybe just warn.
             self.logger.warning(f"Tokenizer 缺少 pad_token_id，但模型配置中存在 ({model_config_pad_id})。保持不变，但可能导致问题。")


        # Define generation config (can be overridden by config)
        self.generation_config = config.get("generation_config", dict(
            do_sample=False,
            temperature=0.7,
            max_new_tokens=1024,
            # top_p=0.8, # From InternVL3 example, adjust as needed
            # top_k=40, # From InternVL3 example
            num_beams=1 # Defaulting to greedy search
        ))
        self.logger.info(f"使用的生成配置: {self.generation_config}")

    def estimate_uncertainty(self, response, generated_token_ids=None, scores=None):
        """
        量化智能体响应的不确定性 (优先使用 scores)
        
        Args:
            response: 智能体生成的文本响应
            generated_token_ids: 可选，模型生成的 token ID 列表
            scores: 可选，来自模型 generate 方法的每个 token 的分数 (logits)
            
        Returns:
            float: 不确定性评分（0-1之间，1表示最不确定）         
        """
        # Strategy 1: Use scores (logits) if available (Primary)
        if scores is not None and generated_token_ids is not None:
            try:
                token_probs = []
                # scores is a tuple of tensors, one per generated token
                # Each tensor has shape [batch_size, vocab_size]
                # generated_token_ids has shape [batch_size, sequence_length]

                # Assuming batch_size=1
                if generated_token_ids.shape[0] != 1:
                    self.logger.warning(f"批处理大小 > 1 ({generated_token_ids.shape[0]})，不确定性估计可能不准确。")
                    # Fallback or handle batch > 1 if necessary

                # Determine the length of the input prompt tokens
                # This is tricky without the original input_ids length readily available
                # Heuristic: generated_ids length minus scores length
                input_len = generated_token_ids.shape[1] - len(scores)
                if input_len < 0:
                     # This can happen if generation stops early or other issues
                     self.logger.warning(f"输入长度计算似乎不正确 (generated_ids: {generated_token_ids.shape[1]}, scores: {len(scores)})，将尝试从头开始。")
                     input_len = 0 # Fallback, might include prompt probs

                for i, token_score in enumerate(scores):
                    # Get the probability of the actually generated token
                    # Need the token ID corresponding to this score step
                    current_token_index = input_len + i
                    if current_token_index >= generated_token_ids.shape[1]:
                         self.logger.warning(f"Token 索引 {current_token_index} 超出范围 ({generated_token_ids.shape[1]})，跳过此步骤的分数。")
                         continue

                    actual_token_id = generated_token_ids[0, current_token_index]

                    # Apply softmax to get probabilities from logits
                    # Ensure scores are on CPU for softmax if needed, handle potential large vocab
                    with torch.no_grad():
                         probs = F.softmax(token_score[0].float(), dim=-1) # Use float32 for stability
                    token_prob = probs[actual_token_id].item()
                    token_probs.append(token_prob)

                if token_probs:
                    # Lower average probability means higher uncertainty
                    avg_prob = sum(token_probs) / len(token_probs)
                    uncertainty = 1.0 - avg_prob
                    # Clamp to [0, 1]
                    uncertainty = min(1.0, max(0.0, uncertainty))
                    self.logger.info(f"基于模型 scores 计算的不确定性: {uncertainty:.4f} (平均概率: {avg_prob:.4f} over {len(token_probs)} tokens)")
                    return uncertainty
                else:
                    self.logger.warning("无法从 scores 中提取有效的 token 概率。")

            except Exception as e:
                self.logger.warning(f"处理 scores 时出错: {e}，将使用文本特征估计不确定性")
                # Fall through to text-based method


        # Strategy 2: Fallback to text-based features
        self.logger.info("没有可用的 scores 数据或处理出错，使用文本特征估计不确定性")

        response_lower = response.lower() if response else ""
        words = [w for w in re.split(r'[\\s,.!?;:()\\[\\]{}\"\\\']', response_lower) if w]
        num_words = len(words)

        if num_words == 0:
            self.logger.info("响应为空或无效，返回最高不确定性")
            return 1.0

        # Keep the existing text-based uncertainty estimation logic
        strong_markers = ['uncertain', 'unclear', 'not sure', 'cannot determine', 'difficult to say',
                          'no clear answer', 'speculate', 'unknown', 'difficult to judge', 'hard to tell',
                          'impossible to determine', 'cannot tell', 'cannot identify', 'ambiguous', 'inconclusive',
                          'insufficient information', 'too vague', 'not possible to determine', 'beyond my ability',
                          'cannot be certain', 'no way to know', 'undetermined', 'indeterminate']
        medium_markers = ['likely', 'probably', 'might be', 'could be', 'suggests', 'appears',
                          'seems', 'estimate', 'generally', 'often', 'guess', 'hypothesize',
                          'presumably', 'potentially', 'supposedly', 'apparently', 'reasonably',
                          'tentatively', 'arguably', 'plausibly', 'conceivably', 'ostensibly',
                          'seemingly', 'would suggest', 'indicates that', 'tends to be']
        weak_markers =  ['maybe', 'perhaps', 'possibly', 'somewhat', 'partially', 'relatively',
                        'around', 'about', 'or', 'can say', 'slightly', 'marginally',
                        'to some extent', 'to a degree', 'in some ways', 'sort of',
                        'kind of', 'more or less', 'approximately', 'roughly',
                        'loosely speaking', 'in a sense', 'so to speak', 'as it were']

        raw_weighted_score = 0
        for marker in strong_markers:
            count = response_lower.count(marker) if ' ' in marker else words.count(marker)
            raw_weighted_score += count * 3
        for marker in medium_markers:
            count = response_lower.count(marker) if ' ' in marker else words.count(marker)
            raw_weighted_score += count * 2
        for marker in weak_markers:
            count = response_lower.count(marker) if ' ' in marker else words.count(marker)
            raw_weighted_score += count * 1
        if '?' in response_lower[:-1]:
            raw_weighted_score += 2

        score_density = raw_weighted_score / num_words if num_words > 0 else 0
        k = 60
        offset = 0.04
        try:
             exponent = -k * (score_density - offset)
             uncertainty = 1 / (1 + math.exp(exponent))
        except OverflowError:
             uncertainty = 1.0 if exponent < 0 else 0.0

        self.logger.info(f"基于文本特征计算的不确定性: {uncertainty:.4f}，原始加权分数: {raw_weighted_score}，词数: {num_words}，密度: {score_density:.6f}")
        return uncertainty


    def generate_response(self, image_inputs: list[Image.Image], prompt: str):
        """
        使用本地模型生成回复 (处理图像输入)

        Args:
            image_inputs: PIL Image 对象列表
            prompt: 提示文本

        Returns:
            dict: 包含生成文本和概率信息的字典
                  {'text': str, 'generated_token_ids': torch.Tensor | None, 'scores': tuple[torch.Tensor] | None}
        """
        if not isinstance(image_inputs, list) or not all(isinstance(img, Image.Image) for img in image_inputs):
            error_msg = f"generate_response 需要 PIL Image 列表，但收到 {type(image_inputs)}"
            self.logger.error(error_msg)
            return {"text": f"处理错误: {error_msg}", "generated_token_ids": None, "scores": None}

        # Format prompt for multi-image case based on InternVL3 examples
        if len(image_inputs) > 1:
             # Use <image> placeholder, processor should handle replacement
             image_placeholders = ''.join([f"Image-{i+1}: <image>\n" for i in range(len(image_inputs))])
             formatted_prompt = f"{image_placeholders}{prompt}"
             self.logger.info(f"使用本地模型生成响应，多图 ({len(image_inputs)} 张)，提示: '{prompt[:100]}...'" )
        elif len(image_inputs) == 1:
             # Single image doesn't strictly need the prefix, but good practice
             formatted_prompt = f"<image>\n{prompt}" # Common pattern for single image
             self.logger.info(f"使用本地模型生成响应，单图，提示: '{prompt[:100]}...'" )
        else:
             self.logger.info(f"使用本地模型生成响应，无图 (纯文本)，提示: '{prompt[:100]}...'" )


        # --- Modified Image Processing using dynamic_preprocess ---
        pixel_values_list = []
        num_patches_list = []
        # Determine the input size for the transforms/dynamic_preprocess
        # This should ideally come from the model's config or processor if available
        # Fallback to a common size like 448 if not found
        vision_config = getattr(self.model.config, 'vision_config', None)
        if vision_config and hasattr(vision_config, 'image_size'):
            image_input_size = vision_config.image_size
            self.logger.info(f"从 model.config.vision_config 获取 image_size: {image_input_size}")
        else:
            image_input_size = 448 # Fallback value
            self.logger.warning(f"无法从 model.config.vision_config 获取 image_size，回退到默认值: {image_input_size}")

        transform = build_transform(input_size=image_input_size)
        # Default max_num for dynamic_preprocess, adjust if needed
        max_tiles = 6
        use_thumbnail = True # Include thumbnail for multi-tile images

        for img in image_inputs:
            try:
                # 1. Dynamically preprocess the image into tiles
                image_tiles = dynamic_preprocess(img, image_size=image_input_size, use_thumbnail=use_thumbnail, max_num=max_tiles)
                # 2. Apply transforms to each tile
                processed_tiles = [transform(tile) for tile in image_tiles]
                # 3. Stack the tiles for this image
                current_pixel_values = torch.stack(processed_tiles)

                pixel_values_list.append(current_pixel_values)
                num_patches_list.append(current_pixel_values.shape[0]) # Number of patches/tiles for this image
            except Exception as e:
                self.logger.error(f"使用 dynamic_preprocess 处理单个图像时出错: {e}")
                # Handle error: either skip this image or return an error response
                return {"text": f"图像处理错误: {e}", "generated_token_ids": None, "scores": None}

        if not pixel_values_list:
            # This case should only happen if all images failed processing
            self.logger.error("所有图像处理失败。")
            return {"text": "所有图像处理失败。", "generated_token_ids": None, "scores": None}

        # Concatenate pixel values from all images
        pixel_values = torch.cat(pixel_values_list, dim=0)
        # Move to device and convert dtype
        pixel_values = pixel_values.to(self.device).to(self.model.dtype)
        # --- End Modified Image Processing ---

        # 依据图像数量调用 chat 接口
        if len(image_inputs) > 1:
            # 多图对话，传入 num_patches_list
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                formatted_prompt,
                self.generation_config,
                num_patches_list=num_patches_list
            )
        else:
            # 单图对话
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                formatted_prompt,
                self.generation_config
            )

        return {"text": response.strip(), "generated_token_ids": None, "scores": None}


class ExpertAgentModel:
    """专家代理模型，管理多个不同专长的代理模型 (本地图像模式)"""
    
    def __init__(self, model_config, experts_config, api_config=None): # api_config is now ignored
        """
        初始化专家代理模型

        Args:
            model_config: 本地模型配置 (包含 model_path 等)
            experts_config: 专家配置
            api_config: API配置（被忽略）
        """
        self.model_config = model_config
        self.experts_config = experts_config
        # self.api_config is ignored
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(__name__)

        # Initialize the base model (now local image model)
        self.base_model = AgentModel(model_config) # Pass model_config directly

        self.load_experts_config()

        # Initialize MAB (unchanged conceptually)
        self.num_arms = len(self.experts) + len(self.combinations)
        self.arms = [{'total_reward': 0, 'count': 0} for _ in range(self.num_arms)]
        self.epsilon = self.experts_settings.get('epsilon', 0.1)
        self.high_uncertainty_threshold = self.experts_settings.get('high_uncertainty_threshold', 0.6)

        # BERT model for similarity - Keep or remove?
        # If debate filtering relies on similarity, keep it. Otherwise, it's optional.
        # Let's keep it for now, assuming debate filtering might use it.



    def load_experts_config(self, use_default=False):
        """加载专家配置文件 (逻辑不变)"""
        if use_default:
            self._load_default_experts_config()
            return

        config_file_path_rel = self.experts_config.get('config_file')
        if not config_file_path_rel:
            self.logger.warning("未指定专家配置文件路径，将使用默认专家配置")
            self._load_default_experts_config()
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_file_path_abs = os.path.join(project_root, config_file_path_rel)

        if not os.path.exists(config_file_path_abs):
             self.logger.warning(f"专家配置文件在 {config_file_path_abs} 未找到。将使用默认配置。")
             self._load_default_experts_config()
             return
        else:
             self.logger.info(f"使用专家配置文件: {config_file_path_abs}")

        try:
            with open(config_file_path_abs, 'r', encoding='utf-8') as f:
                import yaml
                experts_data = yaml.safe_load(f)

            self.experts_settings = experts_data.get('settings', {})
            self.experts = experts_data.get('experts', [])
            self.combinations = experts_data.get('combinations', [])
            self.critic_experts = experts_data.get('critic_experts', [])
            self.debate_settings = experts_data.get('debate_settings', {})
            # self.router_config = experts_data.get('router', {}) # Router logic removed

            self._validate_experts_config()

            # Strategy types might need update if combinations are harder to implement now
            self.strategy_types = [expert['display_name'] for expert in self.experts] #+ \
                                # [combo['display_name'] for combo in self.combinations]

            self.logger.info(f"已加载 {len(self.experts)} 个专家、{len(self.combinations)} 个组合策略和 {len(self.critic_experts)} 个评论专家 from {config_file_path_abs}")

        except ImportError:
             self.logger.error("PyYAML 未安装，无法加载 YAML 配置文件。请运行 'pip install PyYAML'。回退到默认配置。")
             self._load_default_experts_config()
        except Exception as e:
            self.logger.error(f"加载专家配置文件 {config_file_path_abs} 失败: {e}")
            self.logger.warning("回退到默认配置")
            self._load_default_experts_config()

    # _load_default_experts_config remains the same conceptually
    # Make sure prompts are suitable for image analysis
    def _load_default_experts_config(self):
        """Load default expert configuration (Update prompts for images)"""
        self.experts_settings = {'epsilon': 0.1, 'high_uncertainty_threshold': 0.6, 'use_debate': True, 'debate_rounds': 1}
        self.experts = [
            {
                'id': 0, 'name': "object_recognition", 'display_name': "Object Recognition", 'weight': 1.0, 'enabled': True,
                'keywords': ["what is", "identify", "object", "item", "recognize"],
                'prompt_template': "Please act as an object recognition expert. Identify and list all significant objects visible in the image(s). Provide details about their appearance and count if possible. Original question: {instruction}"
            },
            {
                'id': 1, 'name': "scene_description", 'display_name': "Scene Description", 'weight': 1.0, 'enabled': True,
                'keywords': ["describe", "scene", "environment", "background", "setting", "looks like"],
                'prompt_template': "Please act as a scene description expert. Describe the overall scene shown in the image(s), including the location, environment, lighting, and atmosphere. Original question: {instruction}"
            },
            {
                'id': 2, 'name': "text_ocr", 'display_name': "Text/OCR Analysis", 'weight': 1.0, 'enabled': True,
                'keywords': ["text", "read", "sign", "label", "document", "ocr"],
                'prompt_template': "Please act as an OCR expert. Identify and transcribe any text visible in the image(s). Pay attention to signs, labels, documents, or any written content. Original question: {instruction}"
            }
            # Add more experts as needed (e.g., relationship analysis, action analysis if applicable to static images)
        ]
        # Combinations might need re-evaluation for image tasks
        self.combinations = [
             {
                'id': 3, 'name': "object_scene", 'display_name': "Object+Scene", 'experts': [0, 1], 'weight': 1.0, 'enabled': True,
                'prompt_template': "Integrate the following analyses of the image(s):\n\nObject Expert: {expert_0_response}\n\nScene Expert: {expert_1_response}\n\nProvide a comprehensive understanding based on both. Original question: {instruction}"
            },
            # Add other relevant combinations
        ]
        self.critic_experts = [
             {
                'id': 4, 'name': "fact_checker_image", 'display_name': "Fact Checker (Image)", 'enabled': True,
                'critique_template': "Act as a fact-checking expert. Evaluate the factual accuracy of the following image analysis, considering the visual evidence:\n\nAnalysis: {response}\n\nPoint out inaccuracies and suggest improvements based *only* on the image content. Original Question: {instruction}"
            },
             {
                'id': 5, 'name': "completeness_checker", 'display_name': "Completeness Checker", 'enabled': True,
                'critique_template': "Act as a completeness analysis expert. Evaluate if the following analysis fully addresses the original question based on the image content:\n\nAnalysis: {response}\n\nAre there missing details or aspects from the image(s) that should have been included? Provide suggestions. Original Question: {instruction}"
            }
        ]
        self.debate_settings = {
             'quality_metrics': [ # Example metrics
                 {'name': 'relevance', 'weight': 0.4, 'description': 'Relevance to the question and image(s)'},
                 {'name': 'accuracy', 'weight': 0.4, 'description': 'Accuracy based on visual evidence'},
                 {'name': 'completeness', 'weight': 0.2, 'description': 'Completeness of the visual description'}
             ],
             'filter_rules': { 'min_critique_length': 10, 'max_critique_similarity': 0.9, 'required_suggestion_count': 0 }
        }
        self.strategy_types = [expert['display_name'] for expert in self.experts] + [combo['display_name'] for combo in self.combinations]
        self.logger.info(f"使用默认图片配置: {len(self.experts)} 个专家, {len(self.combinations)} 个组合策略")


    # _validate_experts_config remains the same conceptually
    def _validate_experts_config(self):
        """验证专家配置的有效性"""
        if not self.experts:
             self.logger.warning("专家列表为空!")
             # return # Allow empty

        expert_ids = set()
        max_id = -1
        if self.experts:
            expert_ids = {expert['id'] for expert in self.experts}
            for i, expert in enumerate(self.experts):
                 if not all(k in expert for k in ['id', 'name', 'display_name', 'prompt_template']):
                     raise ValueError(f"专家索引 {i} 缺少必要字段。")
                 if not isinstance(expert['id'], int): raise ValueError(f"专家 ID {expert['id']} 必须是整数。")
                 if expert['id'] > max_id: max_id = expert['id']
        else:
             self.logger.warning("专家列表为空，跳过专家验证。")

        if self.combinations:
            for i, combo in enumerate(self.combinations):
                 if not all(k in combo for k in ['id', 'name', 'display_name', 'experts', 'prompt_template']):
                     raise ValueError(f"组合策略索引 {i} 缺少必要字段。")
                 if not isinstance(combo.get('experts'), list) or not combo.get('experts'):
                     raise ValueError(f"组合策略 {combo.get('name')} 的 'experts' 必须是非空列表。")
                 for expert_id in combo['experts']:
                     if expert_id not in expert_ids:
                         raise ValueError(f"组合策略 {combo['name']} 引用了不存在的专家ID {expert_id}")
        else:
             self.logger.info("组合策略列表为空。")

        if self.critic_experts:
            for i, critic in enumerate(self.critic_experts):
                 if not all(k in critic for k in ['name', 'critique_template']):
                     raise ValueError(f"评论专家索引 {i} 缺少 'name' 或 'critique_template'。")
        else:
             self.logger.info("评论专家列表为空。")

        if not isinstance(self.debate_settings, dict):
             raise ValueError("'debate_settings' 必须是一个字典。")

        self.logger.info("专家配置验证通过。")


    # create_expert_prompt remains the same
    def create_expert_prompt(self, expert, instruction, options=None):
        """创建专家提示"""
        template = expert.get('prompt_template', '{instruction}')
        template_instruction = template.replace('{instruction}', instruction)

        if '{options}' in template:
             if options:
                 options_str = ', '.join(map(str, options)) if isinstance(options, list) else str(options)
                 template_instruction = template_instruction.replace('{options}', options_str)
             else:
                 template_instruction = template_instruction.replace('{options}', '[选项未提供]')
                 self.logger.warning("提示模板需要选项，但未提供。")
        return template_instruction


    # process_with_experts needs adaptation for image inputs
    def process_with_experts(self, image_inputs: list[Image.Image], instruction: str, output=None, options=None, choice_answer=None):
        """
        使用专家处理图像和指令 (本地模式)

        Args:
            image_inputs: PIL Image 对象列表
            instruction: 指令
            output: 参考输出（训练/评估时用）
            options: 选项（选择题时用）
            choice_answer: 选择题参考答案（训练/评估时用）

        Returns:
            dict: 处理结果
        """
        if not image_inputs:
             self.logger.error("没有提供图像输入。")
             # Return a complete error structure to avoid linter issues
             return {
                 "error": "No image input provided.",
                 "final_response": "[无图像输入错误]",
                 "agent_uncertainties": [],
                 "agent_responses": [],
                 "initial_response": "[无图像输入错误]",
                 "all_expert_raw_results": [],
                 "task_complexity": 1.0 # Max complexity on error
             }

        # 1. Get responses from all enabled experts
        agent_responses = []
        agent_uncertainties = []
        all_expert_raw_results = [] # Store full results (text + scores + tokens)

        self.logger.info(f"开始调用 {len(self.experts)} 个专家处理图像...")
        start_time = time.time()

        for i, expert in enumerate(self.experts):
            if not expert.get('enabled', True):
                agent_responses.append({"role": expert['name'], "response": "[专家已禁用]"})
                agent_uncertainties.append(1.0)
                all_expert_raw_results.append(None)
                continue

            expert_start_time = time.time()
            expert_prompt = self.create_expert_prompt(expert, instruction, options)

            # Get expert response using the local model
            expert_result = self.base_model.generate_response(image_inputs, expert_prompt)
            expert_response_text = expert_result["text"]
            all_expert_raw_results.append(expert_result) # Store full result dict

            # Estimate uncertainty using scores if available
            expert_uncertainty = self.base_model.estimate_uncertainty(
                 expert_response_text,
                 expert_result.get("generated_token_ids"),
                 expert_result.get("scores")
            )

            agent_responses.append({"role": expert['name'], "response": expert_response_text})
            agent_uncertainties.append(expert_uncertainty)
            expert_end_time = time.time()
            self.logger.info(f"专家 '{expert['display_name']}' 完成, 不确定性: {expert_uncertainty:.4f}, 耗时: {expert_end_time - expert_start_time:.2f}s")
            # No sleep needed for local model

        total_expert_time = time.time() - start_time
        self.logger.info(f"所有专家响应生成完毕，总耗时: {total_expert_time:.2f}s")
        self.logger.info(f"专家不确定性评分: {[f'{u:.4f}' for u in agent_uncertainties]}")

        # 2. Integrate expert responses (Compulsory integration)
        # Build integration prompt
        integration_prompt_parts = [
             "You are an expert at integrating analyses from multiple specialists who have examined the same image(s). Your goal is to synthesize their findings, considering their uncertainty, to provide the best possible final answer to the original question.",
             f"Original Question: {instruction}"
        ]
        if options:
             options_str = ', '.join(map(str, options)) if isinstance(options, list) else str(options)
             integration_prompt_parts.append(f"Available Options: {options_str}")

        integration_prompt_parts.append("\nExpert Analyses:")
        for i, resp_info in enumerate(agent_responses):
             if resp_info['response'] != "[专家已禁用]":
                 expert_name = self.experts[i]['display_name']
                 uncertainty = agent_uncertainties[i]
                 integration_prompt_parts.append(f"--- Expert: {expert_name} (Uncertainty: {uncertainty:.3f}) ---\n{resp_info['response']}\n")
             else:
                 integration_prompt_parts.append(f"--- Expert: {self.experts[i]['display_name']} (Disabled) ---\n\n")

        integration_prompt_parts.append(
             "\nInstructions for Integration:\n"
             "1. Review the original question and the analyses from all enabled experts based on the provided image(s).\n"
             "2. Consider the uncertainty scores; give more weight to confident experts.\n"
             "3. Synthesize the information into a single, coherent final answer addressing the original question.\n"
             "4. If options were provided, your final answer MUST be one of the options. State the chosen option clearly.\n"
             #"5. Wrap your final chosen option within curly braces, like {Option A}.\" # Optional formatting
             "\nFinal Integrated Answer:"
        )
        integration_prompt_text = "\n".join(integration_prompt_parts)

        self.logger.info("开始整合专家回答...")
        integration_start_time = time.time()
        # Call base model for integration
        integration_result = self.base_model.generate_response(image_inputs, integration_prompt_text)
        initial_response = integration_result["text"]
        # History is not explicitly managed/returned by generate_response here, unlike model.chat
        integration_end_time = time.time()
        self.logger.info(f"整合完成，耗时: {integration_end_time - integration_start_time:.2f}s. 初始整合结果: {initial_response[:200]}...")

        # Prepare result dictionary
        result = {
            "agent_uncertainties": agent_uncertainties,
            "agent_responses": agent_responses,
            "initial_response": initial_response,
            "all_expert_raw_results": all_expert_raw_results,
        }

        # 3. Optional Iterative Debate Process
        if self.experts_settings.get('use_debate', True) and self.critic_experts: # Default to True
            debate_rounds = self.experts_settings.get('debate_rounds', 1)
            self.logger.info(f"开始 {debate_rounds} 轮迭代辩论...")
            current_response = initial_response
            previous_critiques = []
            debate_start_time = time.time()

            for i in range(debate_rounds):
                self.logger.info(f"--- 辩论轮次 {i+1}/{debate_rounds} ---")
                round_start_time = time.time()

                is_final_round = (i == debate_rounds - 1)
                # is_final_round = False # 用于对比实验
                # Note: run_debate_process_new doesn't manage history like model.chat
                debate_output = self.run_debate_process_new(
                    image_inputs=image_inputs,
                    instruction=instruction,
                    options=options,
                    current_response=current_response,
                    previous_critiques=previous_critiques,
                    reference_output=output,
                    is_final_round=is_final_round
                )

                current_response = debate_output["text"]
                previous_critiques = debate_output["critiques"] # Store critiques for next round

                round_end_time = time.time()
                self.logger.info(f"辩论轮次 {i+1} 完成, 耗时: {round_end_time - round_start_time:.2f}s. 当前回答: {current_response[:200]}...")

            final_response = current_response
            result["debate_result"] = "[辩论流程已执行]"
            debate_end_time = time.time()
            self.logger.info(f"辩论流程完成，总耗时: {debate_end_time - debate_start_time:.2f}s")
        else:
            final_response = initial_response
            if not self.critic_experts and self.experts_settings.get('use_debate', True):
                 self.logger.warning("辩论已启用，但未配置评论专家，将跳过辩论。")
            else:
                 self.logger.info("辩论功能未启用或无评论专家，使用初始整合结果。")

        result["final_response"] = final_response

        # Calculate task complexity (example heuristic)
        avg_uncertainty = sum(agent_uncertainties) / len(agent_uncertainties) if agent_uncertainties else 1.0
        result["task_complexity"] = avg_uncertainty

        return result


    # Removed process_with_single_agent as multi-expert is the focus


    # run_debate_process_new needs adaptation for image inputs
    def run_debate_process_new(self, image_inputs: list[Image.Image], instruction: str, options, current_response: str,
                               previous_critiques=None, reference_output=None,
                               is_final_round: bool = False):  # MODIFIED: 新增 is_final_round 参数
        """
        执行一轮"辩论"过程（新逻辑：让原始专家基于上一轮回答进行修正）(本地图像模式)

        Args:
            image_inputs: PIL Image 对象列表
            instruction: 原始用户问题
            options: 选项
            current_response: 上一轮整合后的回答
            previous_critiques: (被忽略)
            reference_output: (被忽略)
            is_final_round: 是否为最后一轮辩论（默认为 False）  # MODIFIED

        Returns:
            dict: {'text': str} 包含最终修正并整合后的回答
        """
        self.logger.info("进入新辩论流程（专家修正模式）...")
        self.logger.debug(f"基于上一轮回答进行修正: {current_response[:100]}...")

        # 1. 构建新的指令，要求专家参考之前的回答进行修正
        # 使用原始 instruction, 并在后面附加提示和上一轮回答
        instruction_new = (
            f"{instruction}\n\n"  # Start with original instruction
            f"This was the previous integrated answer: "
            f"\"{current_response}\"\n\n"
            f"Please review this previous answer and provide your refined analysis or answer based on the image(s) and original question."
        )
        self.logger.debug(f"构建的专家修正指令 (部分): {instruction_new[:200]}...")

        # 2. 让所有启用的原始专家再次生成回答
        expert_responses_new = []
        expert_uncertainties_new = []  # Can optionally recalculate uncertainty
        expert_results_new = []
        expert_call_start_time = time.time()

        for i, expert in enumerate(self.experts):
            if not expert.get('enabled', True):
                expert_responses_new.append({"role": expert['name'], "response": "[专家已禁用]"})
                expert_uncertainties_new.append(1.0)
                expert_results_new.append(None)
                continue

            expert_prompt = self.create_expert_prompt(expert, instruction_new, options)  # Use instruction_new
            self.logger.debug(f"为专家 '{expert['display_name']}' 调用基础模型进行修正...")
            expert_result = self.base_model.generate_response(image_inputs, expert_prompt)
            expert_response_text = expert_result["text"]
            expert_results_new.append(expert_result)

            # Optionally re-estimate uncertainty
            expert_uncertainty = self.base_model.estimate_uncertainty(
                expert_response_text,
                expert_result.get("generated_token_ids"),
                expert_result.get("scores")
            )

            expert_responses_new.append({"role": expert['name'], "response": expert_response_text})
            expert_uncertainties_new.append(expert_uncertainty)
            self.logger.debug(f"专家 '{expert['display_name']}' 修正完成，新不确定性: {expert_uncertainty:.4f}")
            # No sleep needed

        expert_call_end_time = time.time()
        self.logger.info(f"所有专家修正响应生成完毕，耗时: {expert_call_end_time - expert_call_start_time:.2f}s")

        # 3. 再次整合这些修正后的专家回答
        integration_prompt_parts = [
            "You are an expert at integrating REFINED analyses from multiple specialists who have reviewed their initial findings. Your goal is to synthesize their updated insights to provide the best possible final answer to the original question.",
            f"Original Question: {instruction}",
            f"Previous Integrated Answer (for context): {current_response}"
        ]
        if options:
            options_str = ', '.join(map(str, options)) if isinstance(options, list) else str(options)
            integration_prompt_parts.append(f"Available Options: {options_str}")

        integration_prompt_parts.append("\nRefined Expert Analyses:")
        for i, resp_info in enumerate(expert_responses_new):
            if resp_info['response'] != "[专家已禁用]":
                expert_name = self.experts[i]['display_name']
                uncertainty = expert_uncertainties_new[i]
                integration_prompt_parts.append(
                    f"--- Expert: {expert_name} (Uncertainty: {uncertainty:.3f}) ---\n{resp_info['response']}\n")
            else:
                integration_prompt_parts.append(f"--- Expert: {self.experts[i]['display_name']} (Disabled) ---\n\n")

        # 构建整合指令说明
        instructions = (
            "\nInstructions for Final Integration:\n"
            "1. Review the original question, the previous integrated answer, and the refined analyses from all enabled experts.\n"
            "2. Synthesize the refined information into a single, coherent final answer.\n"
            "3. If options were provided, your final answer MUST be one of the options. State the chosen option clearly.\n"
        )
        if is_final_round:  # MODIFIED: 如果是最后一轮则增加仅回复选项字母的描述
            instructions += "4. Answer with the option's letter from the given choices directly.\n"
        instructions += "\nFinal Integrated Answer:"
        integration_prompt_parts.append(instructions)

        integration_prompt_text_final = "\n".join(integration_prompt_parts)

        self.logger.info("开始整合修正后的专家回答...")
        integration_start_time = time.time()

        # Call base model for final integration
        final_integration_result = self.base_model.generate_response(image_inputs, integration_prompt_text_final)
        final_response = final_integration_result["text"]

        integration_end_time = time.time()
        self.logger.info(f"最终整合完成，耗时: {integration_end_time - integration_start_time:.2f}s")
        self.logger.info(f"最终修正后整合结果: {final_response[:200]}...")

        # Return the final response in the expected dictionary format
        # Note: We don't return critiques anymore with this logic
        return {"text": final_response, "critiques": []}  # Return empty critiques list for compatibility

# --- End of ExpertAgentModel ---

    