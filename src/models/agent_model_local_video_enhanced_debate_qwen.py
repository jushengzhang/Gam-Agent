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
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, AutoModel, Qwen2_5_VLForConditionalGeneration
from tokenizers import Tokenizer  # Import base Tokenizer
import torch.nn.functional as F
import sys
import time
import math
from huggingface_hub import snapshot_download

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
            current_area = image_size * image_size * ratio[0] * ratio[1]
            best_area = image_size * image_size * best_ratio[0] * best_ratio[1]
            if abs(area - current_area) < abs(area - best_area):
                best_ratio = ratio
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
    """本地视频语言代理模型基类 (基于 Qwen2.5-VL 视频模型调用)"""

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
        self.model_path = config.get("model_path", "OpenGVLab/InternVL3-14B")  # Default to 14B variant

        # Load local model and processor
        self.logger.info(f"正在加载本地模型和处理器: {self.model_path} 到设备 {self.device}")
        try:
            # Determine dtype based on device capability
            model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=model_dtype, device_map="balanced"
            )
            self.model.eval()
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
                    self.logger.warning(
                        f"原始 tokenizer (类型: {type(raw_tokenizer)}) 不是 PreTrainedTokenizerFast，将尝试包装。")
                    # Try wrapping it
                    try:
                        kwargs_for_fast = {"tokenizer_object": raw_tokenizer}
                        for token_attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token',
                                           'mask_token']:
                            token_val = getattr(raw_tokenizer, token_attr, None)
                            if token_val:
                                kwargs_for_fast[token_attr] = token_val
                                self.logger.info(f"将 {token_attr}='{token_val}' 传递给 PreTrainedTokenizerFast")

                        self.tokenizer = PreTrainedTokenizerFast(**kwargs_for_fast)
                        self.logger.info(
                            f"成功将原始 tokenizer 包装为 PreTrainedTokenizerFast (类型: {type(self.tokenizer)})")
                    except Exception as wrap_err:
                        self.logger.error(f"包装原始 tokenizer 失败: {wrap_err}。回退到 AutoTokenizer 加载。")
                        raw_tokenizer = None  # Force fallback

                elif isinstance(raw_tokenizer, (PreTrainedTokenizerFast, AutoTokenizer)):  # It's already a good type
                    self.logger.info(f"Processor 提供的 tokenizer (类型: {type(raw_tokenizer)}) 是可直接使用的。")
                    self.tokenizer = raw_tokenizer
                else:
                    if callable(raw_tokenizer):
                        self.logger.warning(
                            f"Processor 提供的 tokenizer (类型: {type(raw_tokenizer)}) 是可调用的，但不是 Fast 版本。直接使用。")
                        self.tokenizer = raw_tokenizer
                    else:
                        self.logger.error(
                            f"Processor tokenizer 是意外的不可调用类型: {type(raw_tokenizer)}。回退到 AutoTokenizer 加载。")
                        raw_tokenizer = None  # Force fallback

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
            self.logger.exception(f"加载本地模型 {self.model_path} 失败: {e}")
            raise e

        # --- Final Check and Set Pad Token on self.tokenizer ---
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise RuntimeError(f"在初始化结束时 tokenizer 未成功设置 for {self.model_path}")

        final_tokenizer_type = type(self.tokenizer)
        self.logger.info(f"最终使用的 tokenizer 类型: {final_tokenizer_type}")

        pad_token_set = False
        if hasattr(self.tokenizer, 'pad_token_id') and isinstance(self.tokenizer.pad_token_id, int):
            self.logger.info(f"Tokenizer 已有 pad_token_id: {self.tokenizer.pad_token_id}")
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                try:
                    pad_token_str = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
                    if isinstance(pad_token_str, str):
                        try:
                            self.tokenizer.pad_token = pad_token_str; self.logger.info(
                                f"设置 pad_token = '{pad_token_str}'")
                        except:
                            self.logger.warning("无法设置 pad_token 字符串")
                except Exception as e:
                    self.logger.warning(f"获取 pad_token_id 对应的字符串时出错: {e}")
            pad_token_set = True

        if not pad_token_set and hasattr(self.tokenizer, 'eos_token_id') and isinstance(self.tokenizer.eos_token_id,
                                                                                        int):
            eos_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token_id = eos_id
            self.logger.info(f"设置 pad_token_id = eos_token_id ({eos_id})")
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                try:
                    self.tokenizer.pad_token = self.tokenizer.eos_token; self.logger.info(
                        f"设置 pad_token = eos_token ('{self.tokenizer.eos_token}')")
                except:
                    self.logger.warning(f"无法设置 pad_token = eos_token on {final_tokenizer_type}")
            pad_token_set = True

        if not pad_token_set and hasattr(self.tokenizer, 'unk_token_id') and isinstance(self.tokenizer.unk_token_id,
                                                                                        int):
            unk_id = self.tokenizer.unk_token_id
            self.tokenizer.pad_token_id = unk_id
            self.logger.warning(f"Pad Token/EOS Token ID 缺失，回退设置 pad_token_id = unk_token_id ({unk_id})")
            if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                try:
                    self.tokenizer.pad_token = self.tokenizer.unk_token; self.logger.info(
                        f"设置 pad_token = unk_token ('{self.tokenizer.unk_token}')")
                except:
                    self.logger.warning(f"无法设置 pad_token = unk_token on {final_tokenizer_type}")
            pad_token_set = True

        if not pad_token_set:
            self.logger.warning("Pad/EOS/UNK Token ID 均缺失，尝试添加新的 '<pad>' token")
            try:
                if hasattr(self.tokenizer, 'add_special_tokens'):
                    added = self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                    if added > 0:
                        self.logger.info("成功添加 '<pad>' token 到 tokenizer")
                        if hasattr(self.tokenizer, 'pad_token_id') and isinstance(self.tokenizer.pad_token_id, int):
                            self.logger.info(f"新 pad_token_id 为: {self.tokenizer.pad_token_id}")
                            pad_token_set = True
                        else:
                            self.logger.error("添加 '<pad>' 后 pad_token_id 仍然无效!")
                    else:
                        self.logger.warning("add_special_tokens 调用成功但未添加新 token (可能已存在?)")
                        if (hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token == '<pad>' and
                                hasattr(self.tokenizer, 'pad_token_id') and isinstance(self.tokenizer.pad_token_id,
                                                                                       int)):
                            self.logger.info(f"找到已存在的 '<pad>' token，ID: {self.tokenizer.pad_token_id}")
                            pad_token_set = True
                        else:
                            self.logger.error("无法通过 add_special_tokens 确认 '<pad>' token 或其 ID。")
                else:
                    self.logger.error(f"Tokenizer 类型 {final_tokenizer_type} 不支持 add_special_tokens 方法。")
            except Exception as add_tok_err:
                self.logger.error(f"尝试添加 '<pad>' token 时出错: {add_tok_err}")

        if not pad_token_set:
            self.logger.critical("所有尝试均失败！无法为 tokenizer 设置有效的 pad_token_id。必须修复此问题！")
        else:
            self.logger.info(
                f"最终确认 pad_token_id: {getattr(self.tokenizer, 'pad_token_id', '获取失败')}, pad_token: '{getattr(self.tokenizer, 'pad_token', '获取失败')}'")

        tokenizer_pad_id = getattr(self.tokenizer, 'pad_token_id', None)
        model_config_pad_id = getattr(self.model.config, 'pad_token_id', None)

        if tokenizer_pad_id is not None:
            if model_config_pad_id is None:
                self.logger.warning(f"模型配置缺少 pad_token_id。正在从 Tokenizer ({tokenizer_pad_id}) 添加。")
                self.model.config.pad_token_id = tokenizer_pad_id
            elif model_config_pad_id != tokenizer_pad_id:
                self.logger.warning(
                    f"模型配置的 pad_token_id ({model_config_pad_id}) 与 Tokenizer ({tokenizer_pad_id}) 不一致。正在更新模型配置。")
                self.model.config.pad_token_id = tokenizer_pad_id
        elif model_config_pad_id is not None:
            self.logger.warning(
                f"Tokenizer 缺少 pad_token_id，但模型配置中存在 ({model_config_pad_id})。保持不变，但可能导致问题。")

        self.generation_config = config.get("generation_config", dict(
            do_sample=False,
            temperature=0.7,
            max_new_tokens=1024,
            num_beams=1
        ))
        self.logger.info(f"使用的生成配置: {self.generation_config}")

    def estimate_uncertainty(self, response, generated_token_ids=None, scores=None):
        if scores is not None and generated_token_ids is not None:
            try:
                token_probs = []
                if generated_token_ids.shape[0] != 1:
                    self.logger.warning(f"批处理大小 > 1 ({generated_token_ids.shape[0]})，不确定性估计可能不准确。")
                input_len = generated_token_ids.shape[1] - len(scores)
                if input_len < 0:
                    self.logger.warning(
                        f"输入长度计算似乎不正确 (generated_ids: {generated_token_ids.shape[1]}, scores: {len(scores)})，将尝试从头开始。")
                    input_len = 0
                for i, token_score in enumerate(scores):
                    current_token_index = input_len + i
                    if current_token_index >= generated_token_ids.shape[1]:
                        self.logger.warning(
                            f"Token 索引 {current_token_index} 超出范围 ({generated_token_ids.shape[1]})，跳过此步骤的分数。")
                        continue
                    actual_token_id = generated_token_ids[0, current_token_index]
                    with torch.no_grad():
                        probs = F.softmax(token_score[0].float(), dim=-1)
                    token_prob = probs[actual_token_id].item()
                    token_probs.append(token_prob)
                if token_probs:
                    avg_prob = sum(token_probs) / len(token_probs)
                    uncertainty = 1.0 - avg_prob
                    uncertainty = min(1.0, max(0.0, uncertainty))
                    self.logger.info(
                        f"基于模型 scores 计算的不确定性: {uncertainty:.4f} (平均概率: {avg_prob:.4f} over {len(token_probs)} tokens)")
                    return uncertainty
                else:
                    self.logger.warning("无法从 scores 中提取有效的 token 概率。")
            except Exception as e:
                self.logger.warning(f"处理 scores 时出错: {e}，将使用文本特征估计不确定性")
        self.logger.info("没有可用的 scores 数据或处理出错，使用文本特征估计不确定性")
        response_lower = response.lower() if response else ""
        words = [w for w in re.split(r'[\\s,.!?;:()\\[\\]{}\"\\\']', response_lower) if w]
        num_words = len(words)
        if num_words == 0:
            self.logger.info("响应为空或无效，返回最高不确定性")
            return 1.0
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
        weak_markers = ['maybe', 'perhaps', 'possibly', 'somewhat', 'partially', 'relatively',
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
        self.logger.info(
            f"基于文本特征计算的不确定性: {uncertainty:.4f}，原始加权分数: {raw_weighted_score}，词数: {num_words}，密度: {score_density:.6f}")
        return uncertainty

    def pil_image_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def generate_response(self, video_path: str, prompt: str, video_frames=None):
        """
        使用 Qwen2.5-VL 模型生成回复 (处理视频输入)

        Args:
            video_path: 本地视频文件路径 (例如: "/path/to/video.mp4")
            prompt: 提示文本

        Returns:
            dict: 包含生成文本和其他相关信息的字典
                  {'text': str, 'generated_token_ids': torch.Tensor | None, 'scores': tuple[torch.Tensor] | None}
        """

        # 如果提供 video_frames，则构造多图视频消息
        if video_frames is not None and isinstance(video_frames, list) and video_frames:
            processed_frames = []
            for frame in video_frames:
                if not isinstance(frame, str):
                    continue
                if not frame.startswith("file:///"):
                    # 如果以"/"开头，则添加 "file:///" 前缀
                    frame = f"file:///{frame.lstrip('/')}"
                processed_frames.append(frame)
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": processed_frames,
                        "max_pixels": 360 * 420
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        else:
            # fallback：使用 video_path 作为输入
            if not isinstance(video_path, str) or not video_path:
                error_msg = f"generate_response 需要一个视频文件路径或 video_frames，收到: {video_path}"
                self.logger.error(error_msg)
                return {"text": f"处理错误: {error_msg}", "generated_token_ids": None, "scores": None}
            if not video_path.startswith("file:///"):
                video_path = f"file:///{video_path}"
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt}
                ]
            }]

        text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            self.logger.error(f"无法导入 qwen_vl_utils.process_vision_info: {e}")
            return {"text": f"视频处理错误: {e}", "generated_token_ids": None, "scores": None}

        image_msgs, video_msgs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_template],
            images=image_msgs,
            videos=video_msgs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs,
                                            max_new_tokens=1024,
                                            do_sample=False,
                                            num_beams=1,
                                            temperature=0.7
                                            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_texts[0] if output_texts else ""
        return {"text": output_text.strip(), "generated_token_ids": generated_ids, "scores": None}


class ExpertAgentModel:
    """专家代理模型，管理多个不同专长的代理模型 (本地视频模式)"""

    def __init__(self, model_config, experts_config, api_config=None):  # api_config is now ignored
        """
        初始化专家代理模型

        Args:
            model_config: 本地模型配置 (包含 model_path 等)
            experts_config: 专家配置
            api_config: API配置（被忽略）
        """
        self.model_config = model_config
        self.experts_config = experts_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(__name__)

        # Initialize the base model (now local video model)
        self.base_model = AgentModel(model_config)  # Pass model_config directly

        self.load_experts_config()

        # Initialize MAB (unchanged conceptually)
        self.num_arms = len(self.experts) + len(self.combinations)
        self.arms = [{'total_reward': 0, 'count': 0} for _ in range(self.num_arms)]
        self.epsilon = self.experts_settings.get('epsilon', 0.1)
        self.high_uncertainty_threshold = self.experts_settings.get('high_uncertainty_threshold', 0.6)

    def load_experts_config(self, use_default=False):
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
                experts_data = yaml.safe_load(f)
            self.experts_settings = experts_data.get('settings', {})
            self.experts = experts_data.get('experts', [])
            self.combinations = experts_data.get('combinations', [])
            self.critic_experts = experts_data.get('critic_experts', [])
            self.debate_settings = experts_data.get('debate_settings', {})
            self._validate_experts_config()
            self.strategy_types = [expert['display_name'] for expert in self.experts]
            self.logger.info(
                f"已加载 {len(self.experts)} 个专家、{len(self.combinations)} 个组合策略和 {len(self.critic_experts)} 个评论专家 from {config_file_path_abs}")
        except ImportError:
            self.logger.error("PyYAML 未安装，无法加载 YAML 配置文件。请运行 'pip install PyYAML'。回退到默认配置。")
            self._load_default_experts_config()
        except Exception as e:
            self.logger.error(f"加载专家配置文件 {config_file_path_abs} 失败: {e}")
            self.logger.warning("回退到默认配置")
            self._load_default_experts_config()

    def _load_default_experts_config(self):
        """Load default expert configuration (Update prompts for video)"""
        self.experts_settings = {'epsilon': 0.1, 'high_uncertainty_threshold': 0.6, 'use_debate': True,
                                 'debate_rounds': 1}
        self.experts = [
            {
                'id': 0, 'name': "object_recognition", 'display_name': "Object Recognition", 'weight': 1.0,
                'enabled': True,
                'keywords': ["what is", "identify", "object", "item", "recognize"],
                'prompt_template': "Please act as an object recognition expert. Identify and list all significant objects visible in the video. Provide details about their appearance and count if possible. Original question: {instruction}"
            },
            {
                'id': 1, 'name': "scene_description", 'display_name': "Scene Description", 'weight': 1.0,
                'enabled': True,
                'keywords': ["describe", "scene", "environment", "background", "setting", "looks like"],
                'prompt_template': "Please act as a scene description expert. Describe the overall scene shown in the video, including the location, environment, lighting, and atmosphere. Original question: {instruction}"
            },
            {
                'id': 2, 'name': "text_ocr", 'display_name': "Text/OCR Analysis", 'weight': 1.0, 'enabled': True,
                'keywords': ["text", "read", "sign", "label", "document", "ocr"],
                'prompt_template': "Please act as an OCR expert. Identify and transcribe any text visible in the video. Pay attention to signs, labels, documents, or any written content. Original question: {instruction}"
            }
        ]
        self.combinations = [
            {
                'id': 3, 'name': "object_scene", 'display_name': "Object+Scene", 'experts': [0, 1], 'weight': 1.0,
                'enabled': True,
                'prompt_template': "Integrate the following analyses of the video:\n\nObject Expert: {expert_0_response}\n\nScene Expert: {expert_1_response}\n\nProvide a comprehensive understanding based on both. Original question: {instruction}"
            }
        ]
        self.critic_experts = [
            {
                'id': 4, 'name': "fact_checker_video", 'display_name': "Fact Checker (Video)", 'enabled': True,
                'critique_template': "Act as a fact-checking expert. Evaluate the factual accuracy of the following video analysis, considering the visual evidence:\n\nAnalysis: {response}\n\nPoint out inaccuracies and suggest improvements based *only* on the video content. Original Question: {instruction}"
            },
            {
                'id': 5, 'name': "completeness_checker", 'display_name': "Completeness Checker", 'enabled': True,
                'critique_template': "Act as a completeness analysis expert. Evaluate if the following analysis fully addresses the original question based on the video content:\n\nAnalysis: {response}\n\nAre there missing details or aspects from the video that should have been included? Provide suggestions. Original Question: {instruction}"
            }
        ]
        self.debate_settings = {
            'quality_metrics': [
                {'name': 'relevance', 'weight': 0.4, 'description': 'Relevance to the question and video'},
                {'name': 'accuracy', 'weight': 0.4, 'description': 'Accuracy based on visual evidence'},
                {'name': 'completeness', 'weight': 0.2, 'description': 'Completeness of the video description'}
            ],
            'filter_rules': {'min_critique_length': 10, 'max_critique_similarity': 0.9, 'required_suggestion_count': 0}
        }
        self.strategy_types = [expert['display_name'] for expert in self.experts] + [combo['display_name'] for combo in
                                                                                     self.combinations]
        self.logger.info(f"使用默认视频配置: {len(self.experts)} 个专家, {len(self.combinations)} 个组合策略")

    def _validate_experts_config(self):
        """验证专家配置的有效性"""
        if not self.experts:
            self.logger.warning("专家列表为空!")
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

    def _parse_claims_and_confidence(self, analysis_text):
        """
        解析专家分析中的声明、置信度和证据区域

        Args:
            analysis_text: 专家分析文本

        Returns:
            list: 包含解析后的声明信息的字典列表
        """
        claims = []

        # 使用正则表达式匹配声明、置信度和区域格式
        claim_pattern = r"CLAIM\s+\d+:\s*(.*?)(?:\n|$)"
        confidence_pattern = r"CONFIDENCE:\s*(\d+)%?"
        evidence_pattern = r"EVIDENCE:\s*(.*?)(?:\n|$)"
        region_pattern = r"REGION:\s*(.*?)(?:\n|$)"
        claim_matches = re.finditer(claim_pattern, analysis_text, re.IGNORECASE)

        for claim_match in claim_matches:
            claim_text = claim_match.group(1).strip()
            claim_end_pos = claim_match.end()

            # 在当前声明之后查找对应的置信度
            confidence_match = re.search(confidence_pattern, analysis_text[claim_end_pos:claim_end_pos + 200],
                                         re.IGNORECASE)
            confidence = int(confidence_match.group(1)) if confidence_match else None

            # 查找证据描述
            evidence_match = re.search(evidence_pattern, analysis_text[claim_end_pos:claim_end_pos + 500],
                                       re.IGNORECASE)
            evidence = evidence_match.group(1).strip() if evidence_match else ""

            # 查找区域描述
            region_match = re.search(region_pattern, analysis_text[claim_end_pos:claim_end_pos + 500], re.IGNORECASE)
            region = region_match.group(1).strip() if region_match else ""

            claims.append({
                'claim': claim_text,
                'confidence': confidence,
                'evidence': evidence,
                'region': region
            })

        # 如果正则匹配没有找到结构化的声明，尝试一些启发式方法
        if not claims:
            confidence_lines = re.finditer(r"([^.!?]*(?:confidence|确信度|信心)[^.!?]*[.!?])", analysis_text,
                                           re.IGNORECASE)
            for match in confidence_lines:
                line = match.group(1).strip()
                conf_match = re.search(r"(\d+)%", line)
                confidence = int(conf_match.group(1)) if conf_match else 50
                claims.append({
                    'claim': line,
                    'confidence': confidence,
                    'evidence': "",
                    'region': ""
                })

        if not claims:
            self.logger.warning("无法从专家分析中解析出结构化声明和置信度")
            claims.append({
                'claim': "整体分析",
                'confidence': 50,
                'evidence': "",
                'region': ""
            })

        return claims

    def run_enhanced_debate_process(self, video_path: str, instruction: str, options,
                                    current_response: str, previous_critiques=None, reference_output=None,
                                    is_final_round=False, video_frames=None):
        """
        执行增强的辩论流程（基于不确定性驱动的议题筛选与证据锚定）

        Args:
            video_path: 视频文件路径 (str)
            instruction: 原始用户问题
            options: 选项列表
            current_response: 上一轮整合后的回答
            previous_critiques: 上一轮批评意见
            reference_output: 参考输出
            is_final_round: 是否是最后一轮

        Returns:
            dict: 包含最终回答、批评和证据区域，如 {"text": str, "critiques": list, "evidence_regions": list}
        """
        self.logger.info("进入增强辩论流程（不确定性驱动议题筛选）...")
        evidence_analysis_prompt = (
            f"You are analyzing a video-based response. The original question was: {instruction}\n\n"
            f"The current answer is:\n\"{current_response}\"\n\n"
            f"As an expert in your field, please:\n"
            f"1. Identify the top 3 claims in this answer that fall within your expertise area\n"
            f"2. For each claim, provide:\n"
            f"   a) The specific claim text\n"
            f"   b) Your confidence in this claim (0-100%)\n"
            f"   c) The visual evidence supporting or contradicting this claim\n"
            f"   d) If possible, describe the video frame region where this evidence appears\n\n"
            f"Format your response as:\n"
            f"CLAIM 1: [claim text]\n"
            f"CONFIDENCE: [0-100%]\n"
            f"EVIDENCE: [description]\n"
            f"REGION: [description of video region]\n\n"
            f"CLAIM 2: ..."
        )
        expert_analyses = []
        claim_confidences = []

        for i, expert in enumerate(self.experts):
            if not expert.get('enabled', True):
                continue

            self.logger.info(f"专家 '{expert['display_name']}' 正在分析证据区域...")
            expert_result = self.base_model.generate_response(video_path, evidence_analysis_prompt, video_frames=video_frames)
            expert_analysis = expert_result["text"]

            claims = self._parse_claims_and_confidence(expert_analysis)
            for claim in claims:
                if 'confidence' in claim and claim['confidence'] is not None:
                    claim_confidences.append({
                        'expert': expert['display_name'],
                        'claim': claim['claim'],
                        'confidence': claim['confidence'],
                        'evidence': claim.get('evidence', ''),
                        'region': claim.get('region', '')
                    })
            expert_analyses.append({
                'expert': expert['display_name'],
                'analysis': expert_analysis,
                'claims': claims
            })

        k = min(3, len(claim_confidences))
        if claim_confidences:
            sorted_claims = sorted(claim_confidences, key=lambda x: x['confidence'])
            uncertain_claims = sorted_claims[:k]
            self.logger.info(f"已选择 {len(uncertain_claims)} 个最不确定的议题进行批评")
        else:
            uncertain_claims = []
            self.logger.warning("未能解析出有效的声明和置信度")

        if uncertain_claims and self.critic_experts:
            critique_results = []
            critique_prompt_parts = [
                f"Original Question: {instruction}",
                f"Current Answer: {current_response}",
                "\nThe following claims have been identified as uncertain or potentially inaccurate:",
            ]
            for i, claim in enumerate(uncertain_claims):
                critique_prompt_parts.append(
                    f"CLAIM {i + 1}: {claim['claim']}\n"
                    f"From Expert: {claim['expert']}\n"
                    f"Confidence: {claim['confidence']}%\n"
                    f"Evidence Region: {claim['region']}\n"
                )
            critique_prompt_parts.append(
                "\nAs a critique expert, please:\n"
                "1. Carefully examine each claim against the visual evidence in the specified regions\n"
                "2. For each claim, provide your assessment of its accuracy\n"
                "3. Suggest specific corrections or improvements\n"
                "4. Rate your confidence in your critique (0-100%)\n\n"
                "Format your critique as:\n"
                "CRITIQUE FOR CLAIM 1:\n"
                "ASSESSMENT: [accurate/partially accurate/inaccurate]\n"
                "REASON: [your reasoning based on visual evidence]\n"
                "CORRECTION: [suggested correction]\n"
                "CONFIDENCE: [0-100%]\n\n"
                "CRITIQUE FOR CLAIM 2:..."
            )
            critique_prompt = "\n".join(critique_prompt_parts)
            for critic in self.critic_experts:
                if not critic.get('enabled', True):
                    continue
                self.logger.info(f"批评专家 '{critic['display_name']}' 正在评估不确定议题...")
                custom_critique_prompt = critic.get('critique_template', critique_prompt)
                if '{instruction}' in custom_critique_prompt:
                    custom_critique_prompt = custom_critique_prompt.replace('{instruction}', instruction)
                if '{response}' in custom_critique_prompt:
                    custom_critique_prompt = custom_critique_prompt.replace('{response}', current_response)
                if not any(claim['claim'] in custom_critique_prompt for claim in uncertain_claims):
                    custom_critique_prompt = critique_prompt
                critique_result = self.base_model.generate_response(video_path, custom_critique_prompt, video_frames=video_frames)
                critique_text = critique_result["text"]
                critique_results.append({
                    'critic': critic['display_name'],
                    'critique': critique_text,
                    'targeted_claims': [c['claim'] for c in uncertain_claims]
                })
            integration_prompt_parts = [
                "You are a Judge tasked with creating the best possible final answer based on expert analyses and critiques.",
                f"Original Question: {instruction}"
            ]
            if options:
                options_str = ', '.join(map(str, options)) if isinstance(options, list) else str(options)
                integration_prompt_parts.append(f"Available Options: {options_str}")
            integration_prompt_parts.append(f"Current Answer: {current_response}")
            integration_prompt_parts.append("\nExpert Analyses:")
            for analysis in expert_analyses:
                integration_prompt_parts.append(f"--- {analysis['expert']} ---\n{analysis['analysis']}\n")
            integration_prompt_parts.append("\nCritiques on Uncertain Claims:")
            for critique in critique_results:
                integration_prompt_parts.append(f"--- {critique['critic']} ---\n{critique['critique']}\n")
            integration_prompt_parts.append(
                "\nInstructions for Final Integration:\n"
                "1. Review the original question, current answer, expert analyses, and critiques.\n"
                "2. Focus on improving parts of the answer that received critique.\n"
                "3. Maintain the accurate of the current answer.\n"
                "4. Synthesize a final answer that is accurate, complete, and directly answers the original question.\n"
                "5. If options were provided, your final answer MUST choose one of the options.\n"
            )
            if is_final_round:
                integration_prompt_parts.append("6. Answer with the option's letter directly.\n")
            integration_prompt_parts.append("\nFinal Integrated Answer:")
            integration_prompt = "\n".join(integration_prompt_parts)
            self.logger.info("Judge 正在整合专家分析和批评意见...")
            final_result = self.base_model.generate_response(video_path, integration_prompt, video_frames=video_frames)
            final_response = final_result["text"]
            return {"text": final_response, "critiques": critique_results,
                    "evidence_regions": [claim['region'] for claim in uncertain_claims if 'region' in claim]}
        else:
            self.logger.info("无可用批评专家或不确定议题，保持当前回答")
            return {"text": current_response, "critiques": [], "evidence_regions": []}

    def process_with_experts(self, video_path: str = None, instruction: str = "", output=None, options=None,
                             choice_answer=None, video_frames=None):
        """
        使用专家处理视频和指令 (增强本地视频模式)

        Args:
            video_path: 视频文件路径 (str)
            instruction: 指令
            output: 参考输出（训练/评估时用）
            options: 选项（选择题时用）
            choice_answer: 选择题参考答案（训练/评估时用）

        Returns:
            dict: 处理结果
        """
        if not video_path and not video_frames:
            self.logger.error("没有提供视频输入。")
            return {
                "error": "No video input provided.",
                "final_response": "[视频输入错误]",
                "agent_uncertainties": [],
                "agent_responses": [],
                "initial_response": "[视频输入错误]",
                "all_expert_raw_results": [],
                "task_complexity": 1.0
            }
        agent_responses = []
        agent_uncertainties = []
        all_expert_raw_results = []
        self.logger.info(f"开始调用 {len(self.experts)} 个专家进行初始分析...")
        start_time = time.time()

        for i, expert in enumerate(self.experts):
            if not expert.get('enabled', True):
                agent_responses.append({"role": expert['name'], "response": "[专家已禁用]"})
                agent_uncertainties.append(1.0)
                all_expert_raw_results.append(None)
                continue

            expert_start_time = time.time()
            expert_prompt = self.create_expert_prompt(expert, instruction, options)
            expert_prompt += "\n\nImportant: For key claims in your analysis, please mention which parts of the video support your conclusions. When possible, describe the specific video frames or features that provide evidence."
            expert_result = self.base_model.generate_response(video_path, expert_prompt, video_frames=video_frames)
            expert_response_text = expert_result["text"]
            all_expert_raw_results.append(expert_result)
            expert_uncertainty = self.base_model.estimate_uncertainty(
                expert_response_text,
                expert_result.get("generated_token_ids"),
                expert_result.get("scores")
            )
            agent_responses.append({"role": expert['name'], "response": expert_response_text})
            agent_uncertainties.append(expert_uncertainty)
            expert_end_time = time.time()
            self.logger.info(
                f"专家 '{expert['display_name']}' 完成初始分析, 不确定性: {expert_uncertainty:.4f}, 耗时: {expert_end_time - expert_start_time:.2f}s")
        total_expert_time = time.time() - start_time
        self.logger.info(f"所有专家初始分析完成，总耗时: {total_expert_time:.2f}s")

        integration_prompt_parts = [
            "You are a Judge tasked with integrating analyses from multiple specialists who have examined the same video.",
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
                integration_prompt_parts.append(
                    f"--- Expert: {expert_name} (Uncertainty: {uncertainty:.3f}) ---\n{resp_info['response']}\n")
            else:
                integration_prompt_parts.append(f"--- Expert: {self.experts[i]['display_name']} (Disabled) ---\n\n")

        integration_prompt_parts.append(
            "\nInstructions for Integration:\n"
            "1. Synthesize a comprehensive answer that incorporates insights from all experts.\n"
            "2. Give more weight to experts with lower uncertainty scores.\n"
            "3. When experts disagree, explain the different perspectives and provide your best judgment.\n"
            "4. If options were provided, your final answer MUST select one of the options clearly.\n"
            "5. Include references to specific visual evidence that supports key claims.\n"
            "\nIntegrated Answer:"
        )

        integration_prompt = "\n".join(integration_prompt_parts)

        self.logger.info("开始初始整合...")
        integration_start_time = time.time()
        integration_result = self.base_model.generate_response(video_path, integration_prompt, video_frames=video_frames)
        initial_response = integration_result["text"]
        integration_end_time = time.time()
        self.logger.info(f"初始整合完成，耗时: {integration_end_time - integration_start_time:.2f}s")

        result = {
            "agent_uncertainties": agent_uncertainties,
            "agent_responses": agent_responses,
            "initial_response": initial_response,
            "all_expert_raw_results": all_expert_raw_results,
        }

        if self.experts_settings.get('use_debate', True) and self.critic_experts:
            debate_rounds = self.experts_settings.get('debate_rounds', 1)
            self.logger.info(f"开始 {debate_rounds} 轮辩论流程...")

            current_response = initial_response
            previous_critiques = []
            evidence_regions = []

            for i in range(debate_rounds):
                self.logger.info(f"--- 辩论轮次 {i + 1}/{debate_rounds} ---")
                round_start_time = time.time()

                debate_output = self.run_enhanced_debate_process(
                    video_path=video_path,
                    instruction=instruction,
                    options=options,
                    current_response=current_response,
                    previous_critiques=previous_critiques,
                    reference_output=output,
                    is_final_round=(i == debate_rounds - 1),
                    video_frames=video_frames
                )

                current_response = debate_output["text"]
                previous_critiques = debate_output.get("critiques", [])
                evidence_regions.extend(debate_output.get("evidence_regions", []))

                round_end_time = time.time()
                self.logger.info(f"辩论轮次 {i + 1} 完成, 耗时: {round_end_time - round_start_time:.2f}s")

            final_response = current_response
            result["debate_result"] = "[增强辩论流程已执行]"
            result["evidence_regions"] = evidence_regions
        else:
            final_response = initial_response
            if not self.critic_experts and self.experts_settings.get('use_debate', True):
                self.logger.warning("辩论已启用，但未配置批评专家，跳过辩论流程")
            else:
                self.logger.info("辩论功能未启用，使用初始整合结果")

        result["final_response"] = final_response

        avg_uncertainty = sum(agent_uncertainties) / len(agent_uncertainties) if agent_uncertainties else 1.0
        result["task_complexity"] = avg_uncertainty
        return result