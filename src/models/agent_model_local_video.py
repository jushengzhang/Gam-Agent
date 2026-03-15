import torch
import numpy as np
import json
import os
import base64
import io
import re
import logging
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch.nn.functional as F
import sys
import time
import re
import math
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel

# === New Helper Functions from Snippet ===
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = int(image_size * target_aspect_ratio[0]) # Ensure integer
    target_height = int(image_size * target_aspect_ratio[1]) # Ensure integer
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * image_size, 
            (i // target_aspect_ratio[0]) * image_size, 
            ((i % target_aspect_ratio[0]) + 1) * image_size, 
            ((i // target_aspect_ratio[0]) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        
    assert len(processed_images) == blocks
    if use_thumbnail and blocks != 1: # Ensure thumbnail is only added when splitting occurs
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    # Handle edge case where start_idx >= end_idx
    if start_idx >= end_idx:
        # Use a small segment around the middle if bounds are invalid or too close
        mid_idx = (start_idx + end_idx) // 2
        # Fallback to taking num_segments frames around the middle or start
        indices = np.linspace(max(0, mid_idx - num_segments // 2), min(max_frame, mid_idx + num_segments // 2), num_segments, dtype=int)
        return indices

    seg_size = float(end_idx - start_idx) / num_segments
    # Ensure indices are within valid range [0, max_frame]
    frame_indices = np.array([min(max_frame, max(0, int(start_idx + (seg_size / 2) + np.round(seg_size * idx)))) for idx in range(num_segments)])
    return frame_indices


def get_num_frames_by_duration(duration):
    local_num_frames = 4
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments

    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames) # Original snippet had max(128,...)? Keep it for now. Maybe min(128,...) intended? Let's assume max is correct.
    return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    except Exception as e:
        logging.error(f"无法使用decord打开视频文件 {video_path}: {e}")
        raise ValueError(f"无法打开视频文件: {video_path}") from e

    max_frame = len(vr) - 1
    if max_frame < 0: # Check if video is empty or invalid
        raise ValueError(f"视频文件 {video_path} 无效或没有帧。")
        
    fps = float(vr.get_avg_fps())
    if fps <= 0: # Handle case where FPS is invalid
        logging.warning(f"视频 {video_path} 的 FPS 无效 ({fps}), 将使用默认值 30。")
        fps = 30.0 

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    
    if get_frame_by_duration:
        duration = max_frame / fps if fps > 0 else 0
        num_segments = get_num_frames_by_duration(duration)
        
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    if len(frame_indices) == 0:
         raise ValueError(f"无法从视频 {video_path} 计算有效的帧索引。")

    logging.info(f"将从视频 {video_path} 加载 {len(frame_indices)} 帧 (索引: {frame_indices[:5]}...{frame_indices[-5:]})")

    for frame_index in frame_indices:
        try:
            img_array = vr[frame_index].asnumpy()
        except IndexError:
             logging.warning(f"无法获取视频 {video_path} 的帧索引 {frame_index} (max_frame: {max_frame})。跳过此帧。")
             continue # Skip if index is out of bounds for some reason

        img = Image.fromarray(img_array).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0]) # number of patches for this frame
        pixel_values_list.append(pixel_values)
        
    if not pixel_values_list:
         raise ValueError(f"未能从视频 {video_path} 加载任何有效的帧。")

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# === End of New Helper Functions ===

class AgentModel:
    """视频语言代理模型基类（本地模式）"""
    
    def __init__(self, config):
        """
        初始化代理模型 (本地模式)
        
        Args:
            config: 模型配置 (包含 model_path, num_segments 等)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.model_path = config.get("model_path", "OpenGVLab/InternVideo2_5_Chat_8B") # New model path from snippet
        self.num_segments = config.get("num_segments", 128) # Default from snippet
        self.input_size = config.get("input_size", 448) # Default from snippet helper functions
        self.max_num_patches_per_frame = config.get("max_num_patches_per_frame", 1) # Default from snippet load_video call

        # Load local model and tokenizer/processor using new logic
        self.logger.info(f"正在加载本地模型: {self.model_path} 到设备 {self.device}")
        try:
            # Use new loading logic from snippet
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            # Load in bfloat16 directly if supported, else use float16 and convert later if needed
            model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True, # Keep this if useful
                trust_remote_code=True
            ).eval().to(self.device)

            # No processor needed based on snippet logic

            self.logger.info(f"模型 {self.model_path} 加载成功，使用数据类型 {model_dtype}.")
        except Exception as e:
            self.logger.error(f"加载本地模型 {self.model_path} 失败: {e}")
            raise e

        # Ensure tokenizer has pad_token if missing (common issue)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Snippet model might not need config update, depends on base model type
            # self.model.config.pad_token_id = self.model.config.eos_token_id
            self.logger.info("Tokenizer pad_token 设置为 eos_token.")
            
        # Define generation config (can be overridden)
        self.generation_config = config.get("generation_config", dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1, # Snippet uses 0.1, default was likely higher
            num_beams=1 # Snippet uses 1
        ))
        self.logger.info(f"使用的生成配置: {self.generation_config}")


    def process_video(self, video_path):
        """
        处理视频文件，使用新的 load_video 函数

        Args:
            video_path: 视频文件的路径

        Returns:
            tuple: (pixel_values: torch.Tensor, num_patches_list: list[int])
                   pixel_values 是处理后的视频帧张量
                   num_patches_list 是每帧对应的 patch 数量
        """
        self.logger.info(f"开始使用 load_video 处理视频: {video_path}")
        try:
            # Use the new load_video function with parameters from config
            pixel_values, num_patches_list = load_video(
                video_path,
                input_size=self.input_size,
                max_num=self.max_num_patches_per_frame,
                num_segments=self.num_segments,
                get_frame_by_duration=False # Matches snippet direct call
            )
            # self.logger.info(f"视频处理完成。Pixel values shape: {pixel_values.shape}, Num patches list length: {len(num_patches_list)}")
            return pixel_values, num_patches_list
        except Exception as e:
            self.logger.error(f"使用 load_video 处理视频 {video_path} 时出错: {e}")
            # Re-raise to be caught by the caller (ExpertAgentModel)
            raise


    def estimate_uncertainty(self, response, generated_token_ids=None, scores=None):
        """
        量化智能体响应的不确定性 (本地模式简化版) - Scores might not be available from model.chat

        Args:
            response: 智能体生成的文本响应
            generated_token_ids: 可选，模型生成的 token ID 列表 (model.chat might not return these easily)
            scores: 可选，来自模型 generate 方法的每个 token 的分数 (logits) (model.chat might not return these easily)

        Returns:
            float: 不确定性评分（0-1之间，1表示最不确定）
        """
        # Strategy 1: Use scores (logits) if available - LIKELY NOT AVAILABLE from model.chat
        if scores is not None and generated_token_ids is not None:
            # ... (previous logic, but likely unused now)
            self.logger.warning("收到 scores/token_ids，但 model.chat 可能不提供，不确定性估计可能不准确。")
            # Fall through to text-based method

        # Strategy 2: Fallback to text-based features (Now the primary method)
        self.logger.debug("使用文本特征估计不确定性 (因为 model.chat 可能不提供 scores)")

        response_lower = response.lower() if response else "" # Handle None response
        words = [w for w in re.split(r'[\s,.!?;:()\[\]{}"\']', response_lower) if w]
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
        if '?' in response_lower[:-1]: # Check for question marks not at the very end
            raw_weighted_score += 2

        score_density = raw_weighted_score / num_words if num_words > 0 else 0
        # Adjust sigmoid parameters if needed based on observed scores
        k = 60 # Steepness
        offset = 0.04 # Midpoint shift
        exponent = -k * (score_density - offset)
        # Use try-except for potential math overflow with large exponents
        try:
             uncertainty = 1 / (1 + math.exp(exponent))
        except OverflowError:
             uncertainty = 1.0 if exponent < 0 else 0.0 # Approximate limit

        self.logger.info(f"基于文本特征计算的不确定性: {uncertainty:.4f}，原始加权分数: {raw_weighted_score}，词数: {num_words}，密度: {score_density:.6f}")
        return uncertainty


    def generate_response(self, pixel_values, num_patches_list, prompt, history=None):
        """
        使用本地模型 (model.chat) 生成回复

        Args:
            pixel_values (torch.Tensor): 来自 process_video 的视频帧张量
            num_patches_list (list[int]): 来自 process_video 的每帧 patch 数量列表
            prompt (str): 提示文本 (不包含 frame 前缀，将在内部添加)
            history (list | None): 对话历史 (用于多轮对话)

        Returns:
            dict: 包含生成文本和对话历史
                  {'text': str, 'history': list | None}
                  Note: Scores/token_ids are not returned by model.chat in the snippet example
        """
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(num_patches_list, list):
             error_msg = f"generate_response 需要 pixel_values (Tensor) 和 num_patches_list (list)，但收到 {type(pixel_values)}, {type(num_patches_list)}"
             self.logger.error(error_msg)
             return {"text": f"内部错误: {error_msg}", "history": history}

        # Construct the video prefix as shown in the snippet
        # Ensure num_patches_list is not empty
        if not num_patches_list:
             self.logger.error("num_patches_list 为空，无法构建视频前缀。")
             return {"text": "内部错误: 视频处理可能失败。", "history": history}
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        full_prompt = video_prefix + prompt

        # self.logger.info(f"使用 model.chat 生成响应，帧数: {len(num_patches_list)}, 提示 (前缀后): '{prompt[:100]}...'")
        # self.logger.debug(f"完整提示 (带前缀): {full_prompt[:200]}...") # Debug log

        try:
            # Ensure pixel_values are on the correct device and dtype
            processed_pixel_values = pixel_values.to(self.model.dtype).to(self.device)

            # Use model.chat as in the snippet
            with torch.no_grad():
                # Note: return_history=True to manage multi-turn state
                output_text, new_history = self.model.chat(
                    self.tokenizer,
                    processed_pixel_values,
                    full_prompt, # Use the prompt with the video prefix
                    self.generation_config,
                    num_patches_list=num_patches_list,
                    history=history, # Pass existing history for multi-turn
                    return_history=True
                )

            self.logger.info(f"model.chat 生成响应: {output_text[:200]}...")

            return {
                "text": output_text.strip() if output_text else "", # Ensure text is returned, handle None
                "history": new_history # Return updated history for multi-turn calls
                # Scores and token IDs are not available from this .chat call based on snippet
            }

        except Exception as e:
            self.logger.exception(f"model.chat 推理过程中出现错误: {e}") # Use logger.exception
            return {"text": f"处理错误: {str(e)}", "history": history}


class ExpertAgentModel:
    """专家代理模型，管理多个不同专长的代理模型"""
    
    def __init__(self, model_config, experts_config, api_config=None):
        """
        初始化专家代理模型

        Args:
            model_config: 模型配置 (包含 model_path, num_segments 等)
            experts_config: 专家配置
            api_config: API配置（可选, 已弃用 for local model）
        """
        self.model_config = model_config
        self.experts_config = experts_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        # Initialize the base model (now local, using AgentModel updated init)
        self.base_model = AgentModel(model_config) # Pass model_config directly

        self.load_experts_config()

        # Initialize MAB (unchanged conceptually)
        self.num_arms = len(self.experts) + len(self.combinations)
        self.arms = [{'total_reward': 0, 'count': 0} for _ in range(self.num_arms)]
        self.epsilon = self.experts_settings.get('epsilon', 0.1)
        self.high_uncertainty_threshold = self.experts_settings.get('high_uncertainty_threshold', 0.6)


    def load_experts_config(self, use_default=False):
        """加载专家配置文件 (逻辑不变, 但确保路径相对于项目根目录)"""
        if use_default:
            self._load_default_experts_config()
            return

        config_file_path_rel = self.experts_config.get('config_file') # Relative path from config
        if not config_file_path_rel:
            self.logger.warning("未指定专家配置文件路径，将使用默认专家配置")
            self._load_default_experts_config()
            return

        # Construct path relative to the project root (assuming GAM-Agent is root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir)) # GAM-Agent/
        # Configs are usually in GAM-Agent/configs/
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
            self.combinations = experts_data.get('combinations', []) # Combinations might not be usable without underlying expert scores? Review later.
            self.critic_experts = experts_data.get('critic_experts', [])
            self.debate_settings = experts_data.get('debate_settings', {})

            self._validate_experts_config()

            # Strategy types might need update if combinations are harder to implement now
            self.strategy_types = [expert['display_name'] for expert in self.experts] # + \
                                #   [combo['display_name'] for combo in self.combinations]

            self.logger.info(f"已加载 {len(self.experts)} 个专家、{len(self.combinations)} 个组合策略和 {len(self.critic_experts)} 个评论专家 from {config_file_path_abs}")

        except ImportError:
             self.logger.error("PyYAML 未安装，无法加载 YAML 配置文件。请运行 'pip install PyYAML'。回退到默认配置。")
             self._load_default_experts_config()
        except Exception as e:
            self.logger.error(f"加载专家配置文件 {config_file_path_abs} 失败: {e}")
            self.logger.warning("回退到默认配置")
            self._load_default_experts_config()
    
    # _load_default_experts_config remains the same
    def _load_default_experts_config(self):
        """Load default expert configuration"""
        # Use built-in default configuration
        self.experts_settings = {'epsilon': 0.1, 'high_uncertainty_threshold': 0.6, 'use_debate': True, 'debate_rounds': 1} # Example settings
        self.experts = [
             {
                'id': 0, 'name': "scene_understanding", 'display_name': "Scene Understanding", 'weight': 1.0, 'enabled': True,
                'keywords': ["describe", "what kind of", "looks like", "scene", "environment", "background"],
                'prompt_template': "Please act as a scene understanding expert and analyze the scene, environment, and background in this video in detail. Key points to focus on: venue setup, environmental features, surrounding objects, lighting conditions, and overall atmosphere. Original question: {instruction}"
            },
             {
                'id': 1, 'name': "action_analysis", 'display_name': "Action Analysis", 'weight': 1.0, 'enabled': True,
                'keywords': ["what to do", "action", "what happened", "how to do"],
                'prompt_template': "Please act as an action analysis expert and analyze all actions and movements in this video in detail. Key points to focus on: character actions, object movements, action sequences, action speed, and action purpose. Original question: {instruction}"
            },
             {
                'id': 2, 'name': "semantic_understanding", 'display_name': "Semantic Understanding", 'weight': 1.0, 'enabled': True,
                'keywords': ["why", "meaning", "purpose", "significance", "express"],
                'prompt_template': "Please act as a semantic understanding expert and analyze the meaning and information conveyed in this video in detail. Key points to focus on: video theme, underlying messages, cultural context, emotional expression, and video purpose. Original question: {instruction}"
            }
        ]
        # Combinations are kept but their execution logic needs review in process_with_experts
        self.combinations = [
             {
                'id': 3, 'name': "scene_action", 'display_name': "Scene+Action", 'experts': [0, 1], 'weight': 1.0, 'enabled': True,
                'prompt_template': "Please integrate the following two experts' analysis of the video to provide a comprehensive video understanding:\n\nScene Understanding Expert: {expert_0_response}\n\nAction Analysis Expert: {expert_1_response}\n\nPlease provide a complete and coherent video analysis that synthesizes the insights from both experts. Original question: {instruction}"
            },
             {
                'id': 4, 'name': "action_semantic", 'display_name': "Action+Semantic", 'experts': [1, 2], 'weight': 1.0, 'enabled': True,
                'prompt_template': "Please integrate the following two experts' analysis of the video to provide a comprehensive video understanding:\n\nAction Analysis Expert: {expert_1_response}\n\nSemantic Understanding Expert: {expert_2_response}\n\nPlease provide a complete and coherent video analysis that synthesizes the insights from both experts. Original question: {instruction}"
            },
             {
                'id': 5, 'name': "scene_semantic", 'display_name': "Scene+Semantic", 'experts': [0, 2], 'weight': 1.0, 'enabled': True,
                'prompt_template': "Please integrate the following two experts' analysis of the video to provide a comprehensive video understanding:\n\nScene Understanding Expert: {expert_0_response}\n\nSemantic Understanding Expert: {expert_2_response}\n\nPlease provide a complete and coherent video analysis that synthesizes the insights from both experts. Original question: {instruction}"
            },
             {
                'id': 6, 'name': "full_ensemble", 'display_name': "Full Integration", 'experts': [0, 1, 2], 'weight': 1.2, 'enabled': True, # This ID might conflict if experts have non-contiguous IDs
                'prompt_template': "Please integrate the following three experts' analysis of the same video to provide a comprehensive video understanding:\n\nScene Understanding Expert: {expert_0_response}\n\nAction Analysis Expert: {expert_1_response}\n\nSemantic Understanding Expert: {expert_2_response}\n\nOriginal question: {instruction}\n\nPlease provide a comprehensive and coherent video analysis that integrates the insights from all three experts and directly answers the original question."
             }
        ]
        self.critic_experts = [
             {
                'id': 7, 'name': "fact_checker", 'display_name': "Fact Checker", 'enabled': True,
                'critique_template': "Please act as a fact-checking expert and evaluate the factual accuracy in the following analysis:\n\n{response}\n\nPlease point out any potential factual errors or inaccuracies in the analysis and provide improvement suggestions."
            },
             {
                'id': 8, 'name': "logic_analyzer", 'display_name': "Logic Analyzer", 'enabled': True,
                'critique_template': "Please act as a logic analysis expert and evaluate the logical coherence and reasoning quality in the following analysis:\n\n{response}\n\nPlease point out any logical gaps, inconsistencies, or insufficient reasoning in the analysis and provide improvement suggestions."
            }
        ]
        self.debate_settings = {
            'quality_metrics': [
                {'name': 'relevance', 'weight': 0.3, 'description': 'Relevance to the question'},
                {'name': 'factuality', 'weight': 0.3, 'description': 'Factual accuracy of the answer'},
                {'name': 'completeness', 'weight': 0.4, 'description': 'Completeness of the answer'}
            ],
            'filter_rules': { 'min_critique_length': 10, 'max_critique_similarity': 0.9, 'required_suggestion_count': 0 }
        }

        self.strategy_types = [expert['display_name'] for expert in self.experts] # + \
                            #   [combo['display_name'] for combo in self.combinations]
        self.logger.info(f"使用默认配置: {len(self.experts)} 个专家, {len(self.combinations)} 个组合策略")


    # _validate_experts_config remains largely the same
    def _validate_experts_config(self):
        """验证专家配置的有效性"""
        if not self.experts:
             self.logger.warning("专家列表为空!")
             # Allow empty list, but combination logic will fail later
             # return 

        expert_ids = set()
        max_id = -1
        if self.experts: # Check if list is not empty before iterating
            expert_ids = {expert['id'] for expert in self.experts}
            for i, expert in enumerate(self.experts):
                # Simplified validation: just check required fields exist
                if 'id' not in expert or 'name' not in expert or 'display_name' not in expert or 'prompt_template' not in expert:
                    raise ValueError(f"专家索引 {i} 缺少必要的字段 (id, name, display_name, prompt_template)")
                if expert['id'] > max_id:
                    max_id = expert['id']
        else:
             self.logger.warning("专家列表为空，跳过专家验证。")


        combo_start_id = max_id + 1
        if self.combinations: # Check if list is not empty
            for i, combo in enumerate(self.combinations):
                if 'id' not in combo or 'name' not in combo or 'display_name' not in combo or 'experts' not in combo or 'prompt_template' not in combo:
                    raise ValueError(f"组合策略索引 {i} 缺少必要的字段 (id, name, display_name, experts, prompt_template)")
                # Optional: Check if combo ID follows expert IDs
                # combo_id = combo.get('id')
                # if combo_id is None or combo_id < combo_start_id:
                #     self.logger.warning(f"组合策略 ID {combo_id} 可能与专家 ID 冲突或不连续")

                if not isinstance(combo.get('experts'), list) or not combo.get('experts'):
                    raise ValueError(f"组合策略索引 {i} (ID: {combo.get('id')}) 的 'experts' 必须是非空列表")
                
                for expert_id in combo['experts']:
                    if expert_id not in expert_ids:
                        raise ValueError(f"组合策略 {combo['name']} (ID: {combo.get('id')}) 引用了不存在的专家ID {expert_id}")
        else:
             self.logger.info("组合策略列表为空，跳过组合验证。")


        # Basic validation for critic experts and debate settings
        if self.critic_experts:
            for i, critic in enumerate(self.critic_experts):
                 if 'name' not in critic or 'critique_template' not in critic:
                      raise ValueError(f"评论专家索引 {i} 缺少 'name' 或 'critique_template'")
        else:
            self.logger.info("评论专家列表为空。")


        if not isinstance(self.debate_settings, dict):
             raise ValueError("'debate_settings' 必须是一个字典")

        self.logger.info("专家配置验证通过.")


    def create_expert_prompt(self, expert, instruction, options=None):
        """创建专家提示 (逻辑不变)"""
        template = expert.get('prompt_template', '{instruction}')
        template_instruction = template.replace('{instruction}', instruction)

        if '{options}' in template:
            if options:
                options_str = ', '.join(map(str, options)) if isinstance(options, list) else str(options)
                template_instruction = template_instruction.replace('{options}', options_str)
                self.logger.debug(f"已将选项插入提示模板: {options_str}")
            else:
                template_instruction = template_instruction.replace('{options}', '[选项未提供]')
                self.logger.warning("提示模板需要选项，但未提供。")
        return template_instruction


    def process_with_experts(self, video_path, instruction, output=None, options=None, choice_answer=None):
        """
        使用专家处理视频和指令 (本地模式, 使用新模型和加载方式)

        Args:
            video_path: 视频文件路径 <--- CHANGED FROM video_frames
            instruction: 指令
            output: 参考输出（训练/评估时用）
            options: 选项（选择题时用）
            choice_answer: 选择题参考答案（训练/评估时用）

        Returns:
            dict: 处理结果
        """
        # 1. Process Video Path -> pixel_values, num_patches_list
        try:
            pixel_values, num_patches_list = self.base_model.process_video(video_path)
        except Exception as e:
             self.logger.error(f"处理视频文件 {video_path} 时出错: {e}")
             # Return an error structure consistent with previous version
             return {
                 "error": f"Video processing failed: {e}",
                 "final_response": "[视频处理错误]",
                 "agent_uncertainties": [1.0] * len(self.experts) if self.experts else [1.0], # Handle empty experts list
                 "agent_responses": [{"role": exp.get('name', f'expert_{i}'), "response": "[视频处理错误]"} for i, exp in enumerate(self.experts)] if self.experts else [],
                 "task_complexity": 1.0, # Max complexity on error
             }

        # 2. Get responses from all enabled experts (using model.chat)
        agent_responses = []
        agent_uncertainties = []
        all_expert_raw_results = [] # Store full results (text + history)

        self.logger.info(f"开始并行或串行调用 {len(self.experts)} 个专家...")
        start_time = time.time()
        expert_chat_history = None # Maintain history for potential multi-turn expert calls if needed later
        
        all_expert_results = []

        for i, expert in enumerate(self.experts):
            if not expert.get('enabled', True):
                agent_responses.append({"role": expert['name'], "response": "[专家已禁用]"})
                agent_uncertainties.append(1.0)
                all_expert_results.append(None)
                continue

            expert_start_time = time.time()
            expert_prompt = self.create_expert_prompt(expert, instruction, options)

            # Get expert response using the local model.chat
            # For initial expert calls, history is typically None unless reusing previous state
            expert_result = self.base_model.generate_response(
                pixel_values,
                num_patches_list,
                expert_prompt,
                history=None # Start fresh for each expert in this flow
            )
            expert_response_text = expert_result["text"]
            all_expert_results.append(expert_result) # Store full result dict


            # Estimate uncertainty based on text
            expert_uncertainty = self.base_model.estimate_uncertainty(expert_response_text) # No scores/ids available

            agent_responses.append({"role": expert['name'], "response": expert_response_text})
            agent_uncertainties.append(expert_uncertainty)
            expert_end_time = time.time()
            self.logger.info(f"专家 '{expert['display_name']}' 完成, 不确定性: {expert_uncertainty:.4f}, 耗时: {expert_end_time - expert_start_time:.2f}s")

        total_expert_time = time.time() - start_time
        self.logger.info(f"所有专家响应生成完毕，总耗时: {total_expert_time:.2f}s")
        self.logger.info(f"专家不确定性评分: {[f'{u:.4f}' for u in agent_uncertainties]}")

        # 3. Integrate expert responses (Compulsory integration using model.chat)
        # Build integration prompt (same logic as before)
        integration_prompt_parts = [
             "You are a VQA answer integration expert. Your goal is to synthesize the answers from multiple experts, considering their uncertainty, to provide the best possible final answer to the original question.",
             f"Original Question: {instruction}"
        ]
        if options:
             options_str = ', '.join(map(str, options)) if isinstance(options, list) else str(options)
             integration_prompt_parts.append(f"Available Options: {options_str}")

        integration_prompt_parts.append("\nExpert Analyses:")
        for i, resp_info in enumerate(agent_responses):
             # Need to handle case where expert was disabled (resp_info['response'] is placeholder)
             if resp_info['response'] != "[专家已禁用]":
                 expert_name = self.experts[i]['display_name'] # Get display name
                 uncertainty = agent_uncertainties[i]
                 integration_prompt_parts.append(f"--- Expert: {expert_name} (Uncertainty: {uncertainty:.3f}) ---\n{resp_info['response']}\n")
             else:
                 integration_prompt_parts.append(f"--- Expert: {self.experts[i]['display_name']} (Disabled) ---\n")


        integration_prompt_parts.append(
             "\nInstructions for Integration:\n"
             "1. Carefully review the original question and the analyses from all enabled experts.\n"
             "2. Pay attention to the uncertainty scores; give more weight to confident experts, but don't ignore insights from uncertain ones if they seem plausible.\n"
             "3. Identify areas of agreement and disagreement among the experts.\n"
             "4. Synthesize the information into a single, coherent, and comprehensive final answer that directly addresses the original question.\n"
             "5. If options were provided, your final answer MUST be one of the available options. State the chosen option clearly."
             # "6. Wrap your final chosen option within curly braces, like {Option A}." # Removed wrapping for now, simpler parsing
             "\nFinal Integrated Answer:"
        )
        integration_prompt_text = "\n".join(integration_prompt_parts)
        
        integration_prompt_text_format = (
            "Your answer format is as follows:\n"
            "Expert 1 final answer is: ....... with uncertainty XX, Expert 2 final answer is: ....... with uncertainty XX, Expert 3 final answer is: ....... with uncertainty XX\n"
            "Combining my judgment and the votes for the experts' choices, I think the answer should be ....... "
            "(wrap the specific option as answer in {}, your answer must come from options)"
             "\nFinal Integrated Answer:"
        )
        integration_prompt_text = integration_prompt_text + integration_prompt_text_format

        self.logger.info("开始整合专家回答...")
        integration_start_time = time.time()
        # Call base model for integration (using model.chat, history=None)
        integration_result = self.base_model.generate_response(
            pixel_values,
            num_patches_list,
            integration_prompt_text,
            history=None # Start integration fresh
        )
        initial_response = integration_result["text"]
        integration_history = integration_result["history"] # Store history from integration step
        integration_end_time = time.time()
        self.logger.info(f"整合完成，耗时: {integration_end_time - integration_start_time:.2f}s. 初始整合结果: {initial_response[:200]}...")

        # Prepare result dictionary
        result = {
            "agent_uncertainties": agent_uncertainties,
            "agent_responses": agent_responses,
            "initial_response": initial_response,
            "all_expert_raw_results": all_expert_results, # Raw results (text + history)
        }

        # 4. Optional Iterative Debate Process (using model.chat)
        # Ensure debate runs by default if not specified in config
        if self.experts_settings.get('use_debate', True) and self.critic_experts:
            debate_rounds = self.experts_settings.get('debate_rounds', 1) # Get rounds from config
            self.logger.info(f"开始 {debate_rounds} 轮迭代辩论...")
            current_response = initial_response
            current_history = integration_history
            previous_critiques = [] # Start with no previous critiques
            debate_start_time = time.time()

            for i in range(debate_rounds):
                self.logger.info(f"--- 辩论轮次 {i+1}/{debate_rounds} ---")
                round_start_time = time.time()

                debate_output = self.run_debate_process_new(
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    instruction=instruction,
                    options=options,
                    current_response=current_response,
                    current_history=current_history,
                    previous_critiques=previous_critiques, # Pass previous critiques
                    reference_output=output
                )

                # Update response, history, and critiques for the next round
                current_response = debate_output["text"]
                current_history = debate_output["history"]
                previous_critiques = debate_output["critiques"] # Store current critiques for next round

                round_end_time = time.time()
                self.logger.info(f"辩论轮次 {i+1} 完成, 耗时: {round_end_time - round_start_time:.2f}s. 当前回答: {current_response[:200]}...")
                # Optional: Log critiques generated in this round if needed for debugging
                # self.logger.debug(f"本轮生成的评论: {previous_critiques}")

            final_response = current_response
            result["debate_result"] = "[辩论流程已执行]"
            debate_end_time = time.time()
            self.logger.info(f"辩论流程完成，总耗时: {debate_end_time - debate_start_time:.2f}s")
        else:
            final_response = initial_response
            if not self.critic_experts and self.experts_settings.get('use_debate', False):
                 self.logger.warning("辩论已启用，但未配置评论专家，将跳过辩论。")
            else:
                 self.logger.info("辩论功能未启用或无评论专家，使用初始整合结果。")

        result["final_response"] = final_response

        # Calculate task complexity (example heuristic - unchanged)
        avg_uncertainty = sum(agent_uncertainties) / len(agent_uncertainties) if agent_uncertainties else 1.0
        result["task_complexity"] = avg_uncertainty

        return result


    def run_debate_process_new(self, pixel_values, num_patches_list, instruction, options, current_response, current_history, previous_critiques=None, reference_output=None):
        """
        执行一轮“辩论”过程（新逻辑：让原始专家基于上一轮回答进行修正）(本地视频模式)

        Args:
            pixel_values (torch.Tensor): 视频帧张量
            num_patches_list (list[int]): 每帧 patch 数量列表
            instruction (str): 原始用户问题
            options (list | None): 选项
            current_response (str): 上一轮整合后的回答
            current_history (list | None): (被忽略，因为专家修正不依赖上一轮历史)
            previous_critiques (list[dict] | None): (被忽略)
            reference_output (str | None): (被忽略)

        Returns:
            dict: {'text': str, 'history': list | None, 'critiques': list[dict]} 
                  包含最终修正并整合后的回答、更新后的历史以及空评论列表
        """
        self.logger.info("进入新辩论流程（专家修正模式 for Video）...")
        self.logger.debug(f"基于上一轮回答进行修正: {current_response[:100]}...")

        # 1. 构建新的指令，要求专家参考之前的回答进行修正
        instruction_new = (
            f"{instruction}\n\n" 
            f"This was the previous integrated answer: "
            f"\"{current_response}\"\n\n" 
            f"Please review this previous answer and provide your refined analysis or answer based on the video content and original question."
        )
        self.logger.debug(f"构建的专家修正指令 (部分): {instruction_new[:200]}...")

        # 2. 让所有启用的原始专家再次生成回答
        expert_responses_new = []
        expert_uncertainties_new = []
        all_expert_results_new = [] # Store full results (text + history)
        expert_call_start_time = time.time()

        for i, expert in enumerate(self.experts):
            if not expert.get('enabled', True):
                expert_responses_new.append({"role": expert['name'], "response": "[专家已禁用]"})
                expert_uncertainties_new.append(1.0)
                all_expert_results_new.append(None)
                continue

            expert_prompt = self.create_expert_prompt(expert, instruction_new, options) # Use instruction_new
            self.logger.debug(f"为专家 '{expert['display_name']}' 调用基础模型进行修正...")
            # History is None here, as each expert reviews the previous integrated answer freshly
            expert_result = self.base_model.generate_response(
                pixel_values,
                num_patches_list,
                expert_prompt,
                history=None 
            )
            expert_response_text = expert_result["text"]
            all_expert_results_new.append(expert_result)

            expert_uncertainty = self.base_model.estimate_uncertainty(expert_response_text)
            expert_responses_new.append({"role": expert['name'], "response": expert_response_text})
            expert_uncertainties_new.append(expert_uncertainty)
            self.logger.debug(f"专家 '{expert['display_name']}' 修正完成，新不确定性: {expert_uncertainty:.4f}")

        expert_call_end_time = time.time()
        self.logger.info(f"所有专家修正响应生成完毕，耗时: {expert_call_end_time - expert_call_start_time:.2f}s")

        # 3. 再次整合这些修正后的专家回答
        integration_prompt_parts = [
             "You are an expert at integrating REFINED video analyses from multiple specialists. Synthesize their updated insights to provide the best possible final answer to the original question.",
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
                 integration_prompt_parts.append(f"--- Expert: {expert_name} (Uncertainty: {uncertainty:.3f}) ---\n{resp_info['response']}\n")
             else:
                 integration_prompt_parts.append(f"--- Expert: {self.experts[i]['display_name']} (Disabled) ---\n\n")

        integration_prompt_parts.append(
             "\nInstructions for Final Integration:\n"
             "1. Review the original question, the previous integrated answer, and the refined analyses.\n"
             "2. Synthesize the refined information into a single, coherent final answer.\n"
             "3. If options were provided, your final answer MUST be one of the options."
             "\nFinal Integrated Answer:"
        )
        integration_prompt_text_final = "\n".join(integration_prompt_parts)
        #加上格式控制
        integration_prompt_text_format = (
            "Your answer format is as follows:\n"
            "Expert 1 final answer is: ....... with uncertainty XX, Expert 2 final answer is: ....... with uncertainty XX, Expert 3 final answer is: ....... with uncertainty XX\n"
            "Combining my judgment and the votes for the experts' choices, I think the answer should be ....... "
            "(wrap the specific option as answer in {}, your answer must come from options)"
        )
        integration_prompt_text_final = integration_prompt_text_final + integration_prompt_text_format
        self.logger.info("开始整合修正后的专家回答...")
        integration_start_time = time.time()

        # Call base model for final integration. History starts fresh for the final integration.
        final_integration_result = self.base_model.generate_response(
            pixel_values,
            num_patches_list,
            integration_prompt_text_final,
            history=None 
        )
        final_response = final_integration_result["text"]
        final_history = final_integration_result["history"] # Get history from this final step

        integration_end_time = time.time()
        self.logger.info(f"最终整合完成，耗时: {integration_end_time - integration_start_time:.2f}s")
        self.logger.info(f"最终修正后整合结果: {final_response[:200]}...")

        # Return the final response, the history FROM THIS LAST CALL, and empty critiques
        return {"text": final_response, "history": final_history, "critiques": []} 


# --- Remove or adapt router-related methods ---
# Removed as they were API-based and not compatible with local model.chat

# --- End of ExpertAgentModel --- 