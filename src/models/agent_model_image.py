import torch
import numpy as np
import json
import os
import base64
import io
import re
import logging
import requests
import yaml
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import sys
import time
import magic
import re
import math
from dashscope import MultiModalConversation
from openai import OpenAI




class AgentModel:
    
    """视频语言代理模型基类（API模式）"""
    
    def __init__(self, config):
        """
        初始化代理模型
        
        Args:
            config: 模型配置，包含API密钥和端点URL
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.api_key = config.get("api", {}).get("api_key") or config.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        self.api_url = config.get("api", {}).get("api_url", "https://openrouter.ai/api/v1/chat/completions") or config.get("api_url") or os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.model_name = config.get("api", {}).get("model_name", "qwen/qwen-2.5-vl-72b-instruct") or config.get("model_name") or os.environ.get("OPENROUTER_MODEL_NAME", "qwen/qwen-2.5-vl-72b-instruct")
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def process_video(self, video_frames):
        """
        处理视频帧
        
        Args:
            video_frames: 视频帧数组，可以是numpy数组或PyTorch张量
            
        Returns:
            array: 处理后的视频关键帧
        """
        # 检查视频帧类型
        if isinstance(video_frames, torch.Tensor):
            # 如果是张量，确保格式正确
            # 视频帧应该是[num_frames, height, width, channels]格式
            # 但可能是[num_frames, channels, height, width]格式，需要转换
            if video_frames.dim() == 4 and video_frames.shape[1] == 3:
                # 形状是[num_frames, channels, height, width]，需要转换为[num_frames, height, width, channels]
                video_frames = video_frames.permute(0, 2, 3, 1)
                
            # 转换为CPU上的numpy数组
            video_frames = video_frames.cpu().numpy()
        
        # 确保视频帧是numpy数组
        if not isinstance(video_frames, np.ndarray):
            raise ValueError(f"视频帧必须是numpy数组或PyTorch张量，但得到的是{type(video_frames)}")
            
        # 确保维度正确
        if video_frames.ndim != 4:
            raise ValueError(f"视频帧必须是4维数组[num_frames, height, width, channels]，但得到的是{video_frames.shape}")
            
        # 获取视频帧数量
        print(f"视频帧数量: {len(video_frames)}")
        num_frames = len(video_frames)
            
        # 选择关键帧（八分之一均匀采样）
        if num_frames == 1:
            key_frames = video_frames
        else:
            max_key_frames = self.config.get('max_key_frames', 10)  # 采样率为1/8
            max_key_frames = num_frames
            if num_frames > max_key_frames:
                indices = np.linspace(0, num_frames-1, max_key_frames).astype(int)
                key_frames = video_frames[indices]
            else:
                key_frames = video_frames
            
        return key_frames
    
    def estimate_uncertainty(self, response, logprobs=None, top_logprobs=None):
        """
        量化智能体响应的不确定性
        
        Args:
            response: 智能体生成的文本响应
            logprobs: 可选，来自模型的token概率信息
            top_logprobs: 可选，来自模型的top_k token概率信息
            
        Returns:
            float: 不确定性评分（0-1之间，1表示最不确定）         
        """
        # 策略1: 首先尝试使用logprobs和top_logprobs数据（优先级最高）
        if logprobs is not None and top_logprobs is not None:
            try:
                # 计算归一化熵
                def calculate_normalized_entropy(probs):
                    if not probs:
                        return 1.0  # 如果没有概率数据，返回最大不确定性
                    K = len(probs)
                    if K == 0:
                        return 1.0
                    # 计算原始熵
                    H = -sum(p * math.log2(p) for p in probs)
                    # 归一化熵
                    H_hat = H / math.log2(K)
                    return H_hat

                # 计算尺度化 Top-2 差
                def calculate_scaled_top2_diff(probs):
                    if len(probs) < 2:
                        return 0.5  # 如果没有足够的概率数据，返回中等不确定性
                    # 获取前两个概率
                    p1, p2 = sorted(probs, reverse=True)[:2]
                    # 计算差值
                    delta = p1 - p2
                    # 使用sigmoid函数进行尺度化，beta=0.7
                    beta = 0.7
                    G = 1 / (1 + math.exp(-beta * delta))
                    return G

                # 计算统一的置信度分数
                def calculate_confidence_score(H_hat, G, alpha=0.6):
                    # 计算不确定性
                    uncertainty = alpha * H_hat + (1 - alpha) * (1 - G)
                    # 转换为置信度
                    confidence = 1 - uncertainty
                    return confidence

                # 处理每个位置的top_logprobs
                all_probs = []
                for token_probs in top_logprobs:
                    if isinstance(token_probs, dict):
                        # 将logprobs转换为概率
                        probs = [math.exp(lp) for lp in token_probs.values()]
                        all_probs.extend(probs)

                if all_probs:
                    # 计算归一化熵
                    H_hat = calculate_normalized_entropy(all_probs)
                    # 计算尺度化 Top-2 差
                    G = calculate_scaled_top2_diff(all_probs)
                    # 计算统一的置信度分数
                    confidence = calculate_confidence_score(H_hat, G)
                    # 将置信度转换为不确定性（1 - confidence）
                    uncertainty = 1 - confidence
                    self.logger.info(f"基于概率计算的不确定性: {uncertainty:.4f}，归一化熵: {H_hat:.4f}，Top-2差: {G:.4f}")
                    return uncertainty

            except Exception as e:
                self.logger.warning(f"处理概率数据时出错: {e}，将使用文本特征估计不确定性")

        # 策略2: 如果概率数据处理失败，尝试使用logprobs数据
        if logprobs is not None:
            try:
                # 确保logprobs是数值列表
                if isinstance(logprobs, list):
                    # 过滤掉非数值元素
                    valid_logprobs = [lp for lp in logprobs if isinstance(lp, (int, float)) and lp is not None]
                    
                    if valid_logprobs:
                        # 计算平均log概率的绝对值（越低表示越确定）
                        avg_logprob = sum(abs(lp) for lp in valid_logprobs) / len(valid_logprobs)
                        # 将logprob映射到0-1区间的不确定性得分（1表示最不确定）
                        uncertainty = min(1.0, max(0.0, avg_logprob / 10.0))
                        self.logger.info(f"基于普通logprobs计算的不确定性: {uncertainty:.4f}，平均logprob: {avg_logprob:.4f}")
                        return uncertainty
            except Exception as e:
                self.logger.warning(f"处理logprobs时出错: {e}，将使用文本特征估计不确定性")

        # 策略3: 如果没有概率数据或处理出错，基于文本特征估计不确定性
        self.logger.info("没有可用的概率数据，使用文本特征估计不确定性")
        
        # 将响应文本转换为小写以便匹配
        response_lower = response.lower()
        
        words = [w for w in re.split(r'[\s,.!?;:()\[\]{}"\']', response_lower) if w]
        num_words = len(words)
        
        # 如果响应为空，直接返回最高不确定性
        if num_words == 0:
            self.logger.info("响应为空，返回最高不确定性")
            return 1.0
        
        # 定义不同级别的不确定性标记词及其权重
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
        
        # 初始化加权分数
        raw_weighted_score = 0
        
        # 检查强不确定性标记词（权重=3）
        for marker in strong_markers:
            if ' ' in marker:  # 短语
                count = response_lower.count(marker)
            else:  # 单词
                count = words.count(marker)
            raw_weighted_score += count * 3
        
        # 检查中等不确定性标记词（权重=2）
        for marker in medium_markers:
            if ' ' in marker:
                count = response_lower.count(marker)
            else:
                count = words.count(marker)
            raw_weighted_score += count * 2
        
        # 检查弱不确定性标记词（权重=1）
        for marker in weak_markers:
            if ' ' in marker:
                count = response_lower.count(marker)
            else:
                count = words.count(marker)
            raw_weighted_score += count * 1
        
        # 检查响应文本中的问号（非结尾）
        if '?' in response_lower[:-1]:
            raw_weighted_score += 2
        
        # 计算加权分数密度
        score_density = raw_weighted_score / num_words
                
        # 使用Sigmoid函数将分数密度映射到0-1区间
        k = 60  # 陡峭度参数
        offset = 0.04  # 中心点参数
        exponent = -k * (score_density - offset)
        uncertainty = 1 / (1 + math.exp(exponent))
        
        self.logger.info(f"基于文本特征计算的不确定性: {uncertainty:.4f}，原始加权分数: {raw_weighted_score}，词数: {num_words}，密度: {score_density:.6f}")
        return uncertainty
        
    
    def generate_response(self, image_inputs, prompt, video_or_image_path=None):
        """
        通过API生成回复 (处理图像输入)

        Args:
            image_inputs: 图像输入列表，可以是PIL Image对象或numpy数组
            prompt: 提示文本
            video_or_image_path: (此参数在此版本中不再直接用于API调用，但保留以兼容旧接口)

        Returns:
            dict: 包含生成文本和概率信息的字典
        """
        # 准备请求内容（文本+图像）
        content = []

        # 添加文本内容
        content.append({"type": "text", "text": prompt})

        # 将图像输入转换为base64编码并添加到内容中
        image_count = 0
        for img_input in image_inputs:
            try:
                img = None
                # 处理 numpy 数组
                if isinstance(img_input, np.ndarray):
                    # 确保是uint8类型
                    if img_input.dtype != np.uint8:
                        # 假设范围是0-1，需要转换
                        if img_input.max() <= 1.0 and img_input.min() >= 0.0:
                           img_input = (img_input * 255).astype(np.uint8)
                        else:
                           img_input = img_input.astype(np.uint8) # 尝试直接转换

                    # 确保是3通道RGB图像
                    if img_input.ndim == 3 and img_input.shape[-1] == 3:
                        img = Image.fromarray(img_input)
                    elif img_input.ndim == 2: #灰度图
                        img = Image.fromarray(img_input).convert('RGB')
                    elif img_input.ndim == 3 and img_input.shape[-1] == 4: #RGBA
                         img = Image.fromarray(img_input).convert('RGB')
                    else:
                        self.logger.warning(f"跳过无效形状的numpy图像帧: {img_input.shape}")
                        continue
                # 处理 PIL 图像
                elif isinstance(img_input, Image.Image):
                    if img_input.mode != 'RGB':
                        img = img_input.convert('RGB')
                    else:
                        img = img_input
                # 处理其他可能的类型（例如 PyTorch 张量）
                elif isinstance(img_input, torch.Tensor):
                     # 确保在 CPU 上并转换为 numpy
                    img_np = img_input.cpu().numpy()
                     # 可能是 [C, H, W] 或 [H, W, C]
                    if img_np.ndim == 3 and img_np.shape[0] == 3: # [C, H, W]
                        img_np = np.transpose(img_np, (1, 2, 0))
                    elif img_np.ndim != 3 or img_np.shape[-1] != 3: # [H, W, C]
                        self.logger.warning(f"跳过无效形状的Tensor图像帧: {img_np.shape}")
                        continue

                     # 确保数据类型和范围
                    if img_np.dtype != np.uint8:
                        if img_np.max() <= 1.0 and img_np.min() >= 0.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)

                    img = Image.fromarray(img_np)

                else:
                    self.logger.warning(f"不支持的图像输入类型: {type(img_input)}")
                    continue

                # 调整图像大小（如果需要）
                max_size = 1024 # OpenAI 推荐的大小限制可能不同，需查阅文档
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 转换为JPEG格式
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=95) # 使用 JPEG 格式
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # 添加到内容列表
                content.append({
                    "type": "image_url",
                    "image_url": {
                        # OpenAI API 需要 'url' 字段
                        "url": f"data:image/jpeg;base64,{img_str}"
                    }
                })
                image_count += 1
                # self.logger.info(f"成功处理一张图像，大小: {img.size}")

            except Exception as e:
                self.logger.error(f"处理图像输入时出错: {e}", exc_info=True) # 添加 traceback 信息
                continue # 跳过有问题的图像

        if image_count == 0:
            error_msg = "没有有效的图像可以处理"
            self.logger.error(error_msg)
            return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}

        self.logger.info(f"准备发送请求到API，包含 {len(content)} 个内容项（1个文本 + {image_count} 个图像）")

        # 准备 OpenAI API 请求参数
        # 系统提示可以根据需要调整，使其更通用
        api_params = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant capable of analyzing images. Analyze the provided image(s) based on the user's question."
                },
                {
                    "role": "user",
                    "content": content  # content 已经是正确的格式
                }
            ],
            # logprobs 和 top_logprobs 可能不被所有视觉模型支持，或者需要特定方式请求
            # 这里暂时保留，但需要确认API文档
            "logprobs": True,
            "top_logprobs": 5,
            # "return_logprobs": True, # 这个参数不是标准OpenAI参数
            "max_tokens": 1024, # 根据需要调整
            "temperature": 0.7 # 根据需要调整
        }

        try:
            # 统一使用标准 OpenAI 兼容 API 调用
            self.logger.info("使用标准 OpenAI 兼容 API 进行调用...")
            headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    # OpenRouter/兼容API可能需要这些
                    "HTTP-Referer": "https://your-app-url.com", # 替换为你的应用URL或标识
                    "X-Title": "VLM Agent",
                    "Accept": "application/json"
            }
            # self.logger.info(f"API 请求参数: {json.dumps(api_params, ensure_ascii=False, indent=2)}") # 注意不要打印完整的 base64 图像

            response = requests.post(
                self.api_url,
                headers=headers,
                json=api_params,
                timeout=120 # 增加超时时间以处理多图
            )

            self.logger.info(f"API 响应状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                # self.logger.info(f"API 响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    generated_text = ""
                    if "message" in choice and "content" in choice["message"]:
                        generated_text = choice["message"]["content"]
                        self.logger.info(f"成功获取到 content: {generated_text[:200]}...") # 打印部分内容
                    elif "text" in choice: # 兼容旧格式
                         generated_text = choice["text"]
                         self.logger.info(f"成功获取到 content (旧格式): {generated_text[:200]}...")
                    else:
                        error_msg = f"API响应中没有找到文本内容，choice结构: {json.dumps(choice, ensure_ascii=False)}"
                        self.logger.error(error_msg)
                        return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}

                    if not generated_text:
                        error_msg = "API返回的content为空"
                        self.logger.error(error_msg)
                        return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}

                    # 提取logprobs信息 (如果API返回了)
                    logprobs_data = None
                    top_logprobs_data = None
                    if "logprobs" in choice and choice["logprobs"] is not None:
                        try:
                            logprobs_content = choice["logprobs"].get("content")
                            if isinstance(logprobs_content, list) and logprobs_content:
                                logprobs_data = []
                                top_logprobs_data = []
                                for entry in logprobs_content:
                                    if isinstance(entry, dict):
                                        if "logprob" in entry:
                                            logprobs_data.append(entry["logprob"])
                                        if "top_logprobs" in entry and isinstance(entry["top_logprobs"], list):
                                            token_top_logprobs = {item["token"]: item["logprob"]
                                                                  for item in entry["top_logprobs"]
                                                                  if isinstance(item, dict) and "token" in item and "logprob" in item}
                                            if token_top_logprobs:
                                                top_logprobs_data.append(token_top_logprobs)
                                logprobs_data = [lp for lp in logprobs_data if isinstance(lp, (int, float)) and lp is not None]
                                self.logger.info(f"成功提取 logprobs ({len(logprobs_data)}) 和 top_logprobs ({len(top_logprobs_data)})")
                        except Exception as e:
                            self.logger.warning(f"提取logprobs时出错: {str(e)}")

                    return {
                        "text": generated_text,
                        "logprobs": logprobs_data,
                        "top_logprobs": top_logprobs_data
                    }
                else:
                     error_msg = f"API响应格式无效，缺少 'choices' 字段: {json.dumps(result, ensure_ascii=False)}"
                     self.logger.error(error_msg)
                     return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}

            else:
                # 处理 API 错误
                try:
                    error_response = response.json()
                    error_detail = error_response.get('error', {})
                    error_msg = f"API请求失败 - 状态码: {response.status_code}, 类型: {error_detail.get('type')}, 消息: {error_detail.get('message', response.text)}"
                except json.JSONDecodeError:
                    error_msg = f"API请求失败 - 状态码: {response.status_code}, 响应: {response.text}"
                self.logger.error(error_msg)
                # 可以考虑根据特定错误码进行重试或不同处理
                return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}

        except requests.exceptions.Timeout:
            error_msg = f"API请求超时 ({self.api_url})"
            self.logger.error(error_msg)
            return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求过程中出现网络错误: {e}"
            self.logger.error(error_msg)
            return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}
        except Exception as e:
            error_msg = f"处理API响应或请求时出现未知错误: {e}"
            self.logger.error(error_msg, exc_info=True) # 添加 traceback
            return {"text": f"处理错误: {error_msg}", "logprobs": None, "top_logprobs": None}

class ExpertAgentModel:
    """专家代理模型，管理多个不同专长的代理模型"""
    
    def __init__(self, model_config, experts_config, api_config=None):
        """
        初始化专家代理模型
        
        Args:
            model_config: 模型配置
            experts_config: 专家配置
            api_config: API配置（可选）
        """
        self.model_config = model_config
        self.experts_config = experts_config
        self.api_config = api_config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置日志 - 移到前面
        self.logger = logging.getLogger(__name__)
        
        # 合并model_config和api_config作为AgentModel的配置
        agent_config = model_config.copy()
        agent_config["api"] = self.api_config
        
        
        # 初始化基础模型
        self.base_model = AgentModel(agent_config)
        
        # 加载专家配置文件
        self.load_experts_config()
        
        # 初始化MAB（多臂老虎机）
        self.num_arms = len(self.experts) + len(self.combinations)  # 专家数量 + 组合策略数量
        self.arms = [{'total_reward': 0, 'count': 0} for _ in range(self.num_arms)]
        self.epsilon = self.experts_settings.get('epsilon', 0.1)  # 探索概率
        self.high_uncertainty_threshold = self.experts_settings.get('high_uncertainty_threshold', 0.6)
        
        # 初始化BERT模型（用于相似度计算）
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert_model = AutoModel.from_pretrained("bert-base-chinese").to(self.device)
        
    def load_experts_config(self, use_default=False):
        """
        加载专家配置文件
        
        Args:
            use_default: 是否使用默认配置
        """
        # 如果指定使用默认配置，则直接使用内置配置
        if use_default:
            self._load_default_experts_config()
            return
            
        config_file_path = self.experts_config.get('config_file')
        if not config_file_path:
            self.logger.warning("未指定专家配置文件路径，将使用默认专家配置")
            self._load_default_experts_config()
            return
        
        # 获取配置文件的绝对路径
        if not os.path.isabs(config_file_path):
            # 如果是相对路径，则相对于当前工作目录解析
            config_file_path = os.path.join(os.getcwd(), config_file_path)
        
        try:
            if not os.path.exists(config_file_path):
                self.logger.warning(f"专家配置文件不存在: {config_file_path}，将尝试其他路径")
                # 尝试在相对于当前脚本的目录搜索
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(script_dir))  # 上级的上级目录
                alt_config_path = os.path.join(project_root, config_file_path.replace("configs/", ""))
                
                if os.path.exists(alt_config_path):
                    self.logger.info(f"找到替代配置文件路径: {alt_config_path}")
                    config_file_path = alt_config_path
                else:
                    # 回退到默认配置
                    self.logger.warning(f"替代路径也不存在，将使用默认配置")
                    self._load_default_experts_config()
                    return
            
            print("##########experts_config#########################")
            print(config_file_path)
            
            with open(config_file_path, 'r', encoding='utf-8') as f:
                experts_data = yaml.safe_load(f)
                
            # 加载基础设置
            self.experts_settings = experts_data.get('settings', {})
            
            # 加载专家列表
            self.experts = experts_data.get('experts', [])
            
            # 加载组合策略
            self.combinations = experts_data.get('combinations', [])
            
            # 加载评论专家
            self.critic_experts = experts_data.get('critic_experts', [])
            
            # 加载辩论设置
            self.debate_settings = experts_data.get('debate_settings', {})
            
            # 加载路由专家配置
            self.router_config = experts_data.get('router', {})
            
            # 如果路由模型名称未指定，使用默认模型
            if 'model_name' not in self.router_config:
                self.router_config['model_name'] = self.model_name
            
            # 验证配置有效性
            self._validate_experts_config()
            
            # 构建策略类型映射
            self.strategy_types = []
            for expert in self.experts:
                self.strategy_types.append(expert['display_name'])
            for combo in self.combinations:
                self.strategy_types.append(combo['display_name'])
                
            self.logger.info(f"已加载 {len(self.experts)} 个专家、{len(self.combinations)} 个组合策略和 {len(self.critic_experts)} 个评论专家")
            
        except Exception as e:
            self.logger.error(f"加载专家配置文件失败: {e}")
            self.logger.warning("回退到默认配置")
            # 回退到默认配置
            self._load_default_experts_config()
            
    def _load_default_experts_config(self):
        """Load default expert configuration"""
        # Use built-in default configuration
        self.experts_settings = {'epsilon': 0.1, 'high_uncertainty_threshold': 0.6}
        self.experts = [
            {
                'id': 0,
                'name': "scene_understanding",
                'display_name': "Scene Understanding",
                'weight': 1.0,
                'enabled': True,
                'keywords': ["describe", "what kind of", "looks like", "scene", "environment", "background"],
                'prompt_template': "Please act as a scene understanding expert and analyze the scene, environment, and background in this video in detail. Key points to focus on: venue setup, environmental features, surrounding objects, lighting conditions, and overall atmosphere. Original question: {instruction}"
            },
            {
                'id': 1,
                'name': "action_analysis",
                'display_name': "Action Analysis",
                'weight': 1.0,
                'enabled': True,
                'keywords': ["what to do", "action", "what happened", "how to do"],
                'prompt_template': "Please act as an action analysis expert and analyze all actions and movements in this video in detail. Key points to focus on: character actions, object movements, action sequences, action speed, and action purpose. Original question: {instruction}"
            },
            {
                'id': 2,
                'name': "semantic_understanding",
                'display_name': "Semantic Understanding",
                'weight': 1.0,
                'enabled': True,
                'keywords': ["why", "meaning", "purpose", "significance", "express"],
                'prompt_template': "Please act as a semantic understanding expert and analyze the meaning and information conveyed in this video in detail. Key points to focus on: video theme, underlying messages, cultural context, emotional expression, and video purpose. Original question: {instruction}"
            }
        ]
        self.combinations = [
            {
                'id': 3,
                'name': "scene_action",
                'display_name': "Scene+Action",
                'experts': [0, 1],
                'weight': 1.0,
                'enabled': True,
                'prompt_template': "Please integrate the following two experts' analysis of the video to provide a comprehensive video understanding:\n\nScene Understanding Expert: {expert_0_response}\n\nAction Analysis Expert: {expert_1_response}\n\nPlease provide a complete and coherent video analysis that synthesizes the insights from both experts. Original question: {instruction}"
            },
            {
                'id': 4,
                'name': "action_semantic",
                'display_name': "Action+Semantic",
                'experts': [1, 2],
                'weight': 1.0,
                'enabled': True,
                'prompt_template': "Please integrate the following two experts' analysis of the video to provide a comprehensive video understanding:\n\nAction Analysis Expert: {expert_1_response}\n\nSemantic Understanding Expert: {expert_2_response}\n\nPlease provide a complete and coherent video analysis that synthesizes the insights from both experts. Original question: {instruction}"
            },
            {
                'id': 5,
                'name': "scene_semantic",
                'display_name': "Scene+Semantic",
                'experts': [0, 2],
                'weight': 1.0,
                'enabled': True,
                'prompt_template': "Please integrate the following two experts' analysis of the video to provide a comprehensive video understanding:\n\nScene Understanding Expert: {expert_0_response}\n\nSemantic Understanding Expert: {expert_2_response}\n\nPlease provide a complete and coherent video analysis that synthesizes the insights from both experts. Original question: {instruction}"
            },
            {
                'id': 6,
                'name': "full_ensemble",
                'display_name': "Full Integration",
                'experts': [0, 1, 2],
                'weight': 1.2,
                'enabled': True,
                'prompt_template': "Please integrate the following three experts' analysis of the same video to provide a comprehensive video understanding:\n\nScene Understanding Expert: {expert_0_response}\n\nAction Analysis Expert: {expert_1_response}\n\nSemantic Understanding Expert: {expert_2_response}\n\nOriginal question: {instruction}\n\nPlease provide a comprehensive and coherent video analysis that integrates the insights from all three experts and directly answers the original question."
            }
        ]
        
        # Default critic experts
        self.critic_experts = [
            {
                'id': 7,
                'name': "fact_checker",
                'display_name': "Fact Checker",
                'enabled': True,
                'critique_template': "Please act as a fact-checking expert and evaluate the factual accuracy in the following analysis:\n\n{response}\n\nPlease point out any potential factual errors or inaccuracies in the analysis and provide improvement suggestions."
            },
            {
                'id': 8,
                'name': "logic_analyzer",
                'display_name': "Logic Analyzer",
                'enabled': True,
                'critique_template': "Please act as a logic analysis expert and evaluate the logical coherence and reasoning quality in the following analysis:\n\n{response}\n\nPlease point out any logical gaps, inconsistencies, or insufficient reasoning in the analysis and provide improvement suggestions."
            }
        ]
        
        # Default debate settings
        self.debate_settings = {
            'quality_metrics': [
                {'name': 'relevance', 'weight': 0.3, 'description': 'Relevance to the question'},
                {'name': 'factuality', 'weight': 0.3, 'description': 'Factual accuracy of the answer'},
                {'name': 'completeness', 'weight': 0.4, 'description': 'Completeness of the answer'}
            ],
            'filter_rules': {
                'min_critique_length': 50,
                'max_critique_similarity': 0.85,
                'required_suggestion_count': 1
            }
        }
        
        # Build strategy type mapping
        self.strategy_types = []
        for expert in self.experts:
            self.strategy_types.append(expert['display_name'])
        for combo in self.combinations:
            self.strategy_types.append(combo['display_name'])
            
        self.logger.info(f"Using default configuration: {len(self.experts)} experts and {len(self.combinations)} combination strategies")
        
    def _validate_experts_config(self):
        """验证专家配置的有效性"""
        # 检查专家ID是否连续且从0开始
        expert_ids = [expert['id'] for expert in self.experts]
        if not all(i in expert_ids for i in range(len(self.experts))):
            raise ValueError("专家ID必须是连续的整数，从0开始")
            
        # 检查组合策略中引用的专家ID是否有效
        for combo in self.combinations:
            for expert_id in combo['experts']:
                if expert_id not in expert_ids:
                    raise ValueError(f"组合策略 {combo['name']} 引用了不存在的专家ID {expert_id}")
        
    
    def create_expert_prompt(self, expert, instruction, options=None):
        """
        创建专家提示
        
        Args:
            expert: 专家配置
            instruction: 用户指令
            
        Returns:
            str: 提示文本
            
        """
             
        # 使用专家的提示模板并替换指令
        template = expert.get('prompt_template', '')
        template_instruction = template.replace('{instruction}', instruction)

        #如果options在外部传入了但是还是null，直接彻底退出程序，报错，有选择题的时候options不能为null
        if options is None:
            self.logger.error("options不能为null")
            sys.exit()
        
        # 判断template中是否需要options
        if 'options' in template:
            # 错误示例：options是一个字符串
            # options_str = ', '.join(options)  # 如果options是一个字符串，会报错
            
            # 正确示例：options是一个列表
            if isinstance(options, str):
                options_str = options  # 如果options是一个字符串，直接使用
            elif isinstance(options, list):
                options_str = ', '.join(options)  # 将列表转换为逗号分隔的字符串
            else:
                options_str = "" # 如果options不是字符串也不是列表，则为空字符串
            template_instruction = template_instruction.replace('{options}', options_str)
        
        return template_instruction 
    
    
    def process_with_experts(self, video_frames, instruction, output=None , options=None , 
                             choice_answer=None, whether_use_original_video=False , video_or_image_path=None):
        """
        使用专家处理视频和指令
        
        Args:
            video_frames: 视频帧
            instruction: 指令
            output: 参考输出（如有）
            options: 选项（如有）
            choice_answer: 选择答案（如有）,这里不参与推理，只参与训练时奖励计算
            whether_use_original_video: 是否使用原始视频
        Returns:
            dict: 处理结果
        """
        print(f"是否使用原始视频: {whether_use_original_video}")
        # 处理视频
        if whether_use_original_video is True:
            key_frames = video_frames
        else:
            key_frames = self.base_model.process_video(video_frames)
            
        video_or_image_path = video_or_image_path
        
        # 不再使用路由专家分析问题类型
        # question_features = self.analyze_question_with_router(instruction, key_frames , whether_use_original_video, video_or_image_path)
        # self.logger.info(f"路由专家分析结果: {question_features}")
        
        # 生成各专家的回答
        agent_responses = []
        agent_uncertainties = []
        
        # 获取所有专家的回答
        for expert in self.experts:
            # 检查专家是否启用
            if not expert.get('enabled', True):
                # 如果专家未启用，添加空回答和最高不确定性
                agent_responses.append({
                    "role": expert['name'],
                    "response": ""
                })
                agent_uncertainties.append(1.0)
                continue
                
            # 创建专家提示
            expert_prompt = self.create_expert_prompt(expert, instruction , options)
            
            
            # 获取专家回答
            expert_result = self.base_model.generate_response(key_frames, expert_prompt , video_or_image_path)
            expert_response = expert_result["text"]
            
            # 估计不确定性
            expert_uncertainty = self.base_model.estimate_uncertainty(
                expert_response, 
                expert_result.get("logprobs"), 
                expert_result.get("top_logprobs"),
            )
            
            # 保存专家回答和不确定性
            agent_responses.append({
                "role": expert['name'],
                "response": expert_response
            })
            agent_uncertainties.append(expert_uncertainty)

            time.sleep(1)#防止api调用过于频繁
        
        # 记录不确定性分数
        self.logger.info(f"智能体不确定性评分: {agent_uncertainties}")
        
        # 始终整合所有专家的回答
        initial_response = ""
        
        # 原本基于 selected_arm 的条件分支被移除
        # 直接使用整合所有专家回答的逻辑
        integration_prompt_text = (
            "You are a VQA answer integration expert. "
            "Your goal is to integrate the answers of multiple experts.\n" +
            ''.join([f"Expert {i+1} says: {resp['response']}\n" for i, resp in enumerate(agent_responses)]) +
            ''.join([f"Expert {i+1} uncertainty score: {uncertainty:.4f}\n" for i, uncertainty in enumerate(agent_uncertainties)]) +
            "Please consider each expert's uncertainty when summarizing.\n"
            "Your answer format is as follows:\n"
            "Expert 1 final answer is: ......., Expert 2 final answer is: ......., Expert 3 final answer is: .......\n"
            "Combining my judgment and the votes for the experts' choices, I think the answer should be ....... "
            "(wrap the specific option as answer in {}, your answer must come from options)"
        )
        # Add expert uncertainty scores to the integration prompt
        integration_prompt_text = integration_prompt_text + "\nExpert uncertainty scores:\n" + '\n'.join([f"Expert {i+1}: {uncertainty:.4f}" for i, uncertainty in enumerate(agent_uncertainties)])
        # Call API for integration
        integration_result = self.base_model.generate_response(key_frames, integration_prompt_text, video_or_image_path)
        initial_response = integration_result["text"]

        # 进行辩论流程以提高质量，传入参考答案
        result = {
            "agent_uncertainties": agent_uncertainties,
            "agent_responses": agent_responses,
            "initial_response": initial_response
        }
        
        if self.experts_settings.get('use_debate', True):
            #调用run_debate_process_new进行多轮辩论
            for i in range(self.experts_settings.get('debate_rounds', 3)):
                print(f"开始辩论轮次 {i+1}/{self.experts_settings.get('debate_rounds', 3)}")
                temp_response = self.run_debate_process_new(key_frames, instruction, options, initial_response, output , whether_use_original_video, video_or_image_path)
                self.logger.info(f"辩论第{i+1}轮的回答：{temp_response}")
                initial_response = temp_response
            final_response = initial_response
            result["debate_result"] = "已经废弃不在使用此属性"
        else:
            final_response = initial_response
        
        result["final_response"] = final_response
        
        self.logger.info("已强制整合所有专家回答，未使用MAB选择策略。")
        
        return result
    
    def process_with_single_agent(self, video_frames, instruction, output=None, options=None, choice_answer=None, whether_use_original_video=False , video_or_image_path=None):
        """
        使用单个专家进行处理
        Args:
            video_frames: 视频帧
            instruction: 指令
            output: 参考输出（如有）
            options: 选项（如有）
            choice_answer: 选择答案（如有）,这里不参与推理，只参与训练时奖励计算
            whether_use_original_video: 是否使用原始视频
        Returns:
            dict: 处理结果
        """
        
        if whether_use_original_video is True:
            key_frames = video_frames
        else:
            key_frames = self.base_model.process_video(video_frames)
        
        # 将options列表转换为字符串
        options_str = "选项: " + ", ".join(options) if options else ""
        instruction = "This is a video" + instruction + options_str + "Please answer the question and put your chosen answer in {answer} (wrap the specific option with {}), your answer must be from the multiple choice options."
        
        # 使用单个专家进行推理
        result = self.base_model.generate_response(key_frames, instruction, video_or_image_path)
        
        return result
    
        
    def run_debate_process_new(self, key_frames, instruction, options, initial_response, reference_output=None, whether_use_original_video=False , video_or_image_path=None):
        """
        执行辩论过程以提高回答质量
        
        Args:
            key_frames: 视频关键帧
            instruction: 用户问题
            options: 选项
            initial_response: 初始回答
            reference_output: 参考答案（可选）
            
        Returns:
            dict: 辩论结果
        """
        #这里在上一条回答的基础上继续调用专家重新生成答案
        #默认使用全部专家
        # 生成各专家的回答
        agent_responses = []
        instruction_new = f"{instruction}. This is your previous answer: {initial_response}. Please refer to the previous answer to review your answer.\n\n"
        # 获取所有专家的回答
        for expert in self.experts:
            # 创建专家提示
            expert_prompt = self.create_expert_prompt(expert, instruction_new, options)
            # 获取专家回答
            expert_result = self.base_model.generate_response(key_frames, expert_prompt, video_or_image_path=video_or_image_path)
            expert_response = expert_result["text"]
            agent_responses.append({
                "expert_id": expert["id"], 
                "response": expert_response
            })
            time.sleep(1) # 防止API调用过于频繁
            
        # 特化出来一个integration专家
        integration_prompt_text = (
            "You are a VQA answer integration expert. "
            "Your goal is to integrate the answers of multiple experts.\n" +
            ''.join([f"Expert {resp['expert_id']+1} says: {resp['response']}\n" for resp in agent_responses]) +
            "Your answer format is as follows:\n"
            "Expert 1 final answer is: ......., Expert 2 final answer is: ......., Expert 3 final answer is: .......\n"
            "Combining my judgment and the votes for the experts' choices, I think the answer should be ....... "
            "(wrap the specific option as answer in {}, your answer must come from options)"
        )
        # 调用API进行整合
        integration_result = self.base_model.generate_response(key_frames, integration_prompt_text, video_or_image_path=video_or_image_path)
        initial_response = integration_result["text"]
        
        return initial_response
                        
    

    
    