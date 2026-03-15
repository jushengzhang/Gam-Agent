import requests
import json
import logging
import time
from typing import Dict, Any, Optional, Union

class APIClient:
    """
    API客户端类，用于与大语言模型API进行交互
    """
    
    def __init__(self, api_config: Dict[str, Any]):
        """
        初始化API客户端
        
        Args:
            api_config: API配置字典，包含endpoint、key等信息
        """
        self.logger = logging.getLogger(__name__)
        self.config = api_config
        
        # 从配置中获取关键参数
        self.base_url = api_config.get('api_url', '')
        self.api_key = api_config.get('api_key', '')
        self.default_model = api_config.get('model_name', 'qwen/qwen-2.5-vl-72b-instruct')
        self.timeout = api_config.get('timeout', 120)  # 默认120秒超时
        self.max_retries = api_config.get('max_retries', 3)
        self.retry_delay = api_config.get('retry_delay', 2)
        
        # 请求头
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        # 如果有API密钥，添加到请求头
        if self.api_key:
            self.headers['Authorization'] = f"Bearer {self.api_key}"
        
        self.logger.info(f"API客户端初始化完成，目标端点: {self.base_url}")
    
    def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送生成请求到API
        
        Args:
            request_data: 请求数据，包含prompt等
            
        Returns:
            Dict: API响应
        """
        # 如果没有指定模型，使用默认模型
        if 'model' not in request_data:
            request_data['model'] = self.default_model
            
        # 日志记录请求（不包含敏感信息）
        safe_log_data = request_data.copy()
        if 'prompt' in safe_log_data:
            prompt = safe_log_data['prompt']
            safe_log_data['prompt'] = f"{prompt[:100]}..." if len(prompt) > 100 else prompt
            
        self.logger.info(f"发送API请求: {json.dumps(safe_log_data, ensure_ascii=False)}")
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                # 发送HTTP请求
                start_time = time.time()
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=request_data,
                    timeout=self.timeout
                )
                end_time = time.time()
                
                # 记录响应时间
                self.logger.info(f"API响应耗时: {end_time - start_time:.2f}秒")
                
                # 检查HTTP状态码
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # 如果API返回错误信息
                        if 'error' in result:
                            error_msg = result.get('error', {}).get('message', '未知API错误')
                            self.logger.error(f"API返回错误: {error_msg}")
                            
                            # 如果是可以重试的错误
                            if 'rate_limit' in error_msg.lower() or 'timeout' in error_msg.lower():
                                if attempt < self.max_retries - 1:
                                    wait_time = self.retry_delay * (attempt + 1)
                                    self.logger.info(f"API请求受限，等待{wait_time}秒后重试...")
                                    time.sleep(wait_time)
                                    continue
                            
                            # 返回错误信息
                            return {
                                "text": f"API错误: {error_msg}",
                                "error": True,
                                "error_message": error_msg
                            }
                        
                        # 处理不同格式的API响应
                        if 'text' in result:
                            # 直接返回包含text的响应
                            return result
                        elif 'response' in result:
                            # 有些API返回response字段
                            return {"text": result['response']}
                        elif 'choices' in result and len(result['choices']) > 0:
                            # OpenAI格式的响应
                            message = result['choices'][0].get('message', {})
                            content = message.get('content', '')
                            return {"text": content}
                        else:
                            # 返回完整响应以防格式不匹配
                            self.logger.warning(f"API响应格式不符合预期: {result}")
                            return {"text": str(result), "raw_response": result}
                        
                    except json.JSONDecodeError:
                        # 非JSON响应
                        error_msg = f"API返回非JSON响应: {response.text[:200]}..."
                        self.logger.error(error_msg)
                        return {"text": error_msg, "error": True}
                else:
                    # HTTP错误
                    error_msg = f"API HTTP错误 {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    
                    # 如果是服务器错误或者超时，可以重试
                    if response.status_code >= 500 or response.status_code == 429:
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (attempt + 1)
                            self.logger.info(f"HTTP错误 {response.status_code}，等待{wait_time}秒后重试...")
                            time.sleep(wait_time)
                            continue
                    
                    return {"text": f"HTTP错误 {response.status_code}", "error": True}
            
            except requests.exceptions.Timeout:
                error_msg = f"API请求超时 (尝试 {attempt+1}/{self.max_retries})"
                self.logger.error(error_msg)
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.info(f"请求超时，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                
                return {"text": error_msg, "error": True}
                
            except requests.exceptions.ConnectionError:
                error_msg = f"API连接错误 (尝试 {attempt+1}/{self.max_retries})"
                self.logger.error(error_msg)
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.info(f"连接错误，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                
                return {"text": error_msg, "error": True}
                
            except Exception as e:
                # 捕获其他所有异常
                error_msg = f"API请求异常: {str(e)}"
                self.logger.error(error_msg)
                import traceback
                self.logger.error(f"异常堆栈: {traceback.format_exc()}")
                return {"text": error_msg, "error": True}
        
        # 所有重试都失败
        error_msg = f"API请求在{self.max_retries}次尝试后仍然失败"
        self.logger.error(error_msg)
        return {"text": error_msg, "error": True}