import os
import sys
import torch
import numpy as np
from PIL import Image
import yaml
import logging
import warnings

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change import to the local image agent model
from src.models.agent_model_local_image import ExpertAgentModel

# --- Suppress Specific Transformers Warning ---
# Ignore the warning about setting pad_token_id to eos_token_id when both might be None
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:.* for open-end generation.")
# --- End Suppress Warning ---

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    # Point to the new local image config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'configs', 'multi_agent_local_image.yaml')

    if not os.path.exists(config_path):
        logging.error(f"配置文件未找到: {config_path}")
        # Optionally fallback or raise error
        # logging.warning(f"回退到 MMbench_config_Choice.yaml")
        # config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        #                          'configs', 'MMbench_config_Choice.yaml')
        return {} # Return empty dict if config not found

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        config['config_file_path'] = config_path # Store path for logging
    return config

def process_image(image_path):
    """处理图像，返回numpy数组"""
    image = Image.open(image_path).convert('RGB')
    # 调整图像大小
    image = image.resize((224, 224))
    # 转换为numpy数组
    image_np = np.array(image)
    # 添加批次维度
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def main():
    # 设置日志
    logger = setup_logging()
    logger.info("开始本地多智能体图像处理演示 (使用 InternVL3)")
    
    # 加载配置
    config = load_config()
    if not config:
        logger.error("无法加载配置，退出。")
        return
    logger.info(f"配置加载完成 from {config.get('config_file_path', '未知路径')}")
    
    # ----- Configuration Check & Defaults ----- 
    model_config_dict = config.get('local_model', {})
    if 'model_path' not in model_config_dict:
        model_config_dict['model_path'] = 'OpenGVLab/InternVL3-14B'
        logger.warning(f"配置文件中未找到 'model_path'，使用默认值: {model_config_dict['model_path']}")
    # Add checks/defaults for other relevant model params if needed (e.g., generation_config)
    config['local_model'] = model_config_dict
    # ----- End Configuration Check -----

    # ----- Initialize Model ----- 
    logger.info("正在初始化专家代理模型 (本地图像模式)...")
    try:
        expert_agent = ExpertAgentModel(
            model_config=config.get('local_model', {}),
            experts_config=config.get('experts', {}),
            # api_config is ignored by the local model
        )
        logger.info("专家代理模型初始化完成")
    except Exception as e:
        logger.exception(f"初始化专家代理模型时出错: {e}")
        return
    # ----- End Initialize Model -----

    # ----- Set Logger Level to WARNING ----- 
    logging.getLogger().setLevel(logging.WARNING)
    logger.warning("日志级别已设置为 WARNING，将只显示重要消息和错误。")
    # ----- End Set Logger Level -----

    # ----- Get Image Paths ----- 
    # Use a list to hold image paths
    image_paths = []
    # Image 1: Get from config or use default
    default_image_path1 = "./PATH_TO_LOCAL_RESOURCE"
    image_path1 = config.get('demo', {}).get("demo_image_path", default_image_path1) # Check key 'demo_image_path' first
    if not os.path.exists(image_path1):
        logger.error(f"图像文件1未找到: {image_path1}")
    else:
        image_paths.append(image_path1)
        logger.warning(f"使用图像1: {image_path1}")

    # Image 2: Define path directly or get from config (e.g., demo_image_path2)
    image_path2 = "./PATH_TO_LOCAL_RESOURCE"
    if not os.path.exists(image_path2):
         logger.error(f"图像文件2未找到: {image_path2}")
    else:
         image_paths.append(image_path2)
         logger.warning(f"使用图像2: {image_path2}")

    if len(image_paths) < 2:
         logger.error("需要至少两张有效的图像进行多图测试，退出。")
         return
    # ----- End Get Image Paths -----

    # ----- Load Images ----- 
    image_inputs = [] # Initialize list to hold PIL Images
    try:
        for i, path in enumerate(image_paths):
             image = Image.open(path).convert('RGB')
             image_inputs.append(image)
             logger.warning(f"图像 {i+1} 加载完成: {path}")
        logger.warning(f"总共加载了 {len(image_inputs)} 张图像。")
    except Exception as e:
        logger.error(f"加载图像时出错: {e}") # Error might occur on any image
        return
    # ----- End Load Images -----

    # ----- Define Question and Options for Multi-Image ----- 
    # Adjust instruction for multi-image scenario
    instruction = "Image-1: <image>\nImage-2: <image>\n请比较并描述这两张图片的内容。"
    # Options might need adjustment based on the new question/images
    options = ["A. 两张图片都是风景照", "B. 两张图片都包含人物", "C. 一张是人像，一张是风景", "D. 无法比较"]
    # ----- End Define Question and Options -----

    logger.warning(f"\n===== 测试问题: {instruction} =====")
    logger.warning(f"选项: {', '.join(options)}")

    # 使用多智能体处理图像
    try:
        logger.warning("开始本地多智能体处理...") # Use warning to show
        result = expert_agent.process_with_experts(
            image_inputs=image_inputs, # Pass PIL image list
            instruction=instruction,
            options=options,
            # output=None, # Optional reference output
            # choice_answer=None # Optional reference answer
        )

        # 打印结果
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO) # Temporarily set back to INFO

        logger.info("\n--- 处理结果 ---")
        if "error" in result:
            logger.error(f"处理时发生错误: {result['error']}")
        else:
            logger.info("--- 专家回答 ---")
            if result.get("agent_responses"):
                 for i, expert_resp in enumerate(result["agent_responses"]):
                     response_text = expert_resp.get('response', '[无回应]')
                     logger.info(f" - {expert_resp.get('role', f'专家{i+1}')}: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            else:
                logger.warning("结果中未找到 'agent_responses'。")

            if result.get("agent_uncertainties"):
                uncertainties_str = ", ".join([f"{u:.3f}" for u in result["agent_uncertainties"]])
                logger.info(f"专家不确定性: [{uncertainties_str}]")

            initial_resp = result.get('initial_response', '[无初始回应]')
            final_resp = result.get('final_response', '[无最终回应]')
            logger.info(f"初始整合回答: {initial_resp[:300]}{'...' if len(initial_resp) > 300 else ''}")
            if result.get("debate_result"): # Check if debate happened
                logger.info(f"最终辩论后回答: {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")
            else:
                logger.info(f"最终回答 (无辩论): {final_resp[:300]}{'...' if len(final_resp) > 300 else ''}")

            complexity = result.get('task_complexity', -1.0)
            logger.info(f"估计的任务复杂度: {complexity:.4f}")

        # Restore original logger level
        logging.getLogger().setLevel(original_level)

    except Exception as e:
        # Ensure logger is at least WARNING to show exceptions
        logging.getLogger().setLevel(logging.WARNING)
        logger.exception(f"处理问题 '{instruction}' 时发生未捕获的异常: {e}")

if __name__ == "__main__":
    main() 