import os
import sys
import argparse
import logging
import yaml
import torch
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在导入应该能正常工作
from src.image_loader import ImageLoader
from src.models.agent_model import ExpertAgentModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/image_anomaly_detection.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="AI生成图像异常检测")
    parser.add_argument("--config", type=str, default="configs/image_anomaly_experts_config.yaml", help="专家配置文件路径")
    parser.add_argument("--model_config", type=str, default="configs/MMbench_config_Choice.yaml", help="模型配置文件路径")
    parser.add_argument("--image_path", type=str, help="单张图像路径")
    parser.add_argument("--image_dir", type=str, default="data/test_img", help="图像目录路径")
    parser.add_argument("--output_dir", type=str, default="results/anomaly_detection", help="结果输出目录")
    parser.add_argument("--image_size", type=int, default=1024, help="调整图像大小")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--use_debate", action="store_true", help="启用辩论功能")
    parser.add_argument("--api_key", type=str, help="API密钥，覆盖配置文件")
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    logger.info(f"加载模型配置: {args.model_config}")
    model_config = load_config(args.model_config)
    
    print(f"model_config: {model_config}")
    
    # 加载专家配置
    logger.info(f"加载专家配置: {args.model_config}")
    experts_config = load_config(args.model_config)
    
    # 如果提供了API密钥，则覆盖配置
    if args.api_key:
        model_config["model"]["api_key"] = args.api_key
    
    # 初始化图像加载器
    image_loader = ImageLoader(image_size=args.image_size)
    
    # 初始化智能体模型
    logger.info("初始化专家代理模型")
    agent_model = ExpertAgentModel(
        model_config=model_config["model"],
        experts_config=model_config["experts"]
    )
    
    # 加载图像
    if args.image_path:
        logger.info(f"加载单张图像: {args.image_path}")
        image = image_loader.load_single_image(args.image_path)
        images = [{
            "path": args.image_path,
            "filename": os.path.basename(args.image_path),
            "image": image
        }]
    else:
        logger.info(f"从目录加载图像: {args.image_dir}")
        images = image_loader.load_multiple_images(image_dir=args.image_dir)
    
    if not images:
        logger.error("未找到任何图像，程序退出")
        return
    
    logger.info(f"成功加载 {len(images)} 张图像")
    
    # 处理每张图像
    for i, img_data in enumerate(images):
        image = img_data["image"]
        filename = img_data["filename"]
        logger.info(f"处理图像 {i+1}/{len(images)}: {filename}")
        
        # 创建处理指令
        instruction = "这张图片是AI生成的吗？请详细分析所有不合理、违背逻辑的异常特征。"
        
        # 使用专家代理处理图像
        try:
            # 将单个图像转换为视频帧格式（添加一个帧维度）
            # 确保image是numpy数组
            if not isinstance(image, np.ndarray):
                logger.error(f"图像不是numpy数组，而是{type(image)}")
                continue
                
            # 将单图像扩展为单帧视频格式：[1, height, width, channels]
            video_frame = np.expand_dims(image, axis=0)
            
            response = agent_model.process_with_experts(
                video_frame,  # 正确格式的视频帧
                instruction=instruction,
                output=None  # 没有参考输出
            )
            
            # 输出结果
            result_path = os.path.join(args.output_dir, f"{Path(filename).stem}_result.txt")
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"图像: {filename}\n")
                f.write(f"指令: {instruction}\n\n")
                f.write(f"分析结果:\n{response['text']}\n")
                
                # 如果有辩论过程，也保存
                if "debate_process" in response:
                    f.write("\n辩论过程:\n")
                    for i, round_data in enumerate(response["debate_process"]):
                        f.write(f"\n--- 轮次 {i+1} ---\n")
                        f.write(f"批评: {round_data['critique']}\n")
                        f.write(f"修订: {round_data['revision']}\n")
            
            logger.info(f"结果已保存到: {result_path}")
            
        except Exception as e:
            logger.error(f"处理图像 {filename} 时出错: {e}")
            continue
    
    logger.info("所有图像处理完成")

if __name__ == "__main__":
    main() 