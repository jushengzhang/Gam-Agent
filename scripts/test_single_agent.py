import argparse
import os
import sys

# 将项目根目录添加到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigLoader
from src.trainer.signle_agent_test import SingleAgentTest

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练视频语言多专家代理模型')
    parser.add_argument('--config', type=str, default='configs/MMbench_config_Choice.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='运行模式: 训练或评估')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    config_loader = ConfigLoader(config_path)
    
    # 初始化训练器
    trainer = SingleAgentTest(config_loader)
    
    # 根据模式运行
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.evaluate()

if __name__ == '__main__':
    main()