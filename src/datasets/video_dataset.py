import json
import torch
import av
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import base64
from PIL import Image
from PIL import ImageDraw

class VideoQuestionDataset(Dataset):
    """视频问答数据集加载类"""
    
    def __init__(self, json_file, video_dir=None, num_frames=8, image_size=224, video_extension='.mp4', whether_use_original_video=False):
        """
        初始化数据集
        
        Args:
            json_file: JSON数据文件路径
            video_dir: 视频文件目录路径，如果为None，则video_id应该是完整路径
            num_frames: 每个视频提取的帧数
            image_size: 图像大小（高度和宽度）
            video_extension: 视频文件扩展名
            whether_use_original_video: 是否使用原始视频
        """
        # 读取 JSON 文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_extension = video_extension
        self.whether_use_original_video = whether_use_original_video
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
        
    def _get_video_path(self, video_id):
        if self.video_dir is None:
            return os.path.abspath(video_id)

        # 根据前缀选择子目录
        if video_id.startswith('C'):
            subdir = "train_creative"
        elif video_id.startswith('H'):
            subdir = "train_humor"
        elif video_id.startswith('M'):
            subdir = "train_magic"
        else:
            subdir = ""  # 默认处理

        # 直接拼接子目录到根目录（无需 basename 处理）
        video_name = os.path.splitext(video_id)[0]  # 保留原始 video_id 的路径信息
        video_path = os.path.join(
            self.video_dir,
            subdir,
            f"{video_name}{self.video_extension}"
        )
        
        return os.path.abspath(video_path)

    
        #  Base 64 编码格式
    def encode_video(self, video_path):
        #这里在cpu上，为什么不使用gpu编码，因为还要传回来，会占用大量时间
        # 检查是否为压缩视频路径
        if "train_compressed" in video_path:
            if not os.path.exists(video_path):
                print(f"警告: {video_path} 视频未被压缩成功")
                return None
        
        with open(video_path, "rb") as video_file:
            print(f"加载成功video_path: {video_path}")
            return base64.b64encode(video_file.read()).decode("utf-8")

    def _load_video_frames(self, video_path):
        """
        加载视频帧，按照指定的帧数进行采样，并在每帧右上角标注采样时间。

        Args:
            video_path: 视频文件路径

        Returns:
            np.ndarray: 视频帧数组，形状为 [num_frames, height, width, 3]
        """
        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"警告：视频文件不存在: {video_path}，将使用随机帧")
            # 如果文件不存在，则默认返回指定数量的随机图像
            return np.random.rand(self.num_frames, self.image_size, self.image_size, 3)

        # 使用PyAV打开视频
        container = av.open(video_path)

        # 获取视频流信息
        video_stream = container.streams.video[0]
        # 计算视频时长（秒）
        duration = float(video_stream.duration * video_stream.time_base)
        # 目标采样频率
        target_fps = self.num_frames

        # 计算采样时间点
        timestamps = np.arange(0, duration, 1.0 / target_fps)

        frames = []
        # 对每个时间点进行采样
        for ts in timestamps:
            # 使用微秒为单位进行seek
            container.seek(int(ts * 1e6))
            for frame in container.decode(video=0):
                frame_np = frame.to_ndarray(format='rgb24')
                # 将 NumPy 数组转换为 PIL 图像
                frame_pil = Image.fromarray(frame_np)
                draw = ImageDraw.Draw(frame_pil)
                # 在右上角添加时间戳文本,ts保留两位小数
                draw.text((frame_pil.width - 100, 10), f"{ts:.2f}s", fill=(255, 255, 255))  # 白色文本

                # 将 PIL 图像转换回 NumPy 数组
                frame_np = np.array(frame_pil)
                frames.append(frame_np)
                break  # 每个时间点只采样一帧

        # 将采样的帧列表转换为numpy数组
        frames = np.stack(frames)
        container.close()

        return frames


    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            tuple: (指令, 视频帧, 视频ID, 输出, 任务类型, 视频路径)
        """
        # 获取数据
        item = self.data[idx]
        instruction = item['instruction']
        video_id = item['visual_input']
        output = item['output']
        task = item['task']
        options = item['options']
        choice_answer = item['choice_answer']
        
        # 获取视频路径并加载视频帧
        video_path = self._get_video_path(video_id)
        if self.whether_use_original_video is True:
            video_frames = self.encode_video(video_path) # 使用Base64编码的视频
        else:
            video_frames = self._load_video_frames(video_path) # 使用PyAV解码的视频帧
            
        whether_use_original_video = self.whether_use_original_video
        
        return instruction, video_frames, video_id, output, task, options, choice_answer, whether_use_original_video, video_path

def read_video_pyav(container, indices):
    """
    使用PyAV读取视频帧
    
    Args:
        container: PyAV视频容器
        indices: 要读取的帧索引
        
    Returns:
        np.ndarray: 视频帧数组
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

def collate_fn(batch):
    """
    数据批次整理函数
    
    Args:
        batch: 数据批次
        
    Returns:
        dict: 整理后的批次数据
    """
    # 将 batch 转换为字典
    instructions, video_frames, video_ids, outputs, tasks, options, choice_answer, whether_use_original_video, video_paths = zip(*batch)
    
    if whether_use_original_video is False:
        # 将视频帧转换为张量
        video_frames = [torch.from_numpy(frames).float() for frames in video_frames]
    else:
        video_frames = video_frames
    
    return {
        'instruction': instructions,
        'video_frames': video_frames,
        'video_id': video_ids,
        'output': outputs,
        'task': tasks,
        'options': options,
        'choice_answer': choice_answer,
        'whether_use_original_video': whether_use_original_video,
        'video_path': video_paths
    }

def create_dataloader(config):
    """
    创建数据加载器
    
    Args:
        config: 数据集配置
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    # 从配置中获取视频目录（可选）
    video_dir = config.get('video_dir', None)
    num_frames = config.get('num_frames', 8)
    image_size = config.get('image_size', 224)
    whether_use_original_video = config.get('use_original_video', False)
    
    # print(f"use_original_video: {whether_use_original_video}")
    # print(f"config: {config}")
    
    dataset = VideoQuestionDataset(
        config['json_file'],
        video_dir=video_dir,
        num_frames=num_frames,
        image_size=image_size,
        whether_use_original_video=whether_use_original_video
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=config['shuffle'], 
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    return dataloader 