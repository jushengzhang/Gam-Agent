import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    图片加载器：支持单图和多图的加载和处理
    """
    def __init__(self, image_size=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化图片加载器
        
        Args:
            image_size: 调整图片大小的目标尺寸，为None则保持原始大小
            device: 处理设备
        """
        self.image_size = image_size
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"初始化图片加载器，设备：{device}")
    
    def load_single_image(self, image_path):
        """
        加载单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            处理后的图片（numpy数组）
        """
        try:
            self.logger.info(f"正在加载图片: {image_path}")
            image = Image.open(image_path).convert("RGB")
            
            # 如果指定了大小，则调整图片大小
            if self.image_size:
                image = self._resize_image(image)
                
            # 转换为numpy数组
            image_array = np.array(image)
            self.logger.info(f"成功加载图片，形状: {image_array.shape}")
            
            return image_array
        except Exception as e:
            self.logger.error(f"加载图片失败: {e}")
            raise
    
    def load_multiple_images(self, image_dir=None, image_paths=None, file_extensions=["jpg", "jpeg", "png"]):
        """
        加载多张图片
        
        Args:
            image_dir: 图片目录（与image_paths二选一）
            image_paths: 图片路径列表（与image_dir二选一）
            file_extensions: 要加载的文件扩展名列表
            
        Returns:
            处理后的图片列表（numpy数组）
        """
        if image_dir and image_paths:
            raise ValueError("不能同时指定image_dir和image_paths")
        
        if image_dir:
            # 获取目录中的所有图片
            image_paths = []
            for ext in file_extensions:
                pattern = os.path.join(image_dir, f"*.{ext}")
                image_paths.extend(glob.glob(pattern))
                pattern = os.path.join(image_dir, f"*.{ext.upper()}")
                image_paths.extend(glob.glob(pattern))
        
        if not image_paths:
            self.logger.warning("未找到任何图片")
            return []
        
        self.logger.info(f"找到 {len(image_paths)} 张图片")
        
        # 加载所有图片
        images = []
        for path in image_paths:
            try:
                image = self.load_single_image(path)
                images.append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "image": image
                })
            except Exception as e:
                self.logger.error(f"加载图片 {path} 失败: {e}")
                continue
        
        return images
    
    def _resize_image(self, image):
        """
        调整图片大小，保持纵横比
        
        Args:
            image: PIL图片对象
            
        Returns:
            调整大小后的图片
        """
        if isinstance(self.image_size, int):
            # 等比例缩放到较长边为image_size
            width, height = image.size
            if width > height:
                new_width = self.image_size
                new_height = int(height * (self.image_size / width))
            else:
                new_height = self.image_size
                new_width = int(width * (self.image_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif isinstance(self.image_size, tuple) and len(self.image_size) == 2:
            # 直接调整到指定大小
            return image.resize(self.image_size, Image.Resampling.LANCZOS)
        else:
            return image
    
    def batch_process_images(self, images, batch_size=4):
        """
        批量处理图片
        
        Args:
            images: 图片列表（numpy数组）
            batch_size: 批处理大小
            
        Returns:
            批处理后的图片张量
        """
        if not images:
            return []
        
        batches = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # 转换为张量
            batch_tensor = torch.stack([torch.from_numpy(img["image"]).float() for img in batch])
            # 移动到指定设备
            batch_tensor = batch_tensor.to(self.device)
            batches.append({
                "tensor": batch_tensor,
                "paths": [img["path"] for img in batch],
                "filenames": [img["filename"] for img in batch]
            })
        
        return batches 