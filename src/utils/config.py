import os
import yaml
import torch

class ConfigLoader:
    """配置加载器，用于从YAML文件中加载配置"""
    
    def __init__(self, config_path):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.config = self._load_config()
        
    def _load_config(self):
        """
        加载配置文件
        
        Returns:
            dict: 配置字典
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get_dataset_config(self):
        """获取数据集配置"""
        return self.config.get('dataset', {})
    
    def get_model_config(self):
        """获取模型配置"""
        model_config = self.config.get('model', {})
        
        # 处理特殊配置项，如torch.dtype
        if 'torch_dtype' in model_config:
            if model_config['torch_dtype'] == 'float16':
                model_config['torch_dtype'] = torch.float16
            elif model_config['torch_dtype'] == 'float32':
                model_config['torch_dtype'] = torch.float32
                
        return model_config
    
    def get_api_config(self):
        """获取API配置"""
        return self.config.get('api', {})
    
    def get_experts_config(self):
        """获取专家配置"""
        experts_config = self.config.get('experts', {})
        
        # 处理嵌套的专家配置文件
        if 'config_file' in experts_config:
            config_file = experts_config['config_file']
            # 如果是相对路径，相对于主配置文件目录
            if not os.path.isabs(config_file):
                # 先获取父目录路径
                parent_dir = os.path.dirname(self.base_dir)
                config_file = os.path.join(parent_dir, config_file)
                
            # 确保文件存在
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"专家配置文件不存在: {config_file}")
                
            # 加载专家配置
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    nested_config = yaml.safe_load(f)
                    
                # 合并配置
                experts_config['settings'] = nested_config.get('settings', {})
                experts_config['experts'] = nested_config.get('experts', [])
                experts_config['combinations'] = nested_config.get('combinations', [])
                
            except Exception as e:
                raise RuntimeError(f"加载专家配置文件失败: {e}")
                
        return experts_config
    
    def get_metrics_config(self):
        """获取评估指标配置"""
        return self.config.get('metrics', {})
    
    def get_training_config(self):
        """获取训练配置"""
        return self.config.get('training', {}) 