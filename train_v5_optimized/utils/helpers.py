"""
工具函数
=======

通用辅助函数
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import yaml


def setup_logging(
    log_file: Path = None,
    level: str = 'INFO',
) -> logging.Logger:
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logger对象
    """
    # 创建logger
    logger = logging.getLogger('twinbrain_v5')
    logger.setLevel(getattr(logging, level))
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_dir(
    base_dir: str,
    experiment_name: str,
) -> Path:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    Returns:
        输出目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    return output_dir


def save_config(config: dict, save_path: Path):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu',
) -> int:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        device: 设备
        
    Returns:
        epoch: 当前epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    return epoch


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
