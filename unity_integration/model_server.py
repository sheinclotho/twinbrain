"""
Model Server Interface
======================

模型加载和推理接口，支持：
1. 加载训练好的模型
2. 接收虚拟刺激输入
3. 生成预测轨迹
4. 保存输出到数据文件夹
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class ModelServer:
    """
    模型服务器
    
    负责加载模型、处理推理请求、保存输出
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        output_dir: str = "unity_project/brain_data/model_output",
        device: str = "cpu"
    ):
        """
        初始化模型服务器
        
        Args:
            model_path: 训练模型文件路径
            output_dir: 输出目录
            device: 计算设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.model = None
        self.model_config = None
        self.logger = logging.getLogger(__name__)
        
        # 加载模型
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            是否加载成功
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            self.logger.info(f"加载模型: {model_path}")
            
            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 提取模型和配置
            if isinstance(checkpoint, dict):
                # 包含模型状态和元数据的checkpoint
                if 'model_state_dict' in checkpoint:
                    self.model = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    self.model = checkpoint
                
                # 提取配置信息
                if 'config' in checkpoint:
                    self.model_config = checkpoint['config']
                
                # 记录训练信息
                if 'epoch' in checkpoint:
                    self.logger.info(f"  - 训练轮数: {checkpoint['epoch']}")
                if 'best_loss' in checkpoint:
                    self.logger.info(f"  - 最佳损失: {checkpoint['best_loss']:.4f}")
            else:
                self.model = checkpoint
            
            self.logger.info("✓ 模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_future(
        self,
        initial_state: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        subject_id: str = "prediction"
    ) -> List[Dict[str, Any]]:
        """
        预测未来脑状态
        
        Args:
            initial_state: 初始状态 (n_regions, n_features)
            n_steps: 预测步数
            subject_id: 主体ID
        
        Returns:
            预测结果列表 (JSON格式)
        """
        self.logger.info(f"生成 {n_steps} 步预测...")
        
        n_regions = 200  # Schaefer 200
        
        # 如果没有初始状态，生成随机初始状态
        if initial_state is None:
            initial_state = torch.randn(n_regions, 1)
        
        predictions = []
        current_state = initial_state
        
        # 生成预测序列
        for t in range(n_steps):
            # 简单的演化模型（可替换为真实的神经网络预测）
            # 添加一些随机扰动和趋势
            noise = torch.randn_like(current_state) * 0.1
            drift = torch.sin(torch.tensor(t * 0.1)) * 0.05
            current_state = current_state * 0.95 + noise + drift
            
            # 转换为JSON格式
            brain_state = self._state_to_json(
                current_state,
                time_point=t,
                time_second=float(t * 0.5),  # 假设0.5秒/步
                subject_id=subject_id
            )
            
            predictions.append(brain_state)
        
        # 保存预测结果
        self._save_predictions(predictions, subject_id)
        
        self.logger.info(f"✓ 预测完成，生成 {len(predictions)} 个时间点")
        return predictions
    
    def simulate_stimulation(
        self,
        target_regions: List[int],
        amplitude: float = 0.5,
        pattern: str = "sine",
        frequency: float = 10.0,
        duration: int = 50,
        initial_state: Optional[torch.Tensor] = None,
        subject_id: str = "stimulation"
    ) -> List[Dict[str, Any]]:
        """
        模拟虚拟刺激响应
        
        Args:
            target_regions: 目标脑区ID列表
            amplitude: 刺激强度
            pattern: 刺激模式 (sine/pulse/ramp/constant)
            frequency: 频率 (Hz)
            duration: 持续时间步数
            initial_state: 初始状态
            subject_id: 主体ID
        
        Returns:
            模拟结果列表 (JSON格式)
        """
        self.logger.info(f"模拟虚拟刺激...")
        self.logger.info(f"  - 目标脑区: {target_regions}")
        self.logger.info(f"  - 刺激强度: {amplitude}")
        self.logger.info(f"  - 刺激模式: {pattern}")
        
        n_regions = 200
        
        # 初始状态
        if initial_state is None:
            initial_state = torch.randn(n_regions, 1) * 0.1
        
        # 生成刺激信号
        stimulation_signal = self._generate_stimulation_signal(
            n_steps=duration,
            pattern=pattern,
            amplitude=amplitude,
            frequency=frequency
        )
        
        results = []
        current_state = initial_state.clone()
        
        # 模拟时间演化
        for t in range(duration):
            # 应用刺激到目标脑区
            stim = torch.zeros(n_regions, 1)
            for region_id in target_regions:
                if 0 <= region_id < n_regions:
                    stim[region_id, 0] = stimulation_signal[t]
            
            # 简化的动力学模型
            # 真实模型应该使用训练好的神经网络
            noise = torch.randn_like(current_state) * 0.05
            current_state = current_state * 0.9 + stim * 0.3 + noise
            
            # 空间扩散效应（简化）
            if t % 5 == 0:  # 每5步应用一次扩散
                diffusion = torch.randn_like(current_state) * 0.02
                for region_id in target_regions:
                    # 影响邻近脑区
                    for neighbor in range(max(0, region_id-3), min(n_regions, region_id+4)):
                        if neighbor != region_id:
                            diffusion[neighbor, 0] += current_state[region_id, 0] * 0.1
                current_state = current_state + diffusion
            
            # 转换为JSON
            brain_state = self._state_to_json(
                current_state,
                time_point=t,
                time_second=float(t * 0.5),
                subject_id=subject_id,
                stimulation_info={
                    "target_regions": target_regions,
                    "amplitude": amplitude,
                    "pattern": pattern
                }
            )
            
            results.append(brain_state)
        
        # 保存结果
        self._save_predictions(results, subject_id)
        
        self.logger.info(f"✓ 刺激模拟完成，生成 {len(results)} 个时间点")
        return results
    
    def _generate_stimulation_signal(
        self,
        n_steps: int,
        pattern: str,
        amplitude: float,
        frequency: float
    ) -> np.ndarray:
        """
        生成刺激信号
        
        Args:
            n_steps: 步数
            pattern: 模式
            amplitude: 振幅
            frequency: 频率
        
        Returns:
            刺激信号 (n_steps,)
        """
        t = np.linspace(0, n_steps * 0.5, n_steps)  # 时间轴（秒）
        
        if pattern == "sine":
            # 正弦波
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        elif pattern == "pulse":
            # 脉冲
            pulse_interval = int(1.0 / frequency / 0.5)  # 转换为步数
            signal = np.zeros(n_steps)
            signal[::pulse_interval] = amplitude
        
        elif pattern == "ramp":
            # 渐变
            signal = amplitude * np.linspace(0, 1, n_steps)
        
        elif pattern == "constant":
            # 恒定
            signal = np.ones(n_steps) * amplitude
        
        else:
            # 默认恒定
            signal = np.ones(n_steps) * amplitude
        
        return signal
    
    def _state_to_json(
        self,
        state: torch.Tensor,
        time_point: int,
        time_second: float,
        subject_id: str,
        stimulation_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        将脑状态转换为JSON格式
        
        Args:
            state: 状态张量 (n_regions, n_features)
            time_point: 时间点
            time_second: 时间（秒）
            subject_id: 主体ID
            stimulation_info: 刺激信息（可选）
        
        Returns:
            JSON格式的脑状态
        """
        n_regions = state.shape[0]
        
        # 规范化活动值到 [0, 1]
        activity = state[:, 0].numpy() if state.is_cuda == False else state[:, 0].cpu().numpy()
        
        # 使用tanh进行软规范化
        activity_normalized = (np.tanh(activity) + 1) / 2
        
        # 构建脑区数据
        regions = []
        for i in range(n_regions):
            region = {
                "id": i,
                "label": f"Region_{i:03d}",
                "activity": float(activity_normalized[i]),
                "raw_activity": float(activity[i]),
                "network": self._get_network_name(i),  # 简化的网络分配
                "hemisphere": "left" if i < n_regions // 2 else "right"
            }
            regions.append(region)
        
        # 构建JSON结构
        brain_state = {
            "subject_id": subject_id,
            "time_point": time_point,
            "time_second": time_second,
            "timestamp": datetime.now().isoformat(),
            "n_regions": n_regions,
            "regions": regions
        }
        
        # 添加刺激信息（如果有）
        if stimulation_info:
            brain_state["stimulation"] = stimulation_info
        
        return brain_state
    
    def _get_network_name(self, region_id: int) -> str:
        """
        获取脑区所属网络名称（简化版本）
        
        Args:
            region_id: 脑区ID
        
        Returns:
            网络名称
        """
        # Schaefer 200的7个网络
        # 这是简化版本，真实分配需要从atlas获取
        network_size = 200 // 7
        network_id = region_id // network_size
        
        networks = [
            "Visual",
            "Somatomotor",
            "Dorsal Attention",
            "Ventral Attention",
            "Limbic",
            "Frontoparietal",
            "Default Mode"
        ]
        
        return networks[min(network_id, len(networks)-1)]
    
    def _save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        subject_id: str
    ):
        """
        保存预测结果到文件
        
        Args:
            predictions: 预测结果列表
            subject_id: 主体ID
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存每个时间点为独立文件
        for pred in predictions:
            time_point = pred['time_point']
            filename = f"{subject_id}_t{time_point:04d}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(pred, f, indent=2, ensure_ascii=False)
        
        # 保存元数据文件
        metadata = {
            "subject_id": subject_id,
            "timestamp": timestamp,
            "n_timepoints": len(predictions),
            "time_range": {
                "start": predictions[0]['time_point'],
                "end": predictions[-1]['time_point']
            },
            "files": [
                f"{subject_id}_t{p['time_point']:04d}_{timestamp}.json"
                for p in predictions
            ]
        }
        
        metadata_path = self.output_dir / f"{subject_id}_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ 保存到: {self.output_dir}/")
        self.logger.info(f"  - {len(predictions)} 个时间点文件")
        self.logger.info(f"  - 1 个元数据文件")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            "model_loaded": self.model is not None,
            "model_path": str(self.model_path) if self.model_path else None,
            "output_dir": str(self.output_dir),
            "device": self.device
        }
        
        if self.model_config:
            info["config"] = self.model_config
        
        return info
