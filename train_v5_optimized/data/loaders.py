"""
数据加载器 - Data Loaders
=======================

负责加载和预处理EEG和fMRI数据
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import mne
import nibabel as nib

from .eeg_preprocessor import EEGPreprocessor
from .fmri_preprocessor import FMRI_Preprocessor

logger = logging.getLogger(__name__)


class BrainDataLoader:
    """
    统一的脑数据加载器
    
    支持:
    - EEG数据加载和预处理
    - fMRI数据加载和预处理
    - 自动检测BIDS格式
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        modalities: List[str] = ['eeg', 'fmri'],
    ):
        """
        初始化数据加载器
        
        Args:
            data_root: 数据根目录
            modalities: 要加载的模态 ['eeg', 'fmri']
        """
        self.data_root = Path(data_root)
        self.modalities = modalities
        
        # 初始化预处理器
        if 'eeg' in modalities:
            self.eeg_preprocessor = EEGPreprocessor()
        
        if 'fmri' in modalities:
            self.fmri_preprocessor = FMRI_Preprocessor()
        
        logger.info(f"初始化数据加载器: {data_root}, 模态: {modalities}")
    
    def load_subject(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        加载单个被试数据
        
        Args:
            subject_id: 被试ID (e.g., 'sub-01')
            task: 任务名称 (e.g., 'rest', 'task')
            
        Returns:
            包含各模态数据的字典
        """
        data = {'subject_id': subject_id}
        
        # 加载EEG
        if 'eeg' in self.modalities:
            eeg_data = self._load_eeg(subject_id, task)
            if eeg_data is not None:
                data['eeg'] = eeg_data
        
        # 加载fMRI
        if 'fmri' in self.modalities:
            fmri_data = self._load_fmri(subject_id, task)
            if fmri_data is not None:
                data['fmri'] = fmri_data
        
        return data
    
    def _load_eeg(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Optional[Dict[str, any]]:
        """加载EEG数据"""
        try:
            # 查找EEG文件
            eeg_pattern = f"{subject_id}*eeg.set"
            if task:
                eeg_pattern = f"{subject_id}*task-{task}*eeg.set"
            
            eeg_files = list(self.data_root.glob(f"**/{eeg_pattern}"))
            
            if not eeg_files:
                logger.warning(f"未找到EEG文件: {subject_id}")
                return None
            
            eeg_file = eeg_files[0]
            logger.info(f"加载EEG: {eeg_file.name}")
            
            # 加载原始EEG
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True)
            
            # 预处理
            raw = self.eeg_preprocessor.preprocess(raw)
            
            # 提取数据
            data, times = raw.get_data(return_times=True)
            
            return {
                'data': data,  # [n_channels, n_times]
                'times': times,
                'ch_names': raw.ch_names,
                'sfreq': raw.info['sfreq'],
            }
            
        except Exception as e:
            logger.error(f"加载EEG失败 {subject_id}: {e}")
            return None
    
    def _load_fmri(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Optional[Dict[str, any]]:
        """加载fMRI数据"""
        try:
            # 查找fMRI文件
            fmri_pattern = f"{subject_id}*bold.nii*"
            if task:
                fmri_pattern = f"{subject_id}*task-{task}*bold.nii*"
            
            fmri_files = list(self.data_root.glob(f"**/{fmri_pattern}"))
            
            if not fmri_files:
                logger.warning(f"未找到fMRI文件: {subject_id}")
                return None
            
            fmri_file = fmri_files[0]
            logger.info(f"加载fMRI: {fmri_file.name}")
            
            # 加载和预处理
            img = nib.load(str(fmri_file))
            img = self.fmri_preprocessor.preprocess(img)
            
            return {
                'img': img,
                'data': img.get_fdata(),
                'affine': img.affine,
            }
            
        except Exception as e:
            logger.error(f"加载fMRI失败 {subject_id}: {e}")
            return None
    
    def load_all_subjects(
        self,
        task: Optional[str] = None,
        max_subjects: Optional[int] = None,
    ) -> List[Dict[str, any]]:
        """
        加载所有被试数据
        
        Args:
            task: 任务名称
            max_subjects: 最大被试数
            
        Returns:
            被试数据列表
        """
        # 查找所有被试
        subject_dirs = sorted(self.data_root.glob("sub-*"))
        
        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]
        
        all_data = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            data = self.load_subject(subject_id, task)
            
            if data:
                all_data.append(data)
        
        logger.info(f"成功加载 {len(all_data)} 个被试")
        return all_data
