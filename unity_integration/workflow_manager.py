"""
Unity 工作流管理器
==================

统一管理从数据下载到Unity导入的完整工作流：
1. 数据下载（可选）
2. 数据预处理
3. 模型推理
4. 导出为多种格式（JSON, OBJ）
5. 生成Unity配置
6. 支持FreeSurfer surface文件导入
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from .brain_state_exporter import BrainStateExporter
from .obj_generator import BrainOBJGenerator
from .freesurfer_loader import FreeSurferLoader, load_freesurfer_data


@dataclass
class WorkflowConfig:
    """工作流配置"""
    # 输入数据路径
    data_source: str = "local"  # 'local', 'download', 'model', 'example', 'freesurfer'
    data_path: Optional[str] = None
    
    # FreeSurfer specific paths (for data_source='freesurfer')
    freesurfer_lh_surface: Optional[str] = None  # Path to lh.pial
    freesurfer_rh_surface: Optional[str] = None  # Path to rh.pial
    freesurfer_lh_annot: Optional[str] = None    # Path to lh annotation file
    freesurfer_rh_annot: Optional[str] = None    # Path to rh annotation file
    
    # 输出配置
    output_dir: str = "output/unity_export"
    export_formats: List[str] = None  # ['json', 'obj', 'fbx']
    
    # 时间范围
    start_time: int = 0
    end_time: Optional[int] = None
    time_step: int = 5
    
    # 导出选项
    export_connectivity: bool = True
    export_networks: bool = True
    export_obj_per_frame: bool = False  # 每帧导出独立OBJ
    export_surface_mesh: bool = False   # Export FreeSurfer surface mesh (if available)
    
    # Unity配置
    generate_unity_config: bool = True
    generate_materials: bool = True
    
    # 主体信息
    subject_id: str = "unknown"
    atlas_name: str = "Schaefer200"
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json']


class WorkflowManager:
    """
    Unity工作流管理器
    
    提供完整的自动化流程，从数据准备到Unity可视化。
    """
    
    # 活动强度归一化常量
    # 假设数据在标准化后大致在 [-3, 3] 范围内
    # 映射到 [0, 1] 用于可视化
    ACTIVITY_NORMALIZATION_MIN = -3.0
    ACTIVITY_NORMALIZATION_MAX = 3.0
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        atlas_info: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None
    ):
        """
        初始化工作流管理器
        
        Args:
            config: 工作流配置
            atlas_info: 脑图谱信息
            model: 训练好的模型（可选）
        """
        self.config = config or WorkflowConfig()
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.freesurfer_loader = None  # Will be initialized if using FreeSurfer data
        
        # Load or generate atlas info based on data source
        if self.config.data_source == 'freesurfer' and atlas_info is None:
            # Load atlas from FreeSurfer files
            self.atlas_info = self._load_freesurfer_atlas()
        else:
            self.atlas_info = atlas_info or self._load_default_atlas()
        
        # 创建导出器
        self.exporter = BrainStateExporter(
            atlas_info=self.atlas_info,
            model_version="v4"
        )
        
        # 创建OBJ生成器
        self.obj_generator = BrainOBJGenerator(
            atlas_info=self.atlas_info,
            sphere_resolution=16  # 16x16 sphere mesh
        )
        
        # 输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_workflow(self) -> Dict[str, Any]:
        """
        运行完整工作流
        
        Returns:
            包含工作流结果的字典
        """
        self.logger.info("开始Unity工作流...")
        results = {
            'success': False,
            'steps_completed': [],
            'output_files': [],
            'errors': []
        }
        
        try:
            # 步骤1: 加载或下载数据
            brain_data, connectivity = self._step_load_data()
            results['steps_completed'].append('load_data')
            
            # 步骤2: 导出JSON格式
            if 'json' in self.config.export_formats:
                json_files = self._step_export_json(brain_data, connectivity)
                results['output_files'].extend(json_files)
                results['steps_completed'].append('export_json')
            
            # 步骤3: 导出OBJ格式
            if 'obj' in self.config.export_formats:
                obj_files = self._step_export_obj(brain_data)
                results['output_files'].extend(obj_files)
                results['steps_completed'].append('export_obj')
            
            # 步骤3.5: 导出FreeSurfer surface mesh（如果有）
            if self.config.export_surface_mesh and self.freesurfer_loader is not None:
                surface_files = self._step_export_surface_mesh()
                results['output_files'].extend(surface_files)
                results['steps_completed'].append('export_surface_mesh')
            
            # 步骤4: 生成Unity配置
            if self.config.generate_unity_config:
                config_file = self._step_generate_unity_config()
                results['output_files'].append(config_file)
                results['steps_completed'].append('generate_config')
            
            # 步骤5: 生成材质配置
            if self.config.generate_materials:
                material_files = self._step_generate_materials()
                results['output_files'].extend(material_files)
                results['steps_completed'].append('generate_materials')
            
            results['success'] = True
            self.logger.info(f"工作流完成！生成了 {len(results['output_files'])} 个文件")
            
        except Exception as e:
            self.logger.error(f"工作流失败: {e}")
            results['errors'].append(str(e))
            raise
        
        # 保存工作流报告
        report_path = self.output_dir / "workflow_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _step_load_data(self) -> tuple:
        """步骤1: 加载数据"""
        self.logger.info("步骤1: 加载数据...")
        
        if self.config.data_source == 'local':
            return self._load_local_data()
        elif self.config.data_source == 'download':
            return self._download_and_process_data()
        elif self.config.data_source == 'model':
            return self._generate_from_model()
        elif self.config.data_source == 'example':
            return self._generate_example_data()
        elif self.config.data_source == 'freesurfer':
            return self._load_freesurfer_data()
        else:
            # Default to example data for any unknown source
            self.logger.warning(f"未知的数据源 '{self.config.data_source}'，使用示例数据")
            return self._generate_example_data()
    
    def _step_export_json(
        self,
        brain_data: Dict[str, torch.Tensor],
        connectivity: Optional[Dict[str, np.ndarray]]
    ) -> List[str]:
        """步骤2: 导出JSON格式"""
        self.logger.info("步骤2: 导出JSON...")
        
        json_dir = self.output_dir / "json"
        json_dir.mkdir(exist_ok=True)
        
        # 导出序列
        self.exporter.export_sequence(
            brain_activity=brain_data,
            output_dir=json_dir,
            start=self.config.start_time,
            end=self.config.end_time,
            step=self.config.time_step,
            connectivity=connectivity if self.config.export_connectivity else None,
            subject_id=self.config.subject_id
        )
        
        # 收集生成的文件
        json_files = list(json_dir.glob("*.json"))
        return [str(f.relative_to(self.output_dir)) for f in json_files]
    
    def _step_export_obj(self, brain_data: Dict[str, torch.Tensor]) -> List[str]:
        """步骤3: 导出OBJ格式"""
        self.logger.info("步骤3: 导出OBJ...")
        
        obj_dir = self.output_dir / "obj"
        obj_dir.mkdir(exist_ok=True)
        
        obj_files = []
        
        # 获取数据维度
        fmri_data = brain_data.get('fmri')
        if fmri_data is None:
            self.logger.warning("没有fMRI数据，跳过OBJ导出")
            return obj_files
        
        n_regions, n_timepoints, _ = fmri_data.shape
        end_time = self.config.end_time or n_timepoints
        
        if self.config.export_obj_per_frame:
            # 每帧导出独立OBJ
            self.logger.info(f"  导出时间序列OBJ模型...")
            self.obj_generator.export_brain_sequence(
                output_dir=obj_dir,
                activity_sequence=fmri_data,
                start=self.config.start_time,
                end=end_time,
                step=self.config.time_step
            )
            obj_files.extend([str(f.relative_to(self.output_dir)) for f in obj_dir.glob("brain_t*.obj")])
        else:
            # 导出单个聚合OBJ（使用平均活动）
            self.logger.info(f"  导出聚合OBJ模型...")
            mean_activity = fmri_data.mean(dim=1)  # 平均所有时间点
            self.obj_generator.export_brain_model(
                output_path=obj_dir / "brain_regions.obj",
                activity_data=mean_activity
            )
            obj_files.append("obj/brain_regions.obj")
        
        self.logger.info(f"  ✓ 生成了 {len(obj_files)} 个OBJ文件")
        return obj_files
    
    def _step_generate_unity_config(self) -> str:
        """步骤4: 生成Unity配置"""
        self.logger.info("步骤4: 生成Unity配置...")
        
        config_path = self.output_dir / "unity_config.json"
        
        # 生成配置
        unity_config = {
            "project_name": f"TwinBrain_{self.config.subject_id}",
            "atlas": self.config.atlas_name,
            "data_paths": {
                "json_dir": str(Path("json")),  # 相对于输出目录的路径
                "obj_dir": str(Path("obj")),
                "materials_dir": str(Path("materials"))
            },
            "visualization": {
                "region_scale": 1.0,
                "activity_threshold": 0.3,
                "connection_threshold": 0.5,
                "show_connections": self.config.export_connectivity,
                "fps": 10,
                "auto_play": True
            },
            "colors": {
                "low_activity": {"r": 0, "g": 0, "b": 255},
                "high_activity": {"r": 255, "g": 0, "b": 0},
                "connection_structural": {"r": 255, "g": 255, "b": 255, "a": 128},
                "connection_functional": {"r": 255, "g": 255, "b": 0, "a": 128}
            },
            "animation": {
                "start_frame": self.config.start_time,
                "end_frame": self.config.end_time,
                "frame_step": self.config.time_step
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(unity_config, f, indent=2, ensure_ascii=False)
        
        return str(config_path.relative_to(self.output_dir))
    
    def _step_generate_materials(self) -> List[str]:
        """步骤5: 生成材质配置"""
        self.logger.info("步骤5: 生成材质配置...")
        
        materials_dir = self.output_dir / "materials"
        materials_dir.mkdir(exist_ok=True)
        
        material_files = []
        
        # 生成区域材质
        region_material = self._create_region_material_config()
        region_path = materials_dir / "RegionMaterial.json"
        with open(region_path, 'w', encoding='utf-8') as f:
            json.dump(region_material, f, indent=2)
        material_files.append(str(region_path.relative_to(self.output_dir)))
        
        # 生成连接材质
        if self.config.export_connectivity:
            connection_material = self._create_connection_material_config()
            conn_path = materials_dir / "ConnectionMaterial.json"
            with open(conn_path, 'w', encoding='utf-8') as f:
                json.dump(connection_material, f, indent=2)
            material_files.append(str(conn_path.relative_to(self.output_dir)))
        
        return material_files
    
    def _load_local_data(self) -> tuple:
        """从本地文件加载数据"""
        if not self.config.data_path:
            raise ValueError("本地数据源需要指定 data_path")
        
        # 这里应该实现实际的数据加载逻辑
        # 暂时返回示例数据
        self.logger.warning("使用示例数据（本地加载未完全实现）")
        return self._generate_example_data()
    
    def _download_and_process_data(self) -> tuple:
        """下载并处理数据"""
        # 这里应该实现数据下载逻辑
        # 例如从 OpenNeuro, Human Connectome Project 等下载
        self.logger.warning("数据下载功能未完全实现，使用示例数据")
        return self._generate_example_data()
    
    def _generate_from_model(self) -> tuple:
        """使用模型生成数据"""
        if self.model is None:
            raise ValueError("需要提供训练好的模型")
        
        # 这里应该实现使用模型生成预测的逻辑
        self.logger.warning("模型生成功能未完全实现，使用示例数据")
        return self._generate_example_data()
    
    def _generate_example_data(self) -> tuple:
        """生成示例数据"""
        n_regions = 200
        n_timepoints = 200
        n_features = 1
        
        # 生成示例活动数据
        fmri_data = torch.randn(n_regions, n_timepoints, n_features)
        eeg_data = torch.randn(n_regions, n_timepoints, n_features)
        
        brain_data = {
            'fmri': fmri_data,
            'eeg': eeg_data
        }
        
        # 生成示例连接矩阵
        connectivity_matrix = np.random.rand(n_regions, n_regions)
        connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
        connectivity_matrix[connectivity_matrix < 0.7] = 0
        
        connectivity = {
            'structural': connectivity_matrix
        }
        
        return brain_data, connectivity
    
    def _export_single_obj(self, activity_data: torch.Tensor, output_path: Path):
        """导出单个OBJ文件"""
        with open(output_path, 'w') as f:
            f.write("# TwinBrain Brain Regions OBJ Export\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Atlas: {self.config.atlas_name}\n\n")
            
            # 写入顶点（每个脑区一个球体）
            vertex_offset = 1
            for region_id in range(activity_data.shape[0]):
                region_info = self.atlas_info['regions'].get(str(region_id + 1), {})
                xyz = region_info.get('xyz', [0, 0, 0])
                
                # 获取活动强度
                if len(activity_data.shape) == 1:
                    activity = activity_data[region_id].item()
                else:
                    activity = activity_data[region_id].mean().item()
                
                # 归一化活动强度到 [0, 1]
                activity_norm = (activity - self.ACTIVITY_NORMALIZATION_MIN) / \
                               (self.ACTIVITY_NORMALIZATION_MAX - self.ACTIVITY_NORMALIZATION_MIN)
                activity_norm = max(0.0, min(1.0, activity_norm))
                
                # 球体半径基于活动强度
                radius = 2.0 + activity_norm * 3.0
                
                # 简化：用一个点代表脑区（实际应该生成球体网格）
                f.write(f"v {xyz[0]} {xyz[1]} {xyz[2]}\n")
                
                # 写入活动强度作为注释
                f.write(f"# Region {region_id}: activity={activity_norm:.3f}\n")
    
    def _create_region_material_config(self) -> Dict[str, Any]:
        """创建脑区材质配置"""
        return {
            "name": "RegionMaterial",
            "type": "Standard",
            "shader": "Standard",
            "properties": {
                "metallic": 0.0,
                "smoothness": 0.5,
                "emission": {
                    "enabled": True,
                    "color": {"r": 1.0, "g": 1.0, "b": 1.0},
                    "intensity": 0.2
                },
                "renderingMode": "Opaque"
            },
            "color_mapping": {
                "based_on": "activity",
                "gradient": [
                    {"value": 0.0, "color": {"r": 0, "g": 0, "b": 255}},
                    {"value": 0.5, "color": {"r": 0, "g": 255, "b": 0}},
                    {"value": 1.0, "color": {"r": 255, "g": 0, "b": 0}}
                ]
            }
        }
    
    def _create_connection_material_config(self) -> Dict[str, Any]:
        """创建连接材质配置"""
        return {
            "name": "ConnectionMaterial",
            "type": "Line",
            "shader": "Particles/Standard Unlit",
            "properties": {
                "renderingMode": "Fade",
                "blendMode": "Alpha",
                "width": 0.02
            },
            "color_mapping": {
                "structural": {"r": 255, "g": 255, "b": 255, "a": 128},
                "functional": {"r": 255, "g": 255, "b": 0, "a": 128}
            }
        }
    
    def _load_default_atlas(self) -> Dict[str, Any]:
        """加载默认脑图谱信息"""
        # 生成默认的Schaefer 200区域图谱
        atlas_info = {
            'name': 'Schaefer200',
            'n_regions': 200,
            'regions': {}
        }
        
        for i in range(200):
            atlas_info['regions'][str(i + 1)] = {
                'label': f'Region_{i+1}',
                'xyz': [
                    np.random.uniform(-80, 80),
                    np.random.uniform(-100, 80),
                    np.random.uniform(-60, 80)
                ],
                'network': self._assign_network(i)
            }
        
        return atlas_info
    
    def _load_freesurfer_atlas(self) -> Dict[str, Any]:
        """从FreeSurfer文件加载脑图谱信息"""
        if not all([
            self.config.freesurfer_lh_surface,
            self.config.freesurfer_rh_surface,
            self.config.freesurfer_lh_annot,
            self.config.freesurfer_rh_annot
        ]):
            raise ValueError(
                "使用FreeSurfer数据源时，必须提供所有FreeSurfer文件路径：\n"
                "- freesurfer_lh_surface (lh.pial)\n"
                "- freesurfer_rh_surface (rh.pial)\n"
                "- freesurfer_lh_annot (lh annotation file)\n"
                "- freesurfer_rh_annot (rh annotation file)"
            )
        
        self.logger.info("从FreeSurfer文件加载图谱...")
        
        # Use the convenience function to load FreeSurfer data
        atlas_info, loader = load_freesurfer_data(
            lh_surface=self.config.freesurfer_lh_surface,
            rh_surface=self.config.freesurfer_rh_surface,
            lh_annot=self.config.freesurfer_lh_annot,
            rh_annot=self.config.freesurfer_rh_annot,
            atlas_name=self.config.atlas_name
        )
        
        # Store the loader for later use (e.g., for exporting surface mesh)
        self.freesurfer_loader = loader
        
        self.logger.info(f"✓ 从FreeSurfer加载了 {atlas_info['n_regions']} 个脑区")
        return atlas_info
    
    def _load_freesurfer_data(self) -> tuple:
        """加载FreeSurfer数据并生成示例活动数据"""
        # Atlas info should already be loaded during __init__
        if self.atlas_info is None or self.freesurfer_loader is None:
            raise ValueError("FreeSurfer atlas not loaded properly")
        
        n_regions = self.atlas_info['n_regions']
        n_timepoints = self.config.end_time or 200
        n_features = 1
        
        # NOTE: This generates PLACEHOLDER activity data for demonstration
        # In real usage, you should replace this with actual brain activity data:
        # 
        # Example for loading real data:
        # -----------------------------
        # fmri_data = load_your_fmri_data()  # Shape: [n_regions, n_timepoints, n_features]
        # eeg_data = load_your_eeg_data()    # Shape: [n_regions, n_timepoints, n_features]
        # 
        # Data format requirements:
        # - fmri_data: torch.Tensor, shape [n_regions, n_timepoints, features]
        # - eeg_data: torch.Tensor, shape [n_regions, n_timepoints, features]
        # - Values should be normalized (e.g., z-scored)
        # - n_regions must match the number of regions in FreeSurfer atlas
        # 
        self.logger.info("生成示例活动数据（实际使用时应加载真实数据）...")
        self.logger.info(f"  需要加载: [{n_regions} regions, {n_timepoints} timepoints, {n_features} features]")
        fmri_data = torch.randn(n_regions, n_timepoints, n_features)
        eeg_data = torch.randn(n_regions, n_timepoints, n_features)
        
        brain_data = {
            'fmri': fmri_data,
            'eeg': eeg_data
        }
        
        # Generate example connectivity matrix
        # In real usage, load actual structural or functional connectivity:
        # connectivity_matrix = load_your_connectivity_data()  # Shape: [n_regions, n_regions]
        connectivity_matrix = np.random.rand(n_regions, n_regions)
        connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
        # Threshold at 0.7 to keep only strong connections (adjust based on your data)
        connectivity_matrix[connectivity_matrix < 0.7] = 0
        
        connectivity = {
            'structural': connectivity_matrix
        }
        
        return brain_data, connectivity
    
    def _step_export_surface_mesh(self) -> List[str]:
        """步骤3.5: 导出FreeSurfer surface mesh"""
        self.logger.info("步骤3.5: 导出FreeSurfer表面网格...")
        
        if self.freesurfer_loader is None:
            self.logger.warning("没有加载FreeSurfer数据，跳过表面网格导出")
            return []
        
        obj_dir = self.output_dir / "obj"
        obj_dir.mkdir(exist_ok=True)
        
        # Export surface mesh
        exported_files = self.freesurfer_loader.export_surfaces_as_obj(
            output_dir=obj_dir,
            combine_hemispheres=True  # Export as single file
        )
        
        self.logger.info(f"  ✓ 导出了 {len(exported_files)} 个表面网格文件")
        return [str(f.relative_to(self.output_dir)) for f in exported_files]

    
    def _assign_network(self, region_id: int) -> str:
        """为脑区分配网络"""
        if region_id < 20:
            return "Visual"
        elif region_id < 40:
            return "Somatomotor"
        elif region_id < 60:
            return "Dorsal Attention"
        elif region_id < 80:
            return "Ventral Attention"
        elif region_id < 100:
            return "Limbic"
        elif region_id < 140:
            return "Frontoparietal"
        else:
            return "Default Mode"


# 便捷函数
def run_unity_workflow(
    config: Optional[Union[WorkflowConfig, dict]] = None,
    atlas_info: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    运行Unity工作流的便捷函数
    
    Args:
        config: 工作流配置（WorkflowConfig对象或字典）
        atlas_info: 脑图谱信息
        model: 训练好的模型
    
    Returns:
        工作流结果字典
    
    Example:
        >>> from unity_integration.workflow_manager import run_unity_workflow, WorkflowConfig
        >>> 
        >>> config = WorkflowConfig(
        ...     output_dir="output/my_export",
        ...     export_formats=['json', 'obj'],
        ...     time_step=10
        ... )
        >>> 
        >>> results = run_unity_workflow(config)
        >>> print(f"生成了 {len(results['output_files'])} 个文件")
    """
    # 如果config是字典，转换为WorkflowConfig
    if isinstance(config, dict):
        config = WorkflowConfig(**config)
    
    manager = WorkflowManager(config, atlas_info, model)
    return manager.run_full_workflow()
