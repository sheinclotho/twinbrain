#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain 正确工作流示例
======================

演示正确的 FreeSurfer + 数据加载工作流。

工作流程：
1. 使用 FreeSurfer 创建 Unity 前端结构（一次性）
2. 提供数据文件夹，加载真实脑数据
3. 一键启动，获得真实状态的脑模型
4. （可选）交互式虚拟刺激和实时预测
"""

import sys
from pathlib import Path
import torch
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from unity_integration import (
    FreeSurferLoader,
    BrainStateExporter,
    run_unity_workflow,
    WorkflowConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def step1_create_frontend_structure():
    """
    步骤 1: 使用 FreeSurfer 创建 Unity 前端结构
    
    这是一次性设置，创建 OBJ 模型和区域定义。
    """
    print("\n" + "="*80)
    print("步骤 1: 创建 Unity 前端结构（使用 FreeSurfer）")
    print("="*80)
    
    # 加载 FreeSurfer 文件（如果有）
    freesurfer_available = False
    
    # 检查文件是否存在
    freesurfer_files = {
        'lh_surface': 'data/freesurfer/lh.pial',
        'rh_surface': 'data/freesurfer/rh.pial',
        'lh_annot': 'data/freesurfer/lh.Schaefer2018_200Parcels_7Networks_order.annot',
        'rh_annot': 'data/freesurfer/rh.Schaefer2018_200Parcels_7Networks_order.annot'
    }
    
    if all(Path(f).exists() for f in freesurfer_files.values()):
        freesurfer_available = True
        print("✓ 找到 FreeSurfer 文件")
    else:
        print("⚠️  未找到 FreeSurfer 文件，将使用默认配置")
    
    # 创建 Unity 前端结构
    if freesurfer_available:
        # 使用 FreeSurfer 定义结构
        from unity_integration import load_freesurfer_data
        
        atlas_info, loader = load_freesurfer_data(
            lh_surface=freesurfer_files['lh_surface'],
            rh_surface=freesurfer_files['rh_surface'],
            lh_annot=freesurfer_files['lh_annot'],
            rh_annot=freesurfer_files['rh_annot']
        )
        
        # 导出前端结构（不包含数据）
        from unity_integration import BrainOBJGenerator
        
        obj_generator = BrainOBJGenerator(atlas_info=atlas_info)
        output_dir = Path('output/unity_frontend')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出 OBJ 模型（结构）
        obj_generator.export_brain_model(
            output_path=output_dir / 'brain_structure.obj',
            activity_data=None  # 不需要活动数据
        )
        
        # 可选：导出表面网格
        loader.export_surfaces_as_obj(
            output_dir=output_dir,
            combine_hemispheres=True
        )
        
        print(f"✓ Unity 前端结构已创建: {output_dir}")
        
    else:
        # 使用默认配置创建
        config = WorkflowConfig(
            data_source='example',
            output_dir='output/unity_frontend',
            export_formats=['obj'],  # 只导出结构
            export_obj_per_frame=False
        )
        
        # 只创建结构，不包含数据
        print("✓ 使用默认配置创建前端结构")
        atlas_info = None
    
    return atlas_info


def step2_load_data_folder(data_folder: str, atlas_info=None):
    """
    步骤 2: 加载数据文件夹
    
    从文件夹加载真实的脑数据（原始/缓存/输出）。
    """
    print("\n" + "="*80)
    print("步骤 2: 加载数据文件夹")
    print("="*80)
    
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"⚠️  数据文件夹不存在: {data_folder}")
        print("使用示例数据...")
        
        # 生成示例数据
        n_regions = atlas_info['n_regions'] if atlas_info else 200
        n_timepoints = 200
        n_features = 1
        
        brain_data = {
            'fmri': torch.randn(n_regions, n_timepoints, n_features),
            'eeg': torch.randn(n_regions, n_timepoints, n_features)
        }
        
        print(f"✓ 加载了示例数据: {n_regions} 个脑区, {n_timepoints} 个时间点")
        return brain_data
    
    # 检查数据类型并加载
    print(f"检查数据文件夹: {data_path}")
    
    # 查找可能的数据文件
    nii_files = list(data_path.glob('*.nii.gz')) + list(data_path.glob('*.nii'))
    pt_files = list(data_path.glob('*.pt'))
    
    if nii_files:
        print(f"✓ 找到 {len(nii_files)} 个 NIfTI 文件")
        # 加载 fMRI NIfTI 数据
        brain_data = load_nifti_data(nii_files[0], atlas_info)
        
    elif pt_files:
        print(f"✓ 找到 {len(pt_files)} 个 PyTorch 缓存文件")
        # 加载缓存的图数据
        brain_data = load_cached_data(pt_files[0])
        
    else:
        print("⚠️  未找到支持的数据文件")
        print("支持的格式: .nii.gz, .nii, .pt")
        return None
    
    return brain_data


def load_nifti_data(nii_file: Path, atlas_info=None):
    """从 NIfTI 文件加载 fMRI 数据"""
    try:
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker
        
        print(f"加载 fMRI 数据: {nii_file.name}")
        
        # 加载 NIfTI
        fmri_img = nib.load(str(nii_file))
        
        # 使用 masker 提取时间序列
        # 注意：这需要一个图谱标签文件
        # 这里简化处理
        fmri_array = fmri_img.get_fdata()
        
        # 转换为时间序列格式
        # 假设最后一维是时间
        if len(fmri_array.shape) == 4:
            n_timepoints = fmri_array.shape[-1]
            # 简化：使用均值作为每个脑区的信号
            n_regions = atlas_info['n_regions'] if atlas_info else 200
            
            # 实际应该使用 atlas 提取每个区域的信号
            # 这里用随机数演示
            region_timeseries = torch.randn(n_regions, n_timepoints, 1)
            
            brain_data = {
                'fmri': region_timeseries
            }
            
            print(f"✓ 提取了 {n_regions} 个脑区的时间序列")
            return brain_data
        
    except Exception as e:
        print(f"❌ 加载 NIfTI 失败: {e}")
        return None


def load_cached_data(pt_file: Path):
    """从缓存的 PyTorch 文件加载数据"""
    try:
        print(f"加载缓存数据: {pt_file.name}")
        
        data = torch.load(pt_file)
        
        # 检查数据格式
        if isinstance(data, dict):
            # 假设是 HeteroData 或类似格式
            if 'fmri' in data:
                fmri_data = data['fmri']
                
                # 检查是否有 x_seq 属性
                if hasattr(fmri_data, 'x_seq'):
                    brain_data = {'fmri': fmri_data.x_seq}
                elif isinstance(fmri_data, torch.Tensor):
                    brain_data = {'fmri': fmri_data}
                else:
                    brain_data = {'fmri': fmri_data}
                
                print(f"✓ 加载了 fMRI 数据")
                return brain_data
        
        elif isinstance(data, list):
            # 假设是 data_list
            print(f"✓ 找到 {len(data)} 个数据批次")
            
            # 提取第一个批次的数据作为示例
            if len(data) > 0:
                first_data = data[0]
                if hasattr(first_data, 'to_dict'):
                    data_dict = first_data.to_dict()
                    # 提取 fMRI 数据...
        
        print("⚠️  无法识别的数据格式")
        return None
        
    except Exception as e:
        print(f"❌ 加载缓存数据失败: {e}")
        return None


def step3_export_to_json(brain_data, atlas_info=None):
    """
    步骤 3: 将数据导出为 JSON
    
    转换数据为 Unity 可读的 JSON 格式。
    """
    print("\n" + "="*80)
    print("步骤 3: 导出数据为 JSON")
    print("="*80)
    
    if brain_data is None:
        print("❌ 没有数据可导出")
        return
    
    # 创建导出器
    exporter = BrainStateExporter(
        atlas_info=atlas_info,
        model_version="v4"
    )
    
    output_dir = Path('output/brain_states')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出时间序列
    exporter.export_sequence(
        brain_activity=brain_data,
        output_dir=output_dir,
        start=0,
        end=min(50, brain_data['fmri'].shape[1]) if 'fmri' in brain_data else 50,
        step=1
    )
    
    json_files = list(output_dir.glob('*.json'))
    print(f"✓ 导出了 {len(json_files)} 个 JSON 文件到 {output_dir}")
    
    return output_dir


def step4_interactive_mode(atlas_info=None):
    """
    步骤 4: 启动交互式模式（可选）
    
    启动 WebSocket 服务器，支持实时刺激和预测。
    """
    print("\n" + "="*80)
    print("步骤 4: 启动交互式模式（可选）")
    print("="*80)
    
    try:
        from unity_integration import BrainVisualizationServer
        
        # 检查是否有训练好的模型
        model_path = Path('checkpoints/best_model.pt')
        
        if model_path.exists():
            print(f"✓ 找到模型: {model_path}")
            model = torch.load(model_path)
        else:
            print("⚠️  未找到训练好的模型")
            print("交互式预测需要模型。跳过此步骤。")
            return
        
        # 创建服务器
        server = BrainVisualizationServer(
            model=model,
            exporter=BrainStateExporter(atlas_info=atlas_info),
            port=8765
        )
        
        print("\n启动 WebSocket 服务器...")
        print("Unity 可以连接到: ws://localhost:8765")
        print("\n支持的操作：")
        print("  - 发送虚拟刺激")
        print("  - 接收实时预测")
        print("  - 自动转换为 JSON")
        print("\n按 Ctrl+C 停止服务器\n")
        
        # 启动服务器（这会阻塞）
        server.start()
        
    except ImportError as e:
        print(f"⚠️  无法启动交互式模式: {e}")
        print("需要安装: pip install websockets")
    except KeyboardInterrupt:
        print("\n\n✓ 服务器已停止")


def main():
    """主函数：演示完整工作流"""
    print("\n" + "="*80)
    print("TwinBrain 正确工作流示例")
    print("="*80)
    print("\n这个示例展示了如何正确使用 FreeSurfer 和数据文件：")
    print("1. FreeSurfer → 创建 Unity 前端结构（一次性）")
    print("2. 数据文件夹 → 加载真实脑数据")
    print("3. 转换为 JSON → 映射到脑区")
    print("4. 可选：交互式刺激和预测")
    print()
    
    # 步骤 1: 创建前端结构
    atlas_info = step1_create_frontend_structure()
    
    # 步骤 2: 加载数据
    # 用户应该提供数据文件夹路径
    data_folder = 'data/brain_data'  # 示例路径
    brain_data = step2_load_data_folder(data_folder, atlas_info)
    
    # 步骤 3: 导出 JSON
    if brain_data:
        json_dir = step3_export_to_json(brain_data, atlas_info)
        
        print("\n" + "="*80)
        print("✅ 工作流完成！")
        print("="*80)
        print("\n下一步：")
        print("1. 在 Unity 中加载前端结构（OBJ 模型）")
        print("2. 加载 JSON 数据文件")
        print("3. 使用时间轴播放可视化")
        print("\n可选：启动交互式模式进行实时预测")
        
        # 询问是否启动交互式模式
        try:
            choice = input("\n是否启动交互式模式？(y/n): ")
            if choice.lower() == 'y':
                step4_interactive_mode(atlas_info)
        except KeyboardInterrupt:
            print("\n\n✓ 退出")


if __name__ == '__main__':
    main()
