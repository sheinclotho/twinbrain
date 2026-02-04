#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeSurfer to Unity 示例脚本
===========================

演示如何使用 FreeSurfer 表面文件生成 Unity 可视化数据。

使用方法:
    python example_freesurfer_unity.py
"""

from pathlib import Path
from unity_integration import run_unity_workflow, WorkflowConfig

def main():
    """主函数：演示 FreeSurfer 到 Unity 的完整流程"""
    
    print("="*80)
    print("TwinBrain - FreeSurfer to Unity 示例")
    print("="*80)
    print()
    
    # 设置文件路径
    # 注意：这些路径需要指向您的实际 FreeSurfer 文件
    # 这里使用的是示例路径，请根据实际情况修改
    data_dir = Path('data/freesurfer')
    output_dir = Path('output/freesurfer_unity_demo')
    
    # 检查文件是否存在
    files_to_check = {
        'lh_surface': data_dir / 'lh.pial',
        'rh_surface': data_dir / 'rh.pial',
        'lh_annot': data_dir / 'lh.Schaefer2018_200Parcels_7Networks_order.annot',
        'rh_annot': data_dir / 'rh.Schaefer2018_200Parcels_7Networks_order.annot'
    }
    
    missing_files = []
    for name, path in files_to_check.items():
        if not path.exists():
            missing_files.append(str(path))
    
    if missing_files:
        print("⚠️  警告: 以下文件不存在，将使用示例数据代替:")
        for file in missing_files:
            print(f"   - {file}")
        print()
        print("如果您有 FreeSurfer 数据，请将文件放在以下位置:")
        print(f"   {data_dir}")
        print()
        print("或者修改此脚本中的 data_dir 变量指向您的数据目录。")
        print()
        
        # 使用示例数据
        use_example = True
    else:
        print("✓ 找到所有 FreeSurfer 文件")
        print()
        use_example = False
    
    # 配置工作流
    if use_example:
        print("使用示例数据生成演示...")
        config = WorkflowConfig(
            data_source='example',  # 使用示例数据
            output_dir=str(output_dir),
            export_formats=['json', 'obj'],
            start_time=0,
            end_time=50,  # 较少的时间点用于演示
            time_step=5,
            export_connectivity=True,
            export_networks=True,
            generate_unity_config=True,
            generate_materials=True,
            subject_id='demo_subject',
            atlas_name='Schaefer200'
        )
    else:
        print("使用 FreeSurfer 数据...")
        config = WorkflowConfig(
            # FreeSurfer 数据源
            data_source='freesurfer',
            freesurfer_lh_surface=str(files_to_check['lh_surface']),
            freesurfer_rh_surface=str(files_to_check['rh_surface']),
            freesurfer_lh_annot=str(files_to_check['lh_annot']),
            freesurfer_rh_annot=str(files_to_check['rh_annot']),
            
            # 输出配置
            output_dir=str(output_dir),
            export_formats=['json', 'obj'],
            export_surface_mesh=True,  # 导出真实的表面网格
            
            # 时间序列参数
            start_time=0,
            end_time=50,  # 较少的时间点用于演示
            time_step=5,
            
            # 可视化选项
            export_connectivity=True,
            export_networks=True,
            
            # Unity 配置
            generate_unity_config=True,
            generate_materials=True,
            
            # 主体信息
            subject_id='freesurfer_subject',
            atlas_name='Schaefer2018_200Parcels_7Networks'
        )
    
    # 运行工作流
    print()
    print("开始处理...")
    print("-"*80)
    
    try:
        results = run_unity_workflow(config)
        
        # 打印结果
        print()
        print("="*80)
        print("✅ 处理完成！")
        print("="*80)
        print()
        print(f"完成步骤: {', '.join(results['steps_completed'])}")
        print(f"生成文件数: {len(results['output_files'])}")
        print(f"输出目录: {output_dir}")
        print()
        print("生成的文件:")
        for file in sorted(results['output_files']):
            print(f"  ✓ {file}")
        print()
        print("下一步:")
        print("  1. 在 Unity 中创建新项目")
        print(f"  2. 导入 {output_dir}/obj/ 目录中的 OBJ 文件")
        print(f"  3. 使用 {output_dir}/json/ 目录中的 JSON 文件加载动画数据")
        print(f"  4. 参考 {output_dir}/unity_config.json 配置可视化参数")
        print()
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ 处理失败")
        print("="*80)
        print(f"错误: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
