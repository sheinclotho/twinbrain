#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity 自动化工作流示例
======================

演示如何使用新的 WorkflowManager 实现一键导出。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from unity_integration import run_unity_workflow, WorkflowConfig

# 常量
SEPARATOR = SEPARATOR


def example_basic_workflow():
    """示例1: 基础工作流"""
    print(SEPARATOR)
    print("示例1: 基础工作流")
    print(SEPARATOR)
    
    # 最简单的配置
    config = WorkflowConfig(
        output_dir='output/basic_export',
        export_formats=['json'],  # 只导出 JSON
        time_step=10,  # 每 10 帧一个文件
    )
    
    # 运行工作流
    results = run_unity_workflow(config)
    
    print(f"\n✅ 完成！")
    print(f"📁 输出目录: {config.output_dir}")
    print(f"📊 生成文件: {len(results['output_files'])} 个")
    print(f"✓ 完成步骤: {', '.join(results['steps_completed'])}")


def example_full_workflow():
    """示例2: 完整工作流（包含所有功能）"""
    print("\n" + SEPARATOR)
    print("示例2: 完整工作流")
    print(SEPARATOR)
    
    # 完整配置
    config = WorkflowConfig(
        # 数据源
        data_source='local',  # 使用本地数据（当前为示例数据）
        
        # 输出设置
        output_dir='output/full_export',
        export_formats=['json', 'obj'],  # 导出 JSON 和 OBJ
        
        # 时间范围
        start_time=0,
        end_time=100,
        time_step=5,
        
        # 导出选项
        export_connectivity=True,  # 导出连接
        export_networks=True,      # 导出网络信息
        export_obj_per_frame=False,  # 导出单个聚合 OBJ
        
        # Unity 配置
        generate_unity_config=True,  # 生成 Unity 配置
        generate_materials=True,     # 生成材质配置
        
        # 元数据
        subject_id='example_subject',
        atlas_name='Schaefer200'
    )
    
    # 运行工作流
    results = run_unity_workflow(config)
    
    print(f"\n✅ 完成！")
    print(f"📁 输出目录: {config.output_dir}")
    print(f"📊 生成文件数: {len(results['output_files'])}")
    print(f"\n生成的文件类型:")
    for file in results['output_files'][:5]:  # 显示前 5 个
        print(f"  - {file}")
    if len(results['output_files']) > 5:
        print(f"  ... 还有 {len(results['output_files']) - 5} 个文件")
    
    print(f"\n✓ 完成步骤:")
    for step in results['steps_completed']:
        print(f"  ✓ {step}")


def example_animation_workflow():
    """示例3: 动画序列导出"""
    print("\n" + SEPARATOR)
    print("示例3: 动画序列导出")
    print(SEPARATOR)
    
    config = WorkflowConfig(
        output_dir='output/animation',
        export_formats=['json'],
        start_time=0,
        end_time=200,
        time_step=2,  # 密集采样用于流畅动画
        export_connectivity=True,
        subject_id='animation_demo'
    )
    
    results = run_unity_workflow(config)
    
    print(f"\n✅ 动画序列生成完成！")
    print(f"📁 输出目录: {config.output_dir}")
    print(f"🎬 帧数: {len([f for f in results['output_files'] if 'brain_state' in f])}")
    print(f"⏱️  时间范围: {config.start_time} - {config.end_time}")
    print(f"📈 帧步长: {config.time_step}")
    
    print(f"\nUnity 使用方法:")
    print(f"  1. 设置 JSON Path: {config.output_dir}/json/")
    print(f"  2. 勾选 Load Sequence")
    print(f"  3. 勾选 Auto Play")
    print(f"  4. 运行并观察大脑活动动画")


def example_quick_preview():
    """示例4: 快速预览（最小文件）"""
    print("\n" + SEPARATOR)
    print("示例4: 快速预览")
    print(SEPARATOR)
    
    config = WorkflowConfig(
        output_dir='output/quick_preview',
        export_formats=['json'],
        start_time=0,
        end_time=50,
        time_step=25,  # 只导出 3 个时间点: 0, 25, 50
        export_connectivity=False,  # 跳过连接以加速
        export_networks=False,
        generate_unity_config=True,
        generate_materials=False
    )
    
    results = run_unity_workflow(config)
    
    print(f"\n✅ 快速预览生成完成！")
    print(f"📁 输出目录: {config.output_dir}")
    print(f"📊 文件数（最小化）: {len(results['output_files'])}")
    print(f"\n适用场景:")
    print(f"  - 快速测试 Unity 配置")
    print(f"  - 验证数据正确性")
    print(f"  - 低性能设备预览")


def main():
    """运行所有示例"""
    print(SEPARATOR)
    print("Unity 自动化工作流示例")
    print(SEPARATOR)
    print("\n本示例演示如何使用新的工作流管理器")
    print("一键完成从数据处理到 Unity 导出的全过程\n")
    
    try:
        # 运行示例
        example_basic_workflow()
        example_full_workflow()
        example_animation_workflow()
        example_quick_preview()
        
        print("\n" + SEPARATOR)
        print("所有示例完成！")
        print(SEPARATOR)
        print("\n📚 下一步:")
        print("  1. 查看生成的文件: output/ 目录")
        print("  2. 阅读文档: docs/Unity工作流说明.md")
        print("  3. 在 Unity 中加载 JSON 文件")
        print("  4. 使用 unity_config.json 配置项目")
        print("\n💡 提示:")
        print("  - 使用 WorkflowConfig 自定义配置")
        print("  - 参考 docs/Unity工作流说明.md 了解所有选项")
        print("  - 查看 workflow_report.json 了解详细执行信息")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
