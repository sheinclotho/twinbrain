#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例: 从训练缓存加载数据用于Unity可视化
=========================================

演示如何使用训练生成的缓存文件进行Unity导出。
这是最常见和推荐的工作流程。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from unity_integration import run_unity_workflow, WorkflowConfig

# 常量
SEPARATOR = "=" * 80


def example_load_from_training_cache():
    """
    示例1: 从训练缓存加载数据
    
    训练完成后，缓存文件位于:
    - test_file3/sub-01/results/cache/eeg_data.pt
    - test_file3/sub-01/results/cache/hetero_graphs.pt
    """
    print(SEPARATOR)
    print("示例1: 从训练缓存加载数据")
    print(SEPARATOR)
    
    # 配置数据路径指向训练结果目录
    config = WorkflowConfig(
        data_source='local',  # 使用本地数据
        data_path='test_file3/sub-01/results',  # 包含cache文件夹的目录
        
        # 输出设置
        output_dir='output/from_cache',
        export_formats=['json', 'obj'],
        
        # 时间范围
        start_time=0,
        end_time=100,
        time_step=5,
        
        # 导出选项
        export_connectivity=True,
        export_networks=True,
        
        # Unity配置
        generate_unity_config=True,
        
        # 元数据
        subject_id='sub-01',
        atlas_name='Schaefer200'
    )
    
    print(f"\n📁 数据路径: {config.data_path}")
    print(f"   期望找到:")
    print(f"   - {config.data_path}/cache/eeg_data.pt")
    print(f"   - {config.data_path}/cache/hetero_graphs.pt")
    
    # 运行工作流
    try:
        results = run_unity_workflow(config)
        
        print(f"\n✅ 完成！")
        print(f"📁 输出目录: {config.output_dir}")
        print(f"📊 生成文件: {len(results['output_files'])} 个")
        print(f"\n前5个生成的文件:")
        for file in results['output_files'][:5]:
            print(f"  - {file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 提示:")
        print("1. 确保已运行训练: python main.py train --config config/default.yaml")
        print("2. 检查缓存文件是否存在")
        print("3. 如果缓存不存在，会自动使用示例数据")
        return None


def example_specify_cache_directory():
    """
    示例2: 直接指定缓存目录
    
    如果缓存文件在其他位置，可以直接指定
    """
    print("\n" + SEPARATOR)
    print("示例2: 直接指定缓存目录")
    print(SEPARATOR)
    
    # 直接指向cache目录的父目录
    config = WorkflowConfig(
        data_source='local',
        data_path='test_file3/sub-01/results',  # workflow_manager会自动查找cache/子目录
        
        output_dir='output/direct_cache',
        export_formats=['json'],
        time_step=10,
        
        subject_id='sub-01',
        atlas_name='Schaefer200'
    )
    
    print(f"\n📁 查找缓存:")
    print(f"   1. 检查: {config.data_path}/cache/")
    print(f"   2. 检查: {config.data_path}/results/cache/")
    print(f"   3. 搜索: {config.data_path}/**/*.pt")
    
    try:
        results = run_unity_workflow(config)
        print(f"\n✅ 完成！输出: {config.output_dir}")
        return results
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return None


def example_with_connectivity_visualization():
    """
    示例3: 包含连接可视化的完整导出
    """
    print("\n" + SEPARATOR)
    print("示例3: 完整导出（含连接）")
    print(SEPARATOR)
    
    config = WorkflowConfig(
        data_source='local',
        data_path='test_file3/sub-01/results',
        
        output_dir='output/full_visualization',
        export_formats=['json', 'obj'],
        
        # 时间序列动画
        start_time=0,
        end_time=200,
        time_step=5,
        
        # 导出连接和网络
        export_connectivity=True,  # 导出结构连接
        export_networks=True,      # 导出网络分区信息
        
        # Unity配置
        generate_unity_config=True,
        generate_materials=True,
        
        subject_id='sub-01',
        atlas_name='Schaefer200'
    )
    
    print(f"\n⚙️ 配置:")
    print(f"   - 时间范围: {config.start_time} - {config.end_time}")
    print(f"   - 时间步长: {config.time_step}")
    print(f"   - 导出格式: {config.export_formats}")
    print(f"   - 连接矩阵: {'是' if config.export_connectivity else '否'}")
    print(f"   - 网络信息: {'是' if config.export_networks else '否'}")
    
    try:
        results = run_unity_workflow(config)
        
        print(f"\n✅ 完成！")
        print(f"📁 输出目录: {config.output_dir}")
        print(f"📊 文件统计:")
        print(f"   - 总文件数: {len(results['output_files'])}")
        
        # 分类统计
        json_files = [f for f in results['output_files'] if f.endswith('.json')]
        obj_files = [f for f in results['output_files'] if f.endswith('.obj')]
        
        print(f"   - JSON文件: {len(json_files)}")
        print(f"   - OBJ文件: {len(obj_files)}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return None


def check_cache_files():
    """检查训练缓存文件是否存在"""
    print(SEPARATOR)
    print("检查训练缓存文件")
    print(SEPARATOR)
    
    base_path = Path('test_file3')
    
    if not base_path.exists():
        print(f"\n⚠️  数据目录不存在: {base_path}")
        print("   请先运行训练生成数据")
        return False
    
    # 查找所有可能的缓存文件
    cache_files = list(base_path.glob("**/cache/*.pt"))
    
    if cache_files:
        print(f"\n✅ 找到 {len(cache_files)} 个缓存文件:")
        for f in cache_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.relative_to(base_path)} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"\n⚠️  未找到缓存文件")
        print("   请先运行训练:")
        print("   python main.py train --config config/default.yaml")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TwinBrain - 从训练缓存加载数据示例")
    print("=" * 80)
    
    # 首先检查缓存文件
    has_cache = check_cache_files()
    
    if has_cache:
        print("\n" + "=" * 80)
        print("运行导出示例")
        print("=" * 80)
        
        # 运行示例
        example_load_from_training_cache()
        example_specify_cache_directory()
        example_with_connectivity_visualization()
        
        print("\n" + "=" * 80)
        print("✅ 所有示例完成！")
        print("=" * 80)
        print("\n📖 更多信息:")
        print("   - 使用指南: 使用指南.md")
        print("   - Unity集成: Unity一键使用指南.md")
        print("   - 文件格式: MODEL_FORMAT.md")
        print("   - 性能优化: PERFORMANCE.md")
        
    else:
        print("\n" + "=" * 80)
        print("💡 使用提示")
        print("=" * 80)
        print("\n要使用此示例，需要先:")
        print("1. 准备数据 (放在 test_file3/ 目录)")
        print("2. 运行训练:")
        print("   python main.py train --config config/default.yaml")
        print("3. 训练会生成缓存文件:")
        print("   - test_file3/sub-XX/results/cache/eeg_data.pt")
        print("   - test_file3/sub-XX/results/cache/hetero_graphs.pt")
        print("4. 然后运行此示例脚本:")
        print("   python example_load_cache_for_unity.py")
        print("\n如果只是想测试，可以使用其他示例脚本:")
        print("   python example_unity_integration.py  # 使用随机数据")
