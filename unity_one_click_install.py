#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain Unity 一键完整安装脚本
================================

这个脚本整合了所有步骤，真正实现一键完成：
1. 生成FreeSurfer OBJ模型（如果提供）
2. 复制所有文件到Unity项目
3. 安装所有脚本（包括自动化Editor脚本）
4. 配置依赖和设置

使用方法:
    # 完整安装（带FreeSurfer）
    python unity_one_click_install.py \\
        --unity-project /path/to/UnityProject \\
        --freesurfer-dir /path/to/freesurfer
    
    # 基础安装（无FreeSurfer）
    python unity_one_click_install.py \\
        --unity-project /path/to/UnityProject
"""

import argparse
import sys
import logging
from pathlib import Path

# 导入现有的模块
try:
    from unity_package_installer import UnityPackageInstaller
    import setup_unity_project
except ImportError as e:
    print(f"错误: 无法导入必要的模块: {e}")
    print("请确保在TwinBrain项目根目录中运行此脚本")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity 一键完整安装",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整安装（带FreeSurfer）
  python unity_one_click_install.py --unity-project /path/to/UnityProject --freesurfer-dir /path/to/freesurfer
  
  # 基础安装（使用默认球体）
  python unity_one_click_install.py --unity-project /path/to/UnityProject
  
安装后在Unity中: TwinBrain -> 自动设置场景 -> 开始自动设置
        """
    )
    
    parser.add_argument(
        '--unity-project',
        type=Path,
        required=True,
        help='Unity项目路径（必需）'
    )
    
    parser.add_argument(
        '--freesurfer-dir',
        type=Path,
        help='FreeSurfer文件目录路径（可选）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='中间文件输出目录（默认: ./unity_project）'
    )
    
    args = parser.parse_args()
    
    unity_project = args.unity_project.resolve()
    
    if not unity_project.exists():
        logger.error(f"Unity项目不存在: {unity_project}")
        return 1
    
    if not (unity_project / "Assets").exists():
        logger.error(f"无效的Unity项目（缺少Assets目录）: {unity_project}")
        return 1
    
    logger.info("="*80)
    logger.info("TwinBrain Unity 一键完整安装")
    logger.info("="*80)
    logger.info(f"Unity项目: {unity_project}")
    logger.info("")
    
    # 生成并安装Unity资源（使用安静模式避免冗余输出）
    setup = setup_unity_project.UnityWorkflowSetup(output_base=args.output_dir, verbose=False)
    
    # 创建文件夹结构
    setup.create_folder_structure()
    
    # 处理FreeSurfer文件（如果提供）
    if args.freesurfer_dir and args.freesurfer_dir.exists():
        logger.info(f"✓ 使用FreeSurfer数据: {args.freesurfer_dir}")
        lh_surface = args.freesurfer_dir / "lh.pial"
        rh_surface = args.freesurfer_dir / "rh.pial"
        lh_annot = list(args.freesurfer_dir.glob("lh.*.annot"))
        rh_annot = list(args.freesurfer_dir.glob("rh.*.annot"))
        
        setup.process_freesurfer_files(
            lh_surface=lh_surface if lh_surface.exists() else None,
            rh_surface=rh_surface if rh_surface.exists() else None,
            lh_annot=lh_annot[0] if lh_annot else None,
            rh_annot=rh_annot[0] if rh_annot else None
        )
    else:
        setup.process_freesurfer_files()
    
    # 生成Unity脚本
    setup.generate_unity_scripts()
    
    # 安装到Unity项目
    logger.info("✓ 安装到Unity项目...")
    installer = UnityPackageInstaller(unity_project)
    
    # 运行安装
    if not installer.run_installation(setup.output_base):
        logger.error("安装失败")
        return 1
    
    # 复制OBJ文件（如果存在）
    if not setup.copy_obj_to_unity_project(unity_project):
        logger.warning("OBJ文件复制失败或跳过（如果没有FreeSurfer数据这是正常的）")
    
    # 完成
    logger.info("\n" + "="*80)
    logger.info("✅ 一键安装完成！")
    logger.info("="*80)
    
    logger.info("\n📋 后续步骤:")
    logger.info("1. 在Unity Hub中打开项目（Unity会自动下载依赖包）")
    logger.info("2. 菜单: TwinBrain -> 自动设置场景")
    logger.info("3. 点击'开始自动设置'按钮，完成后即可点击Play测试")
    
    if args.freesurfer_dir:
        logger.info("\n✨ 已生成并复制OBJ文件，自动设置时会配置所有200+个OBJ文件")
    
    logger.info("\n📖 详细文档: Unity使用指南.md\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
