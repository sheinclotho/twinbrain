#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain Unity 一键启动脚本
============================

这个脚本提供了TwinBrain Unity集成的完整启动流程。

功能:
1. 检查环境和依赖
2. 启动WebSocket后端服务器
3. 加载训练好的模型
4. 提供实时预测和刺激模拟
5. 导出大脑状态数据供Unity使用

使用方法:
    python unity_startup.py --model path/to/model.pth --port 8765

作者: TwinBrain Team
日期: 2024-02-13
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查必要的依赖包"""
    missing = []
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
    
    try:
        import numpy
        logger.info(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
    
    try:
        import websockets
        logger.info(f"✓ websockets {websockets.__version__}")
    except ImportError:
        missing.append("websockets")
    
    try:
        import nibabel
        logger.info(f"✓ nibabel {nibabel.__version__}")
    except ImportError:
        logger.warning("⚠ nibabel not found (optional for FreeSurfer data)")
    
    if missing:
        logger.error(f"❌ Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def check_unity_structure(output_dir: Path):
    """检查Unity项目结构"""
    logger.info("\n检查Unity项目结构...")
    
    required_dirs = [
        output_dir / "Scripts",
        output_dir / "state",
        output_dir / "data"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if dir_path.exists():
            logger.info(f"✓ {dir_path.relative_to(output_dir)}")
        else:
            logger.warning(f"✗ {dir_path.relative_to(output_dir)}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.warning("⚠ 部分目录缺失。运行 setup_unity_project.py 创建完整结构")
        return False
    
    return True


def load_model(model_path: Optional[Path]):
    """加载训练好的模型"""
    if not model_path or not model_path.exists():
        logger.warning("⚠ 模型文件不存在，将使用空模型（仅演示）")
        return None
    
    try:
        import torch
        logger.info(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info("✓ 模型加载成功")
        return checkpoint
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return None


async def start_server(
    model_path: Optional[Path],
    output_dir: Path,
    host: str,
    port: int
):
    """启动WebSocket服务器"""
    logger.info("\n" + "="*60)
    logger.info("启动TwinBrain WebSocket服务器")
    logger.info("="*60)
    
    try:
        from unity_integration import BrainVisualizationServer
        from unity_integration import BrainStateExporter, StimulationSimulator
        
        # 加载模型
        model = load_model(model_path)
        
        # 创建导出器
        state_dir = output_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        exporter = BrainStateExporter(
            atlas_info=None,
            output_dir=str(state_dir)
        )
        logger.info(f"✓ 状态导出器: {state_dir}")
        
        # 创建模拟器
        simulator = StimulationSimulator(n_regions=200)
        logger.info("✓ 刺激模拟器: 200个脑区")
        
        # 创建服务器
        server = BrainVisualizationServer(
            model=model,
            exporter=exporter,
            simulator=simulator,
            model_path=str(model_path) if model_path else None,
            output_dir=str(output_dir),
            host=host,
            port=port
        )
        
        logger.info(f"\n🚀 服务器启动: ws://{host}:{port}")
        logger.info("等待Unity客户端连接...")
        logger.info("按 Ctrl+C 停止服务器\n")
        
        # 启动服务器
        await server.start()
        
    except ImportError as e:
        logger.error(f"❌ 导入错误: {e}")
        logger.info("确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 服务器错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity 一键启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用训练好的模型启动
  python unity_startup.py --model results/best_model.pth
  
  # 指定输出目录和端口
  python unity_startup.py --output Unity_TwinBrain --port 8080
  
  # 演示模式（无模型）
  python unity_startup.py --demo
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='训练好的模型文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='Unity_TwinBrain',
        help='Unity项目输出目录（默认: Unity_TwinBrain）'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器主机地址（默认: 0.0.0.0）'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='服务器端口（默认: 8765）'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='演示模式（不加载模型）'
    )
    
    args = parser.parse_args()
    
    # 打印标题
    print("\n" + "="*60)
    print("  TwinBrain Unity 集成 - 一键启动")
    print("="*60 + "\n")
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查Unity结构
    output_dir = Path(args.output)
    if not output_dir.exists():
        logger.warning(f"⚠ 输出目录不存在: {output_dir}")
        logger.info(f"创建目录: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    check_unity_structure(output_dir)
    
    # 解析模型路径
    model_path = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"❌ 模型文件不存在: {model_path}")
            sys.exit(1)
    elif not args.demo:
        logger.warning("⚠ 未指定模型文件，将尝试使用默认模型")
        default_model = project_root / "results" / "best_model.pth"
        if default_model.exists():
            model_path = default_model
            logger.info(f"✓ 使用默认模型: {model_path}")
        else:
            logger.warning("⚠ 默认模型不存在，使用演示模式")
    
    # 启动服务器
    try:
        asyncio.run(start_server(model_path, output_dir, args.host, args.port))
    except KeyboardInterrupt:
        logger.info("\n\n👋 服务器已停止")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
