"""
TwinBrain V5 主程序
==================

图原生数字孪生脑训练系统

使用方法:
    python main.py --config configs/default.yaml
    
或直接运行:
    python main.py  # 使用默认配置
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import BrainDataLoader
from models.graph_native_mapper import GraphNativeBrainMapper
from models.graph_native_system import GraphNativeBrainModel, GraphNativeTrainer
from utils.helpers import setup_logging, set_seed, save_config, create_output_dir


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def prepare_data(config: dict, logger: logging.Logger):
    """准备训练数据"""
    logger.info("=" * 60)
    logger.info("步骤 1/4: 加载数据")
    logger.info("=" * 60)
    
    # 初始化数据加载器
    data_loader = BrainDataLoader(
        data_root=config['data']['root_dir'],
        modalities=config['data']['modalities'],
    )
    
    # 加载所有被试
    all_data = data_loader.load_all_subjects(
        task=config['data'].get('task'),
        max_subjects=config['data'].get('max_subjects'),
    )
    
    if not all_data:
        raise ValueError("未加载到任何数据，请检查数据路径配置")
    
    logger.info(f"成功加载 {len(all_data)} 个被试数据")
    
    return all_data


def build_graphs(all_data, config: dict, logger: logging.Logger):
    """构建图结构"""
    logger.info("=" * 60)
    logger.info("步骤 2/4: 构建图结构")
    logger.info("=" * 60)
    
    # 初始化图映射器
    mapper = GraphNativeBrainMapper(
        atlas_name=config['data']['atlas']['name'],
        add_self_loops=config['graph']['add_self_loops'],
        make_undirected=config['graph']['make_undirected'],
    )
    
    # 为每个被试构建图
    graphs = []
    for subject_data in all_data:
        graph_list = []
        
        # fMRI图
        if 'fmri' in subject_data:
            fmri_data = subject_data['fmri']['data']
            # 提取ROI时序 (需要atlas)
            # 这里简化处理，实际需要用atlas提取
            if fmri_data.ndim == 4:  # 4D fMRI
                n_volumes = fmri_data.shape[-1]
                # 简化: 使用平均
                fmri_ts = fmri_data.reshape(-1, n_volumes).mean(axis=0)
                fmri_ts = fmri_ts.reshape(1, -1)  # [1, T]
            
            fmri_graph = mapper.map_fmri_to_graph(
                timeseries=fmri_ts,
                connectivity_matrix=None,  # 自动计算
            )
            graph_list.append(fmri_graph)
        
        # EEG图
        if 'eeg' in subject_data:
            eeg_data = subject_data['eeg']['data']  # [n_channels, n_times]
            eeg_ch_names = subject_data['eeg']['ch_names']
            
            eeg_graph = mapper.map_eeg_to_graph(
                timeseries=eeg_data,
                channel_names=eeg_ch_names,
            )
            graph_list.append(eeg_graph)
        
        # 合并图
        if len(graph_list) > 0:
            # 简化处理: 只取第一个图
            # 实际应该合并为异构图
            graphs.append(graph_list[0])
    
    logger.info(f"成功构建 {len(graphs)} 个图")
    
    return graphs, mapper


def create_model(config: dict, logger: logging.Logger):
    """创建模型"""
    logger.info("=" * 60)
    logger.info("步骤 3/4: 创建模型")
    logger.info("=" * 60)
    
    # 确定节点和边类型
    node_types = config['data']['modalities']
    edge_types = []
    
    for modality in node_types:
        edge_types.append((modality, 'connects', modality))
    
    # 跨模态边
    if len(node_types) > 1:
        edge_types.append((node_types[0], 'projects_to', node_types[1]))
    
    # 输入通道
    in_channels_dict = {modality: 1 for modality in node_types}
    
    # 创建模型
    model = GraphNativeBrainModel(
        node_types=node_types,
        edge_types=edge_types,
        in_channels_dict=in_channels_dict,
        hidden_channels=config['model']['hidden_channels'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        use_prediction=config['model']['use_prediction'],
        prediction_steps=config['model']['prediction_steps'],
        dropout=config['model']['dropout'],
    )
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_model(model, graphs, config: dict, logger: logging.Logger):
    """训练模型"""
    logger.info("=" * 60)
    logger.info("步骤 4/4: 训练模型")
    logger.info("=" * 60)
    
    # 划分训练/验证集
    n_train = int(len(graphs) * 0.8)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:]
    
    logger.info(f"训练集: {len(train_graphs)} 个样本")
    logger.info(f"验证集: {len(val_graphs)} 个样本")
    
    # 创建训练器
    trainer = GraphNativeTrainer(
        model=model,
        node_types=config['data']['modalities'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        use_adaptive_loss=config['training']['use_adaptive_loss'],
        use_eeg_enhancement=config['training']['use_eeg_enhancement'],
        device=config['device']['type'],
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # 训练
        train_loss = trainer.train_epoch(train_graphs)
        
        # 验证
        if epoch % config['training']['val_frequency'] == 0:
            val_loss = trainer.validate(val_graphs)
            
            logger.info(
                f"Epoch {epoch}/{config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存检查点
                output_dir = Path(config['output']['output_dir'])
                checkpoint_path = output_dir / "best_model.pt"
                trainer.save_checkpoint(checkpoint_path, epoch)
                logger.info(f"保存最佳模型: val_loss={val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"早停触发: {patience_counter} 个epoch无改进")
                break
        else:
            logger.info(
                f"Epoch {epoch}/{config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}"
            )
        
        # 定期保存检查点
        if epoch % config['training']['save_frequency'] == 0:
            output_dir = Path(config['output']['output_dir'])
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch)
    
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")


def main():
    """主函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description='TwinBrain V5 训练系统')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (default: configs/default.yaml)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (default: 42)'
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    output_dir = create_output_dir(
        config['output']['output_dir'],
        config['output']['experiment_name']
    )
    config['output']['output_dir'] = str(output_dir)
    
    # 设置日志
    logger = setup_logging(
        output_dir / "training.log",
        level=config['output']['log_level']
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 保存配置
    save_config(config, output_dir / "config.yaml")
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("TwinBrain V5 - 图原生数字孪生脑训练系统")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config or 'configs/default.yaml'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {config['device']['type']}")
    logger.info(f"随机种子: {args.seed}")
    logger.info("=" * 60)
    
    try:
        # 步骤1: 准备数据
        all_data = prepare_data(config, logger)
        
        # 步骤2: 构建图
        graphs, mapper = build_graphs(all_data, config, logger)
        
        # 步骤3: 创建模型
        model = create_model(config, logger)
        
        # 步骤4: 训练
        train_model(model, graphs, config, logger)
        
        logger.info("=" * 60)
        logger.info("✅ 所有任务完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 运行失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
