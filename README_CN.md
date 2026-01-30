# TwinBrain 用户指南（中文版）

## 📖 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [配置系统详解](#配置系统详解)
5. [工作流程说明](#工作流程说明)
6. [超参数调优指南](#超参数调优指南)
7. [命令行使用](#命令行使用)
8. [数据处理流程](#数据处理流程)
9. [最佳实践](#最佳实践)
10. [常见问题](#常见问题)

---

## 项目概述

### 什么是 TwinBrain？

TwinBrain（数字孪生脑）是一个基于多模态脑成像数据的深度学习系统，用于重建和预测大脑活动。

**核心功能**:
- 🧠 多模态脑成像数据处理（fMRI、EEG、DTI）
- 🔗 异构图神经网络建模
- 📊 脑信号重建与预测
- 🎯 时空对齐和分析
- 📈 可视化和结果导出

**技术特点**:
- 配置驱动的灵活架构
- 模块化的工作流设计
- 统一的命令行接口
- 完整的日志和诊断系统

---

## 系统架构

### 整体架构图

```
TwinBrain 系统架构
├─────────────────────────────────────────────┐
│                                             │
│  1. 数据输入层                              │
│     ├── fMRI 数据 (*.nii)                   │
│     ├── EEG 数据 (*.fif)                    │
│     ├── DTI 连接矩阵 (*.npy)                │
│     └── 脑图谱 (Atlas JSON)                 │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  2. 预处理层                                │
│     ├── fMRI 提取 ROI 时间序列              │
│     ├── EEG 源定位和 ROI 映射               │
│     └── 数据标准化和缓存                    │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  3. 图构建层                                │
│     ├── 节点生成（脑区表示）                │
│     ├── 边生成（结构/功能连接）            │
│     └── 异构图数据结构                      │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  4. 模型层                                  │
│     ├── 图编码器（GNN Encoder）             │
│     ├── 时序解码器（Temporal Decoder）      │
│     ├── 特征重建器（Feature Reconstructor） │
│     └── 时间对齐器（Aligner）               │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  5. 训练层                                  │
│     ├── 损失函数（多任务）                  │
│     ├── 优化器（Adam + 学习率调度）         │
│     └── 诊断和可视化                        │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  6. 输出层                                  │
│     ├── 训练好的模型                        │
│     ├── 重建的脑信号                        │
│     ├── 潜在表征（Latent）                  │
│     └── 可视化图表                          │
│                                             │
└─────────────────────────────────────────────┘
```

### 核心模块说明

#### 1. 配置系统 (`config/`)

负责管理所有超参数和系统设置。

```yaml
config/
├── default.yaml       # v4 默认配置（推荐）
├── v3_legacy.yaml     # v3 遗留配置（向后兼容）
└── export.yaml        # 导出配置
```

#### 2. 工作流模块 (`workflows/`)

实现不同的处理流程。

```python
workflows/
├── training.py        # 训练工作流
└── export_latent.py   # 导出工作流
```

#### 3. 数据映射器 (`mapper/`)

处理多模态数据的加载和映射。

```python
mapper/
├── atlas_mapper.py         # 脑图谱映射
├── bids_mapper.py          # BIDS 数据格式
├── eeg_mapper.py           # EEG 数据映射
├── eeg_roi_mapper.py       # EEG ROI 映射
├── multi_modal_mapper.py   # 多模态整合
└── dti_mapper.py           # DTI 连接映射
```

#### 4. 训练模块 (`train/`)

实现模型训练的核心逻辑。

```python
train/
├── hetero_trainer.py       # 异构图训练器
├── dynamic_hetero_gnn.py   # 动态异构 GNN
├── aligner.py              # 时间对齐模块
├── loss_helpers.py         # 损失函数辅助
└── embed_analysis.py       # 嵌入分析
```

#### 5. 工具模块 (`utils/`)

提供各种辅助功能。

```python
utils/
├── config.py           # 配置管理
├── logging_utils.py    # 日志系统
├── utils.py            # 通用工具
├── function.py         # 数据处理函数
├── analysis.py         # 分析工具
└── debug.py            # 调试工具
```

---

## 快速开始

### 环境准备

#### 1. 系统要求

- Python 3.8+
- CUDA 11.x+（可选，用于 GPU 加速）
- 内存：建议 16GB+
- 硬盘：建议 50GB+（用于数据缓存）

#### 2. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 安装依赖
pip install -r requirements.txt
```

**依赖清单**:
- `torch==2.4` - PyTorch 深度学习框架
- `torch-geometric==2.5.3` - 图神经网络库
- `numpy==1.26` - 数值计算
- `networkx==3.3` - 图数据结构
- `nilearn==0.10.4` - fMRI 数据处理
- `nibabel==5.2.1` - 神经影像数据读取
- `matplotlib==3.9` - 可视化
- `pyyaml>=6.0` - YAML 配置解析

### 数据准备

#### 数据结构

```
test_file3/                          # 数据根目录
├── sub-01/                          # 被试 1
│   ├── eeg/                         # EEG 数据
│   │   └── sub-01_task-*.fif
│   ├── func/                        # fMRI 数据
│   │   └── sub-01_task-*_bold.nii
│   ├── dwi/                         # DTI 数据
│   │   └── sub-01_acq-AP_dwi_connectome.npy
│   └── results/                     # 输出目录（自动创建）
│       ├── cache/                   # 缓存数据
│       ├── diagnostics/             # 诊断图表
│       └── hetero_gnn_trained.pt    # 训练模型
│
├── sub-02/                          # 被试 2
│   └── ...
│
└── schaefer200_mask_ready.json      # 脑图谱定义
```

#### 必需文件

1. **fMRI 数据**: NIfTI 格式 (*.nii 或 *.nii.gz)
2. **EEG 数据**: FIF 格式 (*.fif)
3. **DTI 连接矩阵**: NumPy 格式 (*.npy)
4. **脑图谱文件**: JSON 格式

### 第一次运行

#### 方式1: 使用新架构（推荐）

```bash
# 使用默认 v4 配置训练
python main.py train --config config/default.yaml

# 查看配置不运行（检查参数）
python main.py train --config config/default.yaml --dry-run
```

#### 方式2: 使用旧版本（兼容）

```bash
# 直接运行 v4 训练脚本
python main_v4.py
```

---

## 配置系统详解

### 配置文件结构

配置文件使用 YAML 格式，分为以下几个主要部分：

```yaml
# 基本信息
version: "v4"
description: "配置说明"

# 训练参数
training:
  warmup_epochs: 5           # 预热轮数
  warmup_run_epochs: 15      # 预热运行总轮数
  finetune_epochs: 80        # 微调轮数
  batch_size: 1              # 批次大小
  learning_rate: 0.0001      # 学习率
  weight_decay: 0.0001       # 权重衰减
  
  # 学习率倍数
  feature_lr_mul: 12.0       # 特征解码器学习率倍数
  scale_lr_mul: 10.0         # 缩放参数学习率倍数
  
  # 学习率调度器
  scheduler:
    type: "StepLR"           # 调度器类型
    step_size: 30            # 步长
    gamma: 0.7               # 衰减率

# 模型架构
model:
  hidden_dim: 128            # 隐藏层维度
  num_layers: 2              # 编码器层数
  decoder_layers: 3          # 解码器层数
  dropout: 0.1               # Dropout 率
  
  encoder:
    use_batch_norm: true     # 使用批归一化
    activation: "relu"       # 激活函数
  
  decoder:
    use_residual: true       # 使用残差连接
    activation: "relu"       # 激活函数

# 损失函数权重
loss:
  recon_weight: 1.0          # 重建损失权重
  recon_norm_weight: 3.0     # 归一化重建损失权重
  recon_corr_weight: 2.0     # 相关性损失权重
  recon_feat_var_weight: 0.02 # 特征方差正则化权重
  temp_weight: 5.0           # 时间对齐损失权重

# 自动对齐设置
alignment:
  auto_align: true           # 启用自动对齐
  auto_align_max_lag: 150    # 最大滞后帧数

# 数据处理
data:
  use_cache: true            # 使用缓存
  cache_dir: "cache"         # 缓存目录
  
  atlas:
    name: "Schaefer200"      # 图谱名称
    n_parcels: 200           # 脑区数量
    networks: 7              # 网络数量

# 诊断和日志
diagnostics:
  enabled: true              # 启用诊断
  save_plots: true           # 保存图表
  diagnostic_dir: "diagnostics" # 诊断目录
  plot_nodes: [0, 1, 2]      # 要绘制的节点

# 输出设置
output:
  save_checkpoints: true     # 保存检查点
  checkpoint_interval: 20    # 检查点间隔
  save_final_model: true     # 保存最终模型
  results_dir: "results"     # 结果目录
```

### 配置版本对比

#### default.yaml (v4 - 推荐)

**特点**: 更强的时间对齐、更深的解码器、更长的训练

| 参数 | 值 | 说明 |
|------|-----|------|
| `finetune_epochs` | 80 | 微调轮数（v3 为 40） |
| `temp_weight` | 5.0 | 时间对齐权重（v3 为 1.0） |
| `decoder_layers` | 3 | 解码器层数（v3 为 2） |
| `warmup_run_epochs` | 15 | 预热总轮数（v3 为 10） |

**适用场景**: 
- 新项目和实验
- 追求更好的重建质量
- 有充足的计算资源

#### v3_legacy.yaml (v3 - 兼容)

**特点**: 训练更快、资源消耗更少

**适用场景**:
- 复现旧实验结果
- 快速原型验证
- 计算资源有限

### 自定义配置

#### 创建自己的配置

1. 复制默认配置：
```bash
cp config/default.yaml config/my_experiment.yaml
```

2. 编辑配置文件：
```yaml
# config/my_experiment.yaml
version: "my_experiment"
description: "我的实验配置"

training:
  warmup_epochs: 10        # 增加预热
  finetune_epochs: 100     # 增加微调
  learning_rate: 0.0005    # 调整学习率

model:
  hidden_dim: 256          # 增加模型容量
  decoder_layers: 4        # 更深的解码器

loss:
  temp_weight: 10.0        # 更强的时间对齐
```

3. 使用自定义配置：
```bash
python main.py train --config config/my_experiment.yaml
```

---

## 工作流程说明

### 训练工作流

训练工作流分为以下几个阶段：

#### 阶段 1: 数据加载

```
数据加载阶段
├── 1. 发现任务
│   ├── 扫描 EEG 目录
│   ├── 扫描 fMRI 目录
│   └── 记录任务名称
│
├── 2. 加载或生成数据
│   ├── 检查缓存
│   │   ├── stim.pt (刺激数据)
│   │   ├── eeg_data.pt (EEG 数据)
│   │   └── hetero_graphs.pt (异构图)
│   │
│   └── 如果缓存不存在
│       ├── 生成刺激数据
│       ├── 加载 fMRI 数据
│       │   ├── 读取 NIfTI 文件
│       │   ├── 应用脑图谱
│       │   └── 提取 ROI 时间序列
│       │
│       ├── 加载 EEG 数据
│       │   ├── 读取 FIF 文件
│       │   ├── 源定位
│       │   └── ROI 映射
│       │
│       ├── 加载 DTI 连接
│       │
│       └── 构建异构图
│           ├── 节点特征
│           ├── 边关系
│           └── HeteroData 结构
│
└── 3. 保存缓存（如果启用）
```

**时间估计**:
- 首次运行（无缓存）: 5-15 分钟
- 使用缓存: 10-30 秒

#### 阶段 2: 模型初始化

```
模型初始化阶段
├── 1. 创建训练器
│   ├── 图编码器 (Graph Encoder)
│   ├── 时序解码器 (Temporal Decoder)
│   ├── 特征重建器 (Feature Reconstructor)
│   └── 时间对齐器 (Aligner)
│
├── 2. 配置优化器
│   ├── 基础参数组 (学习率: lr)
│   ├── 特征参数组 (学习率: lr × feature_lr_mul)
│   └── 缩放参数组 (学习率: lr × scale_lr_mul)
│
└── 3. 配置学习率调度器
    └── StepLR (step_size=30, gamma=0.7)
```

#### 阶段 3: 初始诊断

```
初始诊断阶段
├── 1. 前向诊断
│   ├── 运行前向传播
│   ├── 检查梯度
│   └── 输出统计信息
│
└── 2. 诊断图表（如果启用）
    ├── fMRI 节点诊断图
    ├── EEG 节点诊断图
    └── 保存到 diagnostics/ 目录
```

#### 阶段 4: 预热训练

```
预热训练阶段 (warmup_run_epochs 轮)
├── 目标: 让模型适应数据分布
├── 特点: 较低的学习率，稳定训练
│
└── 每轮训练
    ├── 前向传播
    ├── 计算损失
    │   ├── 重建损失 (MSE)
    │   ├── 归一化重建损失
    │   ├── 相关性损失
    │   └── 特征方差正则化
    ├── 反向传播
    ├── 参数更新
    └── 日志记录
```

**输出示例**:
```
Epoch 1/15 | Loss: 2.456 | Recon: 1.234 | Norm: 0.567 | Corr: 0.345
Epoch 2/15 | Loss: 2.123 | Recon: 1.089 | Norm: 0.512 | Corr: 0.322
...
```

#### 阶段 5: 互相关分析

```
互相关分析阶段
├── 目的: 评估重建信号与目标信号的时间对齐
│
└── 分析步骤
    ├── 提取重建信号 (fMRI 节点 0, 特征 0)
    ├── 提取目标信号
    ├── 计算互相关
    ├── 找出最佳滞后 (best_lag)
    └── 输出相关系数 (best_corr)
```

**输出示例**:
```
[XCORR] nt=fmri node=0 feat=0 best_lag=15 best_corr=0.8234
```

#### 阶段 6: 微调训练

```
微调训练阶段 (finetune_epochs 轮)
├── 目标: 优化重建质量和时间对齐
├── 特点: 增加时间对齐损失权重 (temp_weight)
│
└── 每轮训练
    ├── 前向传播
    ├── 计算损失（加入时间对齐损失）
    │   ├── 重建损失
    │   ├── 归一化重建损失
    │   ├── 相关性损失
    │   ├── 特征方差正则化
    │   └── 时间对齐损失 × temp_weight
    ├── 反向传播
    ├── 参数更新
    ├── 学习率调度
    └── 日志记录
```

#### 阶段 7: 模型保存

```
模型保存阶段
├── 保存位置: results/hetero_gnn_trained.pt
│
└── 保存内容
    ├── 模型权重
    ├── 优化器状态
    ├── 训练配置
    └── 元数据
```

### 导出工作流

```
导出工作流（开发中）
├── 1. 加载训练好的模型
├── 2. 运行推理
├── 3. 提取潜在表征
└── 4. 导出为 JSON/NPY 格式
```

---

## 超参数调优指南

### 关键超参数说明

#### 1. 学习率相关

**基础学习率** (`training.learning_rate`)
- **默认值**: 0.0001
- **范围**: 0.00001 ~ 0.001
- **调优建议**:
  - 太大: 训练不稳定，损失震荡
  - 太小: 训练缓慢，可能欠拟合
  - 推荐: 从 0.0001 开始，根据损失曲线调整

**特征学习率倍数** (`training.feature_lr_mul`)
- **默认值**: 12.0
- **作用**: 特征解码器的学习率倍数
- **调优建议**:
  - 解码器需要更快学习: 增大 (15.0 ~ 20.0)
  - 训练不稳定: 减小 (8.0 ~ 10.0)

**缩放学习率倍数** (`training.scale_lr_mul`)
- **默认值**: 10.0
- **作用**: 缩放参数的学习率倍数
- **调优建议**: 通常与 feature_lr_mul 保持相近

#### 2. 训练轮数

**预热轮数** (`training.warmup_epochs`)
- **默认值**: 5
- **作用**: 初始稳定训练的轮数
- **调优建议**:
  - 数据复杂: 增加到 8-10
  - 快速验证: 减少到 3

**预热运行轮数** (`training.warmup_run_epochs`)
- **默认值**: 15 (warmup_epochs + 10)
- **作用**: 预热阶段总训练轮数
- **调优建议**: 通常设为 warmup_epochs 的 2-3 倍

**微调轮数** (`training.finetune_epochs`)
- **默认值**: 80
- **范围**: 40 ~ 150
- **调优建议**:
  - 追求质量: 增加到 100-150
  - 快速实验: 减少到 40-60
  - 观察损失是否收敛

#### 3. 模型容量

**隐藏层维度** (`model.hidden_dim`)
- **默认值**: 128
- **范围**: 64 ~ 512
- **调优建议**:
  - 数据复杂/脑区多: 增加到 256
  - 数据简单/计算受限: 减少到 64
  - 影响: 更大的模型容量但需要更多内存

**解码器层数** (`model.decoder_layers`)
- **默认值**: 3
- **范围**: 2 ~ 5
- **调优建议**:
  - 重建质量不佳: 增加到 4
  - 过拟合: 减少到 2
  - 训练慢: 减少层数

#### 4. 损失权重

**重建损失权重** (`loss.recon_weight`)
- **默认值**: 1.0
- **作用**: 基础 MSE 损失的权重
- **调优建议**: 通常保持 1.0，调整其他权重

**归一化重建损失权重** (`loss.recon_norm_weight`)
- **默认值**: 3.0
- **作用**: 归一化空间的 MSE 损失权重
- **调优建议**:
  - 信号幅度差异大: 增加到 5.0
  - 过度强调归一化: 减少到 2.0

**相关性损失权重** (`loss.recon_corr_weight`)
- **默认值**: 2.0
- **作用**: Pearson 相关性损失的权重
- **调优建议**:
  - 重视时间模式: 增加到 3.0-4.0
  - 过拟合相关性: 减少到 1.0

**时间对齐损失权重** (`loss.temp_weight`)
- **默认值**: 5.0 (v4), 1.0 (v3)
- **作用**: 时间对齐损失的权重
- **调优建议**:
  - 时间对齐重要: 增加到 8.0-10.0
  - 训练不稳定: 减少到 3.0
  - **v4 的关键改进**: 这是 v4 相比 v3 的主要提升

**特征方差正则化权重** (`loss.recon_feat_var_weight`)
- **默认值**: 0.02
- **作用**: 防止特征崩溃的正则化
- **调优建议**:
  - 特征崩溃: 增加到 0.05
  - 限制太强: 减少到 0.01

### 调优策略

#### 策略 1: 快速验证

**目标**: 快速检查流程是否正常

```yaml
# config/quick_test.yaml
training:
  warmup_epochs: 2
  warmup_run_epochs: 5
  finetune_epochs: 10
  
model:
  hidden_dim: 64
  decoder_layers: 2
```

**预期时间**: 10-20 分钟

#### 策略 2: 标准训练

**目标**: 平衡质量和速度

```yaml
# config/standard.yaml
training:
  warmup_epochs: 5
  warmup_run_epochs: 15
  finetune_epochs: 60
  
model:
  hidden_dim: 128
  decoder_layers: 3
  
loss:
  temp_weight: 5.0
```

**预期时间**: 1-2 小时

#### 策略 3: 高质量训练

**目标**: 追求最佳重建质量

```yaml
# config/high_quality.yaml
training:
  warmup_epochs: 8
  warmup_run_epochs: 20
  finetune_epochs: 120
  learning_rate: 0.00005  # 更小的学习率
  
model:
  hidden_dim: 256
  decoder_layers: 4
  
loss:
  temp_weight: 8.0
  recon_norm_weight: 4.0
```

**预期时间**: 3-5 小时

#### 策略 4: 调试模式

**目标**: 快速定位问题

```yaml
# config/debug.yaml
training:
  warmup_epochs: 1
  warmup_run_epochs: 2
  finetune_epochs: 3
  
diagnostics:
  enabled: true
  save_plots: true
```

运行时添加调试日志：
```bash
python main.py train --config config/debug.yaml --log-level DEBUG
```

### 调优流程

#### 步骤 1: 基线运行

使用默认配置运行，记录结果：
```bash
python main.py train --config config/default.yaml
```

记录关键指标：
- 最终损失值
- 最佳互相关系数 (best_corr)
- 训练时间

#### 步骤 2: 单参数调优

每次只调整一个参数，观察影响：

```bash
# 实验 1: 增加训练轮数
# 修改 config/exp1.yaml: finetune_epochs: 100
python main.py train --config config/exp1.yaml

# 实验 2: 增加模型容量
# 修改 config/exp2.yaml: hidden_dim: 256
python main.py train --config config/exp2.yaml

# 实验 3: 调整时间对齐权重
# 修改 config/exp3.yaml: temp_weight: 8.0
python main.py train --config config/exp3.yaml
```

#### 步骤 3: 组合优化

将有效的参数组合：

```yaml
# config/optimized.yaml
# 基于实验结果的最优配置
training:
  finetune_epochs: 100     # 来自实验 1
  
model:
  hidden_dim: 256          # 来自实验 2
  
loss:
  temp_weight: 8.0         # 来自实验 3
```

#### 步骤 4: 验证和调整

在不同被试上验证最优配置的稳定性。

---

## 命令行使用

### 基本命令

#### 训练

```bash
# 使用默认 v4 配置
python main.py train --config config/default.yaml

# 使用 v3 遗留配置
python main.py train --config config/v3_legacy.yaml

# 使用自定义配置
python main.py train --config config/my_config.yaml

# 指定数据目录
python main.py train --config config/default.yaml --base-dir /path/to/data

# 调试模式（详细日志）
python main.py train --config config/default.yaml --log-level DEBUG

# 禁用 CUDA（仅使用 CPU）
python main.py train --config config/default.yaml --no-cuda
```

#### 导出

```bash
# 导出潜在表征
python main.py export --config config/export.yaml --subject sub-01

# 指定输出目录
python main.py export --config config/export.yaml --subject sub-01 --output-dir exports/
```

#### 查看配置

```bash
# Dry-run 模式（查看配置不运行）
python main.py train --config config/default.yaml --dry-run
```

**输出示例**:
```yaml
================================================================================
Configuration:
================================================================================
version: v4
description: Improved training with stronger temporal alignment and deeper decoder
training:
  warmup_epochs: 5
  warmup_run_epochs: 15
  finetune_epochs: 80
  ...
================================================================================
```

### 完整参数列表

```bash
python main.py --help
```

**参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `workflow` | 工作流类型 (train/export/infer) | 必需 |
| `--config` | 配置文件路径 | 必需 |
| `--base-dir` | 数据根目录 | test_file3/ |
| `--subject` | 被试 ID（用于 export/infer） | None |
| `--output-dir` | 输出目录 | 从配置读取 |
| `--log-level` | 日志级别 | INFO |
| `--no-cuda` | 禁用 CUDA | False |
| `--dry-run` | 只显示配置不运行 | False |

### 使用旧版本接口

如果需要使用旧版本接口（完全兼容）：

```bash
# v4 训练
python main_v4.py

# 导出潜在表征
python main_export_latent.py
```

---

## 数据处理流程

### fMRI 数据处理

```
fMRI 处理流程
├── 1. 读取 NIfTI 文件
│   └── 使用 nibabel 加载 4D 数据
│
├── 2. 应用脑图谱
│   ├── 加载 Schaefer200 图谱
│   ├── 使用 NiftiLabelsMasker
│   └── 提取每个脑区的平均时间序列
│
├── 3. 标准化
│   ├── Z-score 标准化（可选）
│   └── 形状: (n_regions, n_timepoints, n_features)
│
└── 4. 保存为 Tensor
    └── 形状: torch.Tensor (200, T, 1)
```

**关键参数**:
- **脑区数量**: 200 (Schaefer200 图谱)
- **时间点**: 取决于 fMRI 采集长度
- **特征维度**: 1（平均 BOLD 信号）

### EEG 数据处理

```
EEG 处理流程
├── 1. 读取 FIF 文件
│   └── 使用 MNE-Python 加载 Raw 数据
│
├── 2. 预处理
│   ├── 滤波（可选）
│   ├── 去除坏道
│   └── 参考电极设置
│
├── 3. 源定位
│   ├── 构建正向模型
│   ├── 计算逆解
│   └── 得到源空间活动
│
├── 4. ROI 映射
│   ├── 将源空间映射到脑区
│   ├── 每个脑区的 EEG 信号
│   └── 形状: (n_regions, n_timepoints, n_features)
│
└── 5. 保存为 Tensor
    └── 形状: torch.Tensor (200, T, F)
```

**关键参数**:
- **脑区数量**: 200 (与 fMRI 对应)
- **时间点**: EEG 采样点数
- **特征维度**: 取决于 EEG 处理方式

### DTI 连接处理

```
DTI 处理流程
├── 1. 加载连接矩阵
│   └── NumPy 格式 (n_regions, n_regions)
│
├── 2. 归一化（可选）
│   ├── 对角线设为 0
│   └── Min-max 归一化
│
└── 3. 用于边权重
    └── 构建异构图的边特征
```

### 图构建流程

```
异构图构建流程
├── 1. 节点特征
│   ├── fMRI 节点: (200, T_fmri, 1)
│   ├── EEG 节点: (200, T_eeg, F_eeg)
│   └── 刺激节点: (n_stim, T_stim, F_stim)
│
├── 2. 边关系
│   ├── fmri-to-fmri: 结构连接（DTI）
│   ├── eeg-to-eeg: 结构连接（DTI）
│   ├── fmri-to-eeg: 同脑区连接
│   ├── eeg-to-fmri: 同脑区连接
│   └── stim-to-fmri, stim-to-eeg: 刺激影响
│
├── 3. HeteroData 结构
│   ├── data['fmri'].x: fMRI 节点特征
│   ├── data['eeg'].x: EEG 节点特征
│   ├── data['stim'].x: 刺激特征
│   ├── data['fmri', 'to', 'fmri'].edge_index: 边索引
│   └── ...其他边关系
│
└── 4. 时间序列
    ├── data['fmri'].x_seq: fMRI 时间序列
    ├── data['eeg'].x_seq: EEG 时间序列
    └── data['stim'].x_seq: 刺激时间序列
```

---

## 最佳实践

### 数据组织

#### 1. 使用 BIDS 格式

建议遵循 BIDS（Brain Imaging Data Structure）标准：

```
dataset/
├── sub-01/
│   ├── anat/
│   ├── func/
│   ├── eeg/
│   └── dwi/
└── derivatives/
    └── twinbrain/
        └── sub-01/
            └── results/
```

#### 2. 文件命名规范

```
# fMRI
sub-01_task-rest_bold.nii.gz
sub-01_task-motor_bold.nii.gz

# EEG
sub-01_task-rest_eeg.fif
sub-01_task-motor_eeg.fif

# DTI
sub-01_acq-AP_dwi_connectome.npy
```

### 训练技巧

#### 1. 使用缓存

**启用缓存** (默认):
```yaml
data:
  use_cache: true
  cache_dir: "cache"
```

**好处**:
- 首次运行后，后续运行快 10-50 倍
- 节省重复的数据处理时间

**清除缓存**（当数据更新时）:
```bash
rm -rf test_file3/sub-*/results/cache/
```

#### 2. 监控训练

**查看实时日志**:
```bash
# 训练时
python main.py train --config config/default.yaml

# 查看日志文件
tail -f logs/twinbrain_*.log
```

**关键指标**:
- `Loss`: 总损失（应该持续下降）
- `Recon`: 重建损失
- `best_corr`: 互相关系数（应该接近 1.0）

**正常训练的损失曲线**:
```
Epoch 1:  Loss: 2.50 → 快速下降阶段
Epoch 10: Loss: 1.80 → 
Epoch 20: Loss: 1.20 → 稳定下降阶段
Epoch 40: Loss: 0.85 → 
Epoch 60: Loss: 0.65 → 收敛阶段
Epoch 80: Loss: 0.58 → 
```

#### 3. 检查点管理

**自动保存**:
```yaml
output:
  save_checkpoints: true
  checkpoint_interval: 20  # 每 20 轮保存一次
```

**加载检查点**（功能开发中）:
```python
# 未来支持
trainer.load_checkpoint('results/checkpoint_epoch_60.pt')
```

### 诊断和调试

#### 1. 启用诊断

```yaml
diagnostics:
  enabled: true
  save_plots: true
  diagnostic_dir: "diagnostics"
```

**生成的图表**:
- `fmri_node0_diagnostics.png`: fMRI 节点诊断
- `eeg_node0_diagnostics.png`: EEG 节点诊断
- `recon_vs_target_fmri.png`: 重建vs目标对比

#### 2. 使用调试模式

```bash
python main.py train --config config/default.yaml --log-level DEBUG
```

**输出更详细的信息**:
- 每个模块的输入输出形状
- 梯度统计信息
- 中间计算结果

#### 3. 可视化结果

**查看诊断图表**:
```bash
ls test_file3/sub-01/results/diagnostics/

# 使用图像查看器
eog test_file3/sub-01/results/diagnostics/*.png
```

### 性能优化

#### 1. GPU 加速

**检查 CUDA 可用性**:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**预期加速**:
- CPU: 1x（基准）
- GPU: 5-10x（取决于 GPU 型号）

#### 2. 批处理大小

由于异构图的特性，当前版本 `batch_size=1`。

#### 3. 内存管理

**监控内存使用**:
```bash
# 训练时在另一个终端运行
watch -n 1 nvidia-smi  # GPU 内存
htop                    # CPU 内存
```

**减少内存使用**:
- 减小 `model.hidden_dim`
- 减小 `model.decoder_layers`
- 禁用诊断: `diagnostics.enabled: false`

---

## 常见问题

### Q1: 训练时出现 CUDA out of memory 错误

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:

1. 减小模型容量：
```yaml
model:
  hidden_dim: 64      # 从 128 减少
  decoder_layers: 2   # 从 3 减少
```

2. 使用 CPU：
```bash
python main.py train --config config/default.yaml --no-cuda
```

3. 清理 GPU 缓存：
```python
import torch
torch.cuda.empty_cache()
```

### Q2: 训练速度很慢

**问题**: 训练一轮需要很长时间

**可能原因和解决方案**:

1. **没有使用 GPU**:
```bash
# 检查
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，安装 CUDA 版本的 PyTorch
```

2. **首次运行数据加载慢**:
- 正常现象，启用缓存后会快很多
- 确保 `data.use_cache: true`

3. **诊断功能占用时间**:
```yaml
diagnostics:
  enabled: false  # 训练时禁用
```

### Q3: 重建质量不理想

**问题**: `best_corr` 值很低（< 0.5）

**调优建议**:

1. **增加训练轮数**:
```yaml
training:
  finetune_epochs: 120  # 从 80 增加
```

2. **增强时间对齐**:
```yaml
loss:
  temp_weight: 8.0  # 从 5.0 增加
```

3. **增加模型容量**:
```yaml
model:
  hidden_dim: 256     # 从 128 增加
  decoder_layers: 4   # 从 3 增加
```

4. **检查数据质量**:
- 确保 fMRI 和 EEG 数据质量良好
- 检查是否有坏道或伪迹

### Q4: 如何复现旧实验结果？

**问题**: 需要复现使用 v3 配置的旧实验

**解决方案**:

```bash
# 使用 v3_legacy 配置
python main.py train --config config/v3_legacy.yaml
```

或使用旧版本接口：
```bash
# 使用原始 main_v4.py（已配置为 v4）
# 需要手动修改代码中的参数以匹配 v3
```

### Q5: 缓存数据何时需要清除？

**需要清除缓存的情况**:

1. **数据文件更新**:
```bash
rm -rf test_file3/sub-*/results/cache/
```

2. **预处理参数改变**:
- 更换了脑图谱
- 修改了 EEG 处理参数

3. **发现缓存损坏**:
```bash
# 清除特定被试的缓存
rm -rf test_file3/sub-01/results/cache/

# 清除所有缓存
find test_file3 -name cache -type d -exec rm -rf {} +
```

### Q6: 如何并行处理多个被试？

**当前版本**: 串行处理（一个接一个）

**未来支持**: 多进程并行

**临时方案**:
```bash
# 手动为每个被试创建配置，分别运行
python main.py train --config config/default.yaml --base-dir test_file3/sub-01 &
python main.py train --config config/default.yaml --base-dir test_file3/sub-02 &
```

### Q7: 导出功能如何使用？

**当前状态**: 基础框架已实现，完整功能开发中

**临时方案**:
```bash
# 使用旧版本导出脚本
python main_export_latent.py
```

### Q8: 如何添加新的损失函数？

**步骤**:

1. 在 `train/loss_helpers.py` 中定义新损失
2. 在配置文件中添加权重
3. 在训练器中集成

**示例**:
```python
# train/loss_helpers.py
def my_custom_loss(pred, target):
    return torch.mean((pred - target) ** 2)
```

```yaml
# config/my_config.yaml
loss:
  my_custom_weight: 1.0
```

### Q9: 训练中断如何恢复？

**当前版本**: 不支持自动恢复

**建议**:
- 使用较短的训练轮数测试
- 确保系统稳定后再长时间训练

**未来支持**: 从检查点恢复训练

---

## 附录

### A. 文件清单

#### 核心文件

```
twinbrain/
├── main.py                         # 统一入口点 ⭐
├── main_v4.py                      # v4 训练脚本（兼容）
├── main_export_latent.py           # 导出脚本（兼容）
│
├── config/                         # 配置目录 ⭐
│   ├── default.yaml                # v4 默认配置
│   ├── v3_legacy.yaml              # v3 遗留配置
│   └── export.yaml                 # 导出配置
│
├── workflows/                      # 工作流模块 ⭐
│   ├── __init__.py
│   ├── training.py                 # 训练工作流
│   └── export_latent.py            # 导出工作流
│
├── utils/                          # 工具模块
│   ├── config.py                   # 配置管理 ⭐
│   ├── logging_utils.py            # 日志系统 ⭐
│   ├── utils.py                    # 通用工具
│   ├── function.py                 # 数据处理函数
│   ├── analysis.py                 # 分析工具
│   └── debug.py                    # 调试工具
│
├── train/                          # 训练模块
│   ├── hetero_trainer.py           # 异构图训练器
│   ├── dynamic_hetero_gnn.py       # 动态异构GNN
│   ├── aligner.py                  # 对齐模块
│   ├── loss_helpers.py             # 损失函数
│   └── ...
│
├── mapper/                         # 数据映射模块
│   ├── atlas_mapper.py             # 图谱映射
│   ├── eeg_mapper.py               # EEG映射
│   ├── bids_mapper.py              # BIDS格式
│   └── ...
│
└── preprocess/                     # 预处理模块
    ├── eeg_preprocessor.py
    └── fmri_preprocessor.py
```

### B. 依赖版本

```
# requirements.txt
networkx==3.3
numpy==1.26
torch==2.4
torch-geometric==2.5.3
matplotlib==3.9
nilearn==0.10.4
nibabel==5.2.1
pyyaml>=6.0
mne>=1.0  # EEG 处理
scipy>=1.10  # 科学计算
scikit-learn>=1.3  # 机器学习工具
```

### C. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 异构图 | Heterogeneous Graph | 包含多种节点和边类型的图 |
| ROI | Region of Interest | 感兴趣区域，这里指脑区 |
| BOLD | Blood-Oxygen-Level-Dependent | 血氧水平依赖，fMRI 测量的信号 |
| 时序解码器 | Temporal Decoder | 解码时间序列的神经网络模块 |
| 潜在表征 | Latent Representation | 神经网络学到的抽象特征 |
| 互相关 | Cross-correlation | 衡量两个信号时间对齐的指标 |
| DTI | Diffusion Tensor Imaging | 弥散张量成像 |
| 源定位 | Source Localization | 从头皮EEG推断大脑源活动 |

### D. 参考资源

#### 论文

- Schaefer et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex
- PyTorch Geometric 文档: https://pytorch-geometric.readthedocs.io/

#### 工具

- MNE-Python: https://mne.tools/
- Nilearn: https://nilearn.github.io/
- NetworkX: https://networkx.org/

---

## 📧 获取帮助

### 文档

- **快速开始**: 本文档 [快速开始](#快速开始) 部分
- **设计理念**: `REFACTORING_SUMMARY_CN.md`
- **实施细节**: `REFACTORING_IMPLEMENTATION.md`
- **完成报告**: `REFACTORING_COMPLETE.md`

### 反馈

- **Bug 报告**: 提交 GitHub Issue
- **功能建议**: 提交 GitHub Issue
- **贡献代码**: 提交 Pull Request

---

## 🎓 最后的话

TwinBrain 项目采用现代化的配置驱动架构，旨在提供灵活、可扩展、易于使用的脑成像数据分析平台。

**关键要点**:

1. ✅ **配置驱动**: 通过修改 YAML 文件调整所有参数
2. ✅ **模块化**: 清晰的代码组织，易于维护和扩展
3. ✅ **向后兼容**: 保留旧版本接口，平滑迁移
4. ✅ **完整文档**: 详细的使用指南和最佳实践

**开始使用**:

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
# 将数据放在 test_file3/ 目录

# 3. 运行训练
python main.py train --config config/default.yaml

# 4. 查看结果
ls test_file3/sub-01/results/
```

祝您使用愉快！🎉

---

**文档版本**: 1.0  
**更新日期**: 2026-01-30  
**维护团队**: TwinBrain Development Team
