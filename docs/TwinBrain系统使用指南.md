# TwinBrain 数字孪生脑系统 - 使用指南

## 📖 目录

1. [项目简介](#项目简介)
2. [系统架构](#系统架构)
3. [安装和配置](#安装和配置)
4. [数据准备](#数据准备)
5. [训练流程](#训练流程)
6. [预测和推理](#预测和推理)
7. [Unity前端集成](#unity前端集成)
8. [配置参数详解](#配置参数详解)
9. [命令行使用](#命令行使用)
10. [常见问题](#常见问题)

---

## 项目简介

### 什么是 TwinBrain？

TwinBrain（数字孪生脑）是一个基于深度学习的多模态大脑活动建模与预测系统。该系统能够：

- 🧠 **整合多模态脑成像数据**：fMRI、EEG、DTI等
- 🔮 **预测大脑未来状态**：基于当前状态和刺激输入预测未来活动
- 🎯 **模拟虚拟刺激**：支持施加虚拟扰动并观察系统响应
- 🌐 **Unity可视化**：输出JSON格式供前端3D可视化展示
- 📊 **多模态融合**：异构图神经网络建模脑区间连接

### 核心特性

#### 1. 多模态数据处理
- **fMRI数据**：提取脑区时间序列，建立功能连接
- **EEG数据**：源定位后映射到脑区，提供高时间分辨率
- **DTI数据**：提供结构连接矩阵，指导图构建

#### 2. 异构图神经网络
- 使用PyTorch Geometric构建异构图
- 节点表示不同模态的脑区
- 边表示结构和功能连接
- 支持跨模态信息传递

#### 3. 时序预测能力
- **自回归多步预测**：基于历史状态预测未来脑活动
- 支持多步预测（训练中已实现）
- 使用滑动窗口和teacher forcing进行训练
- 注意：当前版本仅支持单模态内预测，不支持模态间双向预测（如fMRI→EEG）

#### 4. Unity集成
- 输出标准JSON格式
- 包含脑区活跃度、连接强度等信息
- 支持实时或批量导出

---

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    TwinBrain 系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                           │
│  │ 数据输入层  │                                            │
│  │             │                                            │
│  │  • fMRI     │  → 预处理  →  脑区时间序列               │
│  │  • EEG      │  → 源定位  →  脑区电位序列               │
│  │  • DTI      │  → 提取    →  结构连接矩阵               │
│  │  • Atlas    │  → 加载    →  脑区定义                   │
│  └─────────────┘                                           │
│         ↓                                                   │
│  ┌─────────────┐                                           │
│  │ 图构建层    │                                            │
│  │             │                                            │
│  │  • 节点生成 │  每个脑区 = 一个节点                      │
│  │  • 边生成   │  DTI连接 / 功能连接                       │
│  │  • 异构图   │  支持多模态节点                           │
│  └─────────────┘                                           │
│         ↓                                                   │
│  ┌─────────────┐                                           │
│  │ 模型层      │                                            │
│  │             │                                            │
│  │  • 图编码器 │  提取空间特征                             │
│  │  • 时序建模 │  GRU建模时间依赖                          │
│  │  • 特征解码 │  重建原始信号                             │
│  │  • 时间对齐 │  跨模态对齐                               │
│  │  • 预测器   │  预测未来状态                             │
│  └─────────────┘                                           │
│         ↓                                                   │
│  ┌─────────────┐                                           │
│  │ 输出层      │                                            │
│  │             │                                            │
│  │  • 训练模型 │  保存的checkpoint                         │
│  │  • 重建信号 │  用于评估质量                             │
│  │  • 预测结果 │  未来状态                                 │
│  │  • JSON导出 │  供Unity可视化                            │
│  └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块说明

#### 1. 数据映射器 (mapper/)
- `atlas_mapper.py`: 脑图谱管理
- `bids_mapper.py`: BIDS格式数据处理
- `eeg_mapper.py`: EEG数据映射到脑区
- `multi_modal_mapper.py`: 多模态数据整合

#### 2. 训练模块 (train/)
- `dynamic_hetero_gnn.py`: 主模型定义
- `hetero_trainer.py`: 训练器
- `coder.py`: 编码器和解码器
- `aligner.py`: 跨模态对齐
- `loss_helpers.py`: 损失函数

#### 3. 工作流 (workflows/)
- `training.py`: 训练工作流
- `export_latent.py`: 特征导出工作流

#### 4. 工具模块 (utils/)
- `config.py`: 配置管理
- `logging_utils.py`: 日志系统
- `function.py`: 数据处理函数
- `analysis.py`: 分析工具

---

## 安装和配置

### 环境要求

- Python 3.8+
- CUDA 11.0+（可选，用于GPU加速）
- 8GB+ RAM
- 10GB+ 磁盘空间

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python main.py --help
```

### 依赖包说明

主要依赖：
- `torch`: PyTorch深度学习框架
- `torch-geometric`: 图神经网络库
- `numpy`: 数值计算
- `nibabel`: 神经影像数据读取
- `mne`: EEG数据处理
- `scipy`: 科学计算
- `pyyaml`: 配置文件解析
- `matplotlib`: 可视化

---

## 数据准备

### 数据结构

推荐使用BIDS格式组织数据：

```
data/
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   ├── func/
│   │   ├── sub-01_task-rest_bold.nii.gz
│   │   └── sub-01_task-task_bold.nii.gz
│   ├── eeg/
│   │   └── sub-01_task-rest_eeg.fif
│   └── dwi/
│       └── sub-01_dwi.nii.gz
├── sub-02/
│   └── ...
└── derivatives/
    └── dti/
        ├── sub-01_dti_matrix.npy
        └── sub-02_dti_matrix.npy
```

### 脑图谱文件

需要准备脑图谱JSON文件，示例：

```json
{
  "name": "Schaefer200",
  "regions": {
    "1": {"label": "7Networks_LH_Vis_1", "xyz": [-5, -85, 5]},
    "2": {"label": "7Networks_LH_Vis_2", "xyz": [-10, -90, 10]},
    ...
  }
}
```

### fMRI数据预处理

fMRI数据应预先完成：
1. 运动校正
2. 配准到MNI空间
3. 空间平滑
4. 时间滤波

### EEG数据预处理

EEG数据应完成：
1. 伪迹去除
2. 重参考
3. 滤波
4. 坏导排除

---

## 训练流程

### 快速开始

使用默认配置训练：

```bash
python main.py train --config config/default.yaml
```

### 训练流程详解

#### 第1阶段：数据加载和图构建

```python
# 系统会自动执行：
1. 加载fMRI数据 → 提取脑区时间序列
2. 加载EEG数据 → 源定位并映射到脑区
3. 加载DTI矩阵 → 构建结构连接
4. 构建异构图 → 整合多模态数据
```

输出：`HeteroData`对象，包含：
- `data['fmri'].x`: fMRI节点特征 [N_regions, T_time, F_features]
- `data['eeg'].x`: EEG节点特征
- `data['fmri', 'connects', 'fmri'].edge_index`: fMRI连接
- ...

#### 第2阶段：模型初始化

模型结构：
```
DynamicHeteroGNN
├── GraphEncoder: 图卷积层 (SAGEConv)
│   └── 多层异构卷积
├── GRU: 时序建模
│   └── 处理时间依赖
├── TemporalDecoder: 时间解码器
│   └── 重建时间序列
├── NodeDecoder: 节点解码器
│   └── 恢复原始特征
└── Aligner: 跨模态对齐
    └── 对齐不同模态
```

#### 第3阶段：Warmup训练

目的：让模型学习基本的重建能力

```yaml
# 配置参数
training:
  warmup_epochs: 5
  warmup_run_epochs: 10
```

损失函数：
- 重建损失（reconstruction loss）
- 时间一致性损失（temporal consistency）

#### 第4阶段：Fine-tuning训练

目的：优化时序对齐和跨模态融合

```yaml
# 配置参数
training:
  finetune_epochs: 80
  learning_rate: 0.0001
```

增强的损失：
- 重建损失（权重调整）
- 时间对齐损失（增强）
- 跨模态对齐损失
- 可选：多步预测损失（需配置启用）

#### 第5阶段：诊断和评估

自动生成：
- 重建质量图
- 互相关分析
- 损失曲线
- 诊断报告

保存位置：`diagnostics/`目录

---

## 预测和推理

### 单步预测

从当前状态预测下一时刻：

```python
from workflows.inference import predict_next_state

# 加载训练好的模型
model = load_trained_model("results/checkpoint.pt")

# 准备当前状态
current_state = {
    'fmri': fmri_features,  # [N_regions, T_current, F]
    'eeg': eeg_features
}

# 预测下一步
next_state = predict_next_state(model, current_state)
```

### 多步预测

预测未来多个时间步：

```python
# 训练时启用预测功能
# 在config/default.yaml中配置
prediction:
  enabled: true
  context_length: 50  # 使用最后50步
  steps: 10  # 预测未来10步
  weight: 0.1  # 预测损失权重

# 训练时会自动学习预测能力
# 训练完成后可用于推理
future_states = predict_multiple_steps(
    model, 
    initial_state, 
    n_steps=10
)
```

**注意**：
- 当前实现仅支持单模态内预测（fMRI预测fMRI，EEG预测EEG）
- 不支持模态间预测（如fMRI→EEG或EEG→fMRI）
- 使用自回归方式逐步预测未来状态

### 施加虚拟刺激

模拟对特定脑区的刺激：

```python
# 定义刺激
stimulation = {
    'target_regions': [10, 15, 20],  # 目标脑区ID
    'amplitude': 0.5,  # 刺激强度
    'duration': 5,  # 持续时间步
    'pattern': 'pulse'  # 刺激模式：pulse/continuous/ramp
}

# 预测响应
response = simulate_stimulation(
    model,
    initial_state,
    stimulation,
    n_steps=20
)
```

输出包括：
- 每个时间步的全脑活动状态
- 刺激区域的响应曲线
- 连接模式的变化

---

## Unity前端集成

### JSON输出格式

系统支持导出标准JSON格式供Unity可视化：

#### 脑活动状态JSON

```json
{
  "timestamp": 1234567890,
  "brain_state": {
    "regions": [
      {
        "id": 1,
        "label": "Visual_Cortex_L",
        "position": {"x": -5, "y": -85, "z": 5},
        "activity": 0.75,
        "modality": "fmri",
        "features": {
          "amplitude": 0.75,
          "frequency": 10.5,
          "phase": 1.57
        }
      },
      {
        "id": 2,
        "label": "Motor_Cortex_R",
        "position": {"x": 40, "y": -20, "z": 60},
        "activity": 0.85,
        "modality": "fmri"
      }
    ],
    "connections": [
      {
        "source": 1,
        "target": 2,
        "strength": 0.65,
        "type": "functional"
      }
    ],
    "global_stats": {
      "mean_activity": 0.68,
      "std_activity": 0.12,
      "max_activity": 0.95,
      "active_regions": 150
    }
  },
  "metadata": {
    "subject": "sub-01",
    "atlas": "Schaefer200",
    "time_point": 100,
    "model_version": "current"
  }
}
```

### 导出脑状态

```bash
# 导出单个时间点
python main.py export-brain-state \
  --model results/checkpoint.pt \
  --data test_file3/sub-01 \
  --output brain_state.json \
  --time-point 100

# 导出时间序列
python main.py export-brain-sequence \
  --model results/checkpoint.pt \
  --data test_file3/sub-01 \
  --output brain_sequence/ \
  --start 0 \
  --end 200 \
  --step 5
```

### 实时预测API

用于实时交互：

```python
from twinbrain_api import TwinBrainServer

# 启动服务器
server = TwinBrainServer(
    model_path="results/checkpoint.pt",
    host="0.0.0.0",
    port=8080
)

# Unity可以通过HTTP请求获取预测
# GET /api/brain-state?subject=sub-01&time=100
# POST /api/predict 
#   Body: {"current_state": {...}, "stimulation": {...}}
```

### Unity集成示例

Unity C#代码示例：

```csharp
using UnityEngine;
using System.Collections;
using Newtonsoft.Json;

public class BrainVisualization : MonoBehaviour
{
    private string apiUrl = "http://localhost:8080/api";
    
    IEnumerator GetBrainState()
    {
        string url = $"{apiUrl}/brain-state?subject=sub-01&time=100";
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                BrainStateData data = JsonConvert.DeserializeObject<BrainStateData>(
                    request.downloadHandler.text
                );
                
                // 更新可视化
                UpdateBrainVisualization(data);
            }
        }
    }
    
    void UpdateBrainVisualization(BrainStateData data)
    {
        foreach (var region in data.brain_state.regions)
        {
            // 根据activity设置颜色
            Color color = GetActivityColor(region.activity);
            
            // 更新对应脑区的材质
            GameObject regionObj = GetRegionObject(region.id);
            regionObj.GetComponent<Renderer>().material.color = color;
        }
    }
    
    Color GetActivityColor(float activity)
    {
        // 活跃度映射到颜色：蓝色(低) → 红色(高)
        return Color.Lerp(Color.blue, Color.red, activity);
    }
}
```

---

## 配置参数详解

### 配置文件结构

配置文件使用YAML格式，包含以下部分：

```yaml
version: "current"
description: "配置说明"

# 路径配置
paths:
  base_dir: "test_file3/"
  output_dir: "results/"
  atlas_file: "atlases/schaefer200_mask_ready.json"
  dti_file: "dti_matrix.npy"

# 训练参数
training:
  warmup_epochs: 5
  warmup_run_epochs: 10
  finetune_epochs: 80
  learning_rate: 0.0001
  batch_size: 1
  
# 模型架构
model:
  hidden_dim: 128
  num_gnn_layers: 4
  decoder_layers: 3
  dropout: 0.3
  temporal_T: 200
  spatial_T: 384
  
# 损失函数权重
loss:
  recon_weight: 1.0
  temp_weight: 5.0
  recon_norm_weight: 3.0
  align_weight: 0.1
  
# 数据处理
data:
  tasks: ['rest', 'task']
  roi_extraction_method: "mean"
  normalize: true
  cache_preprocessed: true
  
# EEG特定配置
eeg:
  source_localization: true
  n_components: 200
  freq_bands:
    delta: [0.5, 4]
    theta: [4, 8]
    alpha: [8, 13]
    beta: [13, 30]
    gamma: [30, 100]
```

### 关键参数说明

#### 训练参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| warmup_epochs | 5 | 预热阶段轮数，学习基本重建 |
| warmup_run_epochs | 10 | 预热后继续训练轮数 |
| finetune_epochs | 80 | 微调阶段轮数，优化对齐 |
| learning_rate | 0.0001 | 初始学习率 |
| lr_decay_step | 20 | 学习率衰减步长 |
| lr_decay_gamma | 0.5 | 学习率衰减系数 |

#### 模型参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| hidden_dim | 128 | 隐藏层维度 |
| num_gnn_layers | 4 | GNN层数 |
| decoder_layers | 3 | 解码器层数 |
| dropout | 0.1 | Dropout比例 |
| temporal_T | 200 | 时间投影维度 |
| spatial_T | 384 | 空间时间维度 |

**调优建议**：
- `hidden_dim`: 更大的值能捕获更复杂的模式，但增加计算量（64/128/256）
- `num_gnn_layers`: 更多层能传播更远的信息，但可能过拟合（2-6层）
- `decoder_layers`: 3层解码器提供良好的重建质量
- `dropout`: 0.1-0.3之间，防止过拟合

#### 损失函数权重

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| recon_weight | 1.0 | 重建损失权重 |
| temp_weight | 5.0 | 时间对齐权重（强化版） |
| recon_norm_weight | 3.0 | 归一化重建损失权重 |
| recon_corr_weight | 2.0 | 相关性重建权重 |

**调优建议**：
- `temp_weight`: 5.0提供强化的时间对齐
- 权重比例比绝对值更重要
- 如果重建质量差，增大`recon_weight`
- 如果时间对齐差，增大`temp_weight`

#### EEG配置

```yaml
eeg:
  # 是否进行源定位
  source_localization: true
  
  # 源定位方法
  inverse_method: "MNE"  # MNE/dSPM/sLORETA
  
  # PCA降维保留的成分数
  n_components: 200
  
  # 频带定义
  freq_bands:
    delta: [0.5, 4]    # δ波
    theta: [4, 8]      # θ波
    alpha: [8, 13]     # α波
    beta: [13, 30]     # β波
    gamma: [30, 100]   # γ波
  
  # EEG解码器配置
  decoder:
    channels: 256
    kernel_size: 5
    num_layers: 3
    dropout: 0.2
```

### 当前配置特点

```yaml
# config/default.yaml
training:
  warmup_epochs: 5
  finetune_epochs: 80
  learning_rate: 0.0001
  gradient_accumulation_steps: 1
  
model:
  hidden_dim: 128
  decoder_layers: 3
  dropout: 0.1

loss:
  recon_weight: 1.0
  temp_weight: 5.0  # 强化时间对齐
  recon_norm_weight: 3.0
  
prediction:
  enabled: false  # 可启用多步预测
  context_length: 50  # 使用历史步数
  steps: 10  # 预测未来步数
  weight: 0.1  # 预测损失权重
```

主要特点：
- 深度解码器（3层）提升重建质量
- 强化的时间对齐（temp_weight=5.0）
- 支持可选的多步预测功能
- 梯度累积支持

---

## 命令行使用

### 训练命令

```bash
# 基础训练
python main.py train --config config/default.yaml

# 指定数据目录
python main.py train \
  --config config/default.yaml \
  --base-dir /path/to/subjects

# 指定输出目录
python main.py train \
  --config config/default.yaml \
  --output-dir /path/to/results

# 调试模式
python main.py train \
  --config config/default.yaml \
  --log-level DEBUG

# 查看配置（不运行）
python main.py train \
  --config config/default.yaml \
  --dry-run

# 禁用CUDA
python main.py train \
  --config config/default.yaml \
  --no-cuda
```

### 导出命令

```bash
# 导出潜在表征
python main.py export \
  --config config/export.yaml \
  --subject sub-01

# 导出脑状态JSON
python main.py export-brain-state \
  --model results/checkpoint.pt \
  --data test_file3/sub-01 \
  --output brain_state.json

# 导出时间序列
python main.py export-sequence \
  --model results/checkpoint.pt \
  --data test_file3/sub-01 \
  --output sequence/ \
  --start 0 --end 200 --step 5
```

### 预测命令

```bash
# 单步预测
python main.py predict \
  --model results/checkpoint.pt \
  --input current_state.json \
  --output next_state.json

# 多步预测
python main.py predict-sequence \
  --model results/checkpoint.pt \
  --input initial_state.json \
  --output future_states/ \
  --n-steps 10

# 施加刺激预测
python main.py simulate \
  --model results/checkpoint.pt \
  --input initial_state.json \
  --stimulation stimulation.json \
  --output response/ \
  --n-steps 20
```

### 可视化命令

```bash
# 生成训练曲线
python main.py visualize-training \
  --log-file results/training.log \
  --output training_curves.png

# 生成脑连接图
python main.py visualize-connectivity \
  --data brain_state.json \
  --output connectivity.png \
  --threshold 0.3

# 生成活动热图
python main.py visualize-activity \
  --data brain_sequence/ \
  --output activity_heatmap.png
```

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**方案1**: 减小模型参数
```yaml
model:
  hidden_dim: 64  # 从128减到64
  num_gnn_layers: 3  # 从4减到3
```

**方案2**: 减少时间维度
```yaml
model:
  temporal_T: 100  # 从200减到100
  spatial_T: 200   # 从384减到200
```

**方案3**: 使用CPU训练
```bash
python main.py train --config config/default.yaml --no-cuda
```

### Q2: 如何提高重建质量？

1. **增加训练轮数**
```yaml
training:
  finetune_epochs: 100  # 增加到100
```

2. **增大重建损失权重**
```yaml
loss:
  recon_weight: 2.0  # 从1.0增到2.0
```

3. **使用更深的解码器**
```yaml
model:
  decoder_layers: 4  # 从3增到4
```

4. **降低dropout**
```yaml
model:
  dropout: 0.2  # 从0.3减到0.2
```

### Q3: 时间对齐效果不好？

使用默认配置，它有更强的时间对齐：
```bash
python main.py train --config config/default.yaml
```

或手动增强：
```yaml
loss:
  temp_weight: 10.0  # 进一步增大
```

### Q4: EEG数据处理失败？

检查：
1. EEG文件格式是否正确（支持.fif, .set等）
2. 是否有坏导或伪迹未去除
3. 采样率是否合理（建议>=250Hz）

配置源定位：
```yaml
eeg:
  source_localization: true
  inverse_method: "MNE"  # 尝试不同方法
```

### Q5: 如何加速训练？

1. **使用GPU**
```bash
# 确保CUDA可用
python -c "import torch; print(torch.cuda.is_available())"
```

2. **使用缓存**
```yaml
data:
  cache_preprocessed: true  # 启用预处理缓存
```

3. **减少诊断频率**
```yaml
training:
  diagnostic_interval: 10  # 每10个epoch一次，而非每次
```

### Q6: 预测结果不稳定？

1. **增加warmup**
```yaml
training:
  warmup_epochs: 10  # 从5增到10
```

2. **使用更小的学习率**
```yaml
training:
  learning_rate: 0.00005  # 从0.0001减半
```

3. **增加模型容量**
```yaml
model:
  hidden_dim: 256  # 从128增到256
```

### Q7: Unity集成时JSON解析失败？

检查JSON格式：
```bash
# 验证JSON格式
python -m json.tool brain_state.json
```

确保包含必需字段：
- `brain_state.regions`
- `brain_state.connections`
- `metadata`

### Q8: 如何启用多步预测功能？

在配置文件中启用：
```yaml
prediction:
  enabled: true  # 启用预测
  context_length: 50  # 使用历史步数
  steps: 10  # 预测未来步数
  weight: 0.1  # 损失权重
```

训练时会学习预测能力，训练完成后可用于推理。

### Q9: 内存不足导致程序崩溃？

减少数据加载量：
```yaml
data:
  max_timepoints: 200  # 限制时间点数量
  subsample_rate: 2    # 降采样
```

使用数据生成器而非一次加载：
```yaml
data:
  use_generator: true
  chunk_size: 50
```

### Q10: 如何选择合适的脑图谱？

常用选项：
- **Schaefer200**: 200个脑区，适合平衡精度和计算量
- **Schaefer400**: 400个脑区，更精细但计算量大
- **AAL116**: 116个脑区，经典图谱
- **Brainnetome246**: 246个脑区，功能导向

推荐：Schaefer200作为默认选择

---

## 附录

### A. 文件结构

```
twinbrain/
├── main.py                   # 主入口
├── config/                   # 配置文件
│   ├── default.yaml          # 默认配置
│   ├── v3_legacy.yaml        # 向后兼容配置
│   └── export.yaml           # 导出配置
├── workflows/                # 工作流
│   ├── training.py           # 训练流程
│   ├── export_latent.py      # 导出流程
│   └── inference.py          # 推理流程
├── mapper/                   # 数据映射
│   ├── atlas_mapper.py
│   ├── bids_mapper.py
│   ├── eeg_mapper.py
│   └── multi_modal_mapper.py
├── train/                    # 训练模块
│   ├── dynamic_hetero_gnn.py # 模型定义
│   ├── hetero_trainer.py     # 训练器
│   ├── coder.py              # 编解码器
│   └── aligner.py            # 对齐器
├── utils/                    # 工具
│   ├── config.py             # 配置管理
│   ├── logging_utils.py      # 日志系统
│   ├── function.py           # 数据处理
│   └── analysis.py           # 分析工具
├── atlases/                  # 脑图谱
│   ├── schaefer_2018/
│   └── aal/
└── preprocess/               # 预处理
    ├── eeg_preprocessor.py
    └── fmri_preprocessor.py
```

### B. 缩写和术语

- **fMRI**: functional Magnetic Resonance Imaging，功能性磁共振成像
- **EEG**: Electroencephalography，脑电图
- **DTI**: Diffusion Tensor Imaging，扩散张量成像
- **ROI**: Region of Interest，感兴趣区域
- **BIDS**: Brain Imaging Data Structure，脑成像数据结构
- **GNN**: Graph Neural Network，图神经网络
- **GRU**: Gated Recurrent Unit，门控循环单元
- **MNI**: Montreal Neurological Institute，蒙特利尔神经学研究所（标准空间）

### C. 参考资源

- [PyTorch文档](https://pytorch.org/docs/)
- [PyTorch Geometric文档](https://pytorch-geometric.readthedocs.io/)
- [MNE-Python文档](https://mne.tools/)
- [Nibabel文档](https://nipy.org/nibabel/)
- [BIDS规范](https://bids-specification.readthedocs.io/)

---

**文档版本**: 2.0  
**最后更新**: 2026-02-02  
**维护者**: TwinBrain Development Team
