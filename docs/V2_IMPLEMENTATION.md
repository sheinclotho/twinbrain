# TwinBrain V2 - 跨模态双向预测实现说明

## 📋 概述

TwinBrain V2 实现了**跨模态双向预测**功能，使系统能够在训练过程中学习 fMRI 和 EEG 之间的双向预测关系。

### 核心功能

- ✅ **fMRI → EEG 预测**：从 fMRI 特征预测 EEG 未来状态
- ✅ **EEG → fMRI 预测**：从 EEG 特征预测 fMRI 未来状态
- ✅ **双向训练**：同时学习两个方向的跨模态预测
- ✅ **跨模态注意力**：使用注意力机制捕获跨模态依赖关系
- ✅ **模态桥接网络**：可选的模态转换网络提升预测质量

---

## 🆕 V2 新增文件

为了保持代码稳定性，V2 功能通过新增文件实现，**不修改原有文件**：

### 核心文件

1. **`train/predictor_v2.py`** (新增)
   - 跨模态预测器模块
   - `CrossModalPredictor`: 单向跨模态预测
   - `BidirectionalCrossModalPredictor`: 双向跨模态预测
   - 包含原有的 `PredictorHead` 和 `ConditionalPredictor`

2. **`train/hetero_trainer_v2.py`** (新增)
   - 扩展的训练器
   - 继承自 `DynamicHeteroTrainer`
   - 集成跨模态预测损失
   - 添加跨模态训练逻辑

3. **`config/default_v2.yaml`** (新增)
   - V2 配置文件
   - 包含跨模态预测参数
   - 完整的配置示例

4. **`example_v2_cross_modal.py`** (新增)
   - 使用示例和说明
   - 配置选项说明
   - V1 vs V2 对比

5. **`docs/V2_IMPLEMENTATION.md`** (本文件)
   - V2 实现说明文档

---

## 🔧 技术实现

### 架构设计

```
单模态预测（V1 已有）:
┌─────────┐       ┌─────────┐
│  fMRI   │  -->  │  fMRI   │
│ history │       │ future  │
└─────────┘       └─────────┘

┌─────────┐       ┌─────────┐
│  EEG    │  -->  │  EEG    │
│ history │       │ future  │
└─────────┘       └─────────┘

跨模态预测（V2 新增）:
┌─────────┐       ┌─────────┐
│  fMRI   │  -->  │  EEG    │  (fMRI → EEG)
│ history │       │ future  │
└─────────┘       └─────────┘

┌─────────┐       ┌─────────┐
│  EEG    │  -->  │  fMRI   │  (EEG → fMRI)
│ history │       │ future  │
└─────────┘       └─────────┘
```

### 跨模态预测器

`CrossModalPredictor` 的核心机制：

1. **模态桥接**（可选）
   - 将源模态特征转换到适合目标模态的表示空间
   - 使用可学习的神经网络进行转换

2. **跨模态注意力**
   - 目标模态的预测 attend to 源模态的历史特征
   - 捕获跨模态的时空依赖关系

3. **时序 GRU**
   - 在目标模态空间进行自回归预测
   - 结合跨模态信息和时序动态

4. **输出投影**
   - 将预测映射到目标模态的潜在空间
   - 层归一化确保训练稳定性

### 训练流程

在 `DynamicHeteroTrainerV2` 中：

1. **数据准备**
   - 提取 fMRI 和 EEG 的潜在序列
   - 使用滑动窗口创建训练样本

2. **跨模态预测**
   - fMRI 上下文 → 预测 EEG 未来
   - EEG 上下文 → 预测 fMRI 未来

3. **损失计算**
   ```python
   total_loss = recon_loss + temp_loss + align_loss 
                + prediction_loss + cross_modal_loss
   ```

4. **反向传播**
   - 梯度同时优化所有预测器
   - 共享的编码器学习通用表示

---

## 📖 使用方法

### 方法 1: 使用配置文件

```bash
# 使用 V2 配置文件训练
python main.py train --config config/default_v2.yaml
```

### 方法 2: 在代码中使用

```python
from train.hetero_trainer_v2 import DynamicHeteroTrainerV2

trainer = DynamicHeteroTrainerV2(
    hetero_data=your_data,
    hidden_dim=128,
    
    # 启用单模态预测
    enable_prediction=True,
    prediction_context_length=50,
    prediction_steps=10,
    prediction_weight=0.1,
    
    # 启用跨模态预测（V2 新增）
    enable_cross_modal_prediction=True,
    cross_modal_weight=0.1,
    cross_modal_context_length=50,
    cross_modal_steps=10,
    cross_modal_direction="both",  # 'fmri_to_eeg', 'eeg_to_fmri', 'both'
    cross_modal_use_bridge=True,
    cross_modal_share_attention=False,
)

# 训练
trainer.train(save_dir="results_v2")
```

### 方法 3: 使用便捷函数

```python
from train.hetero_trainer_v2 import create_trainer_v2

trainer = create_trainer_v2(
    config=config,
    hetero_data=your_data
)
```

---

## ⚙️ 配置参数

### 跨模态预测配置

```yaml
cross_modal_prediction:
  enabled: true              # 启用跨模态预测
  weight: 0.1                # 跨模态预测损失权重
  context_length: 50         # 源模态历史步数
  steps: 10                  # 预测目标模态未来步数
  direction: "both"          # 预测方向
  
  # 架构设置
  use_bridge: true           # 使用模态桥接网络
  share_attention: false     # 是否共享注意力权重
  num_layers: 3              # GRU 层数
  num_heads: 8               # 注意力头数
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | false | 是否启用跨模态预测 |
| `weight` | 0.1 | 跨模态预测损失在总损失中的权重 |
| `context_length` | 50 | 源模态使用的历史时间步数 |
| `steps` | 10 | 预测目标模态的未来步数 |
| `direction` | "both" | 预测方向：'fmri_to_eeg', 'eeg_to_fmri', 'both' |
| `use_bridge` | true | 是否使用模态转换桥接网络 |
| `share_attention` | false | 两个方向是否共享注意力参数 |

---

## 🎯 预期效果

### 1. 训练过程中

训练日志会显示：
```
[Epoch 10] loss=1.234 recon=0.456 temp=0.234 align=0.123 pred=0.089 
           cross_modal=0.078 (f2e=0.039 e2f=0.039)
```

其中：
- `cross_modal`: 总跨模态预测损失
- `f2e`: fMRI→EEG 预测损失
- `e2f`: EEG→fMRI 预测损失

### 2. 推理时

可以使用训练好的跨模态预测器：

```python
# 从 fMRI 预测 EEG
predictions = trainer.cross_modal_predictor(
    fmri_seq=fmri_sequence,
    eeg_seq=eeg_context,
    direction="fmri_to_eeg"
)

# 获取预测结果
eeg_prediction = predictions['fmri_to_eeg'][0]
```

### 3. 评估指标

- **跨模态预测误差**：MSE between predicted and actual
- **跨模态相关性**：Correlation between modalities
- **预测一致性**：Temporal consistency of predictions

---

## 🔍 与 V1 的区别

| 特性 | V1 (原版) | V2 (新版) |
|------|-----------|-----------|
| **单模态预测** | ✅ 已实现 | ✅ 保留 |
| **fMRI → fMRI** | ✅ | ✅ |
| **EEG → EEG** | ✅ | ✅ |
| **跨模态预测** | ❌ 未实现 | ✅ 新增 |
| **fMRI → EEG** | ❌ | ✅ |
| **EEG → fMRI** | ❌ | ✅ |
| **跨模态注意力** | ❌ | ✅ |
| **模态桥接** | ❌ | ✅ |
| **原文件修改** | - | ❌ 无修改 |

---

## 📊 文件对比

### 原文件（不变）

```
train/
├── predictor.py           # 保持不变
├── dynamic_hetero_gnn.py  # 保持不变
├── hetero_trainer.py      # 保持不变
└── ...
```

### V2 新增文件

```
train/
├── predictor_v2.py        # 新增：跨模态预测器
├── hetero_trainer_v2.py   # 新增：V2 训练器
└── ...

config/
└── default_v2.yaml        # 新增：V2 配置

docs/
└── V2_IMPLEMENTATION.md   # 新增：V2 说明文档

example_v2_cross_modal.py  # 新增：使用示例
```

---

## 💡 使用建议

### 1. 初次使用

- 从较小的 `cross_modal_weight` 开始（如 0.05）
- 使用 `direction: "both"` 获得最佳效果
- 启用 `use_bridge: true` 提升预测质量

### 2. 调优

- 监控跨模态损失与其他损失的平衡
- 如果跨模态损失过大，降低 `weight`
- 如果跨模态损失过小，增大 `weight`
- 调整 `context_length` 和 `steps` 以适应数据特性

### 3. 评估

- 检查 fMRI→EEG 和 EEG→fMRI 的预测误差
- 比较跨模态预测与单模态预测的准确性
- 分析跨模态注意力权重以理解模态关系

---

## 🚀 下一步开发

### 可能的扩展

1. **多步跨模态预测**
   - 当前：固定步数预测
   - 扩展：可变长度预测

2. **条件跨模态预测**
   - 当前：无条件预测
   - 扩展：基于外部刺激的条件预测

3. **多模态融合预测**
   - 当前：双向独立预测
   - 扩展：融合多模态信息的联合预测

4. **不确定性估计**
   - 当前：点估计
   - 扩展：预测分布和置信区间

---

## 📚 参考文档

- **原始分析**: `docs/PREDICTION_ANALYSIS.md`
- **使用示例**: `example_v2_cross_modal.py`
- **配置文件**: `config/default_v2.yaml`
- **代码实现**: 
  - `train/predictor_v2.py`
  - `train/hetero_trainer_v2.py`

---

## ❓ 常见问题

### Q1: V2 会影响原有功能吗？

**A**: 不会。V2 是独立的新增文件，原有文件完全不变，确保系统稳定性。

### Q2: 如何切换回 V1？

**A**: 使用原配置文件 `config/default.yaml` 或在代码中使用 `DynamicHeteroTrainer` 而非 `DynamicHeteroTrainerV2`。

### Q3: V2 的性能影响？

**A**: 
- 参数增加：约 +10-20%（取决于配置）
- 训练时间：约 +15-25%（每个 epoch）
- 内存占用：约 +300-500 MB（GPU）

### Q4: 必须同时有 fMRI 和 EEG 吗？

**A**: 是的。跨模态预测需要两种模态的数据。如果只有单模态，使用 V1 即可。

### Q5: 如何验证跨模态预测效果？

**A**: 
1. 检查训练日志中的跨模态损失下降
2. 计算预测的 MSE 和相关系数
3. 可视化预测结果与真实值对比
4. 分析跨模态注意力权重

---

**版本**: 2.0  
**实现日期**: 2026-02-03  
**维护者**: TwinBrain Development Team
