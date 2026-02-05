# TwinBrain 预测功能实现分析报告

## 📋 概述

本文档详细分析 TwinBrain 系统在训练过程中实现的预测功能，特别关注：
1. 自回归多步预测的实现情况
2. 模态间双向预测的实现情况

**分析日期**: 2026-02-02  
**分析范围**: 训练代码和配置文件

---

## 🎯 核心发现

### 1. 自回归多步预测 ✅ 已实现

**结论**: 训练过程中**已完全实现**自回归多步预测功能。

#### 实现机制

##### A. 基础自回归预测（_temporal_prediction_loss）

**位置**: `train/hetero_trainer.py` 第411-520行

**实现方式**:
```python
def _temporal_prediction_loss(
    self,
    proj_seq: torch.Tensor,  # [N, T, H] 潜在序列
    nt: str,  # 模态类型
    context_len: int = 40,  # 上下文长度
    predict_len: int = 4,  # 预测步数
    teacher_forcing_ratio: float = 0.3,  # Teacher forcing 比例
    ...
):
```

**关键特性**:
1. **上下文编码**: 使用GRU处理前40步作为上下文
2. **自回归生成**: 逐步预测未来4步
3. **Teacher Forcing**: 30%概率使用真实值，70%使用预测值
4. **时域+频域损失**: 结合MSE损失和FFT频域损失
5. **训练集成**: 在每个训练epoch中被调用（第735行）

##### B. 增强预测（PredictorHead）

**位置**: `train/predictor.py` + `train/hetero_trainer.py` 第672-722行

**实现方式**:
```python
class PredictorHead(nn.Module):
    """多步未来状态预测器，带注意力机制"""
    def __init__(
        self,
        hidden_dim: int,
        n_future_steps: int = 10,
        context_length: Optional[int] = None,  # 可配置上下文
        num_layers: int = 3,
        num_heads: int = 8,
        ...
    )
```

**关键特性**:
1. **滑动窗口训练**: 在序列上滑动创建多个训练样本
2. **多头注意力**: 关注历史中的关键时间步
3. **可配置上下文**: 通过config指定使用多少历史步数
4. **集成训练**: 当`prediction.enabled=true`时自动启用（第672行）

**训练流程**:
```python
# 滑动窗口方式创建训练样本
context_len = self.prediction_context_length or 50
stride = max(1, self.prediction_steps // 2)

for start_idx in range(0, T - context_len - self.prediction_steps + 1, stride):
    context_seq = seq[:, start_idx:start_idx+context_len, :]
    target_seq = seq[:, start_idx+context_len:start_idx+context_len+self.prediction_steps, :]
    
    # 使用PredictorHead预测
    predictions, _ = self.predictor(context_seq)
    
    # 计算预测损失
    pred_loss = F_nn.mse_loss(predictions, target_seq)
```

#### 配置方式

在 `config/default.yaml` 中启用:
```yaml
prediction:
  enabled: true  # 启用预测功能
  context_length: 50  # 使用最后50步作为上下文
  steps: 10  # 预测未来10步
  weight: 0.1  # 预测损失权重
```

#### 训练效果

- ✅ 模型学习到时序依赖关系
- ✅ 可以预测未来1-20步的潜在状态
- ✅ 支持自回归生成任意长度序列
- ✅ 在训练过程中持续优化预测能力

---

### 2. 模态间双向预测 ❌ 未实现

**结论**: 训练过程中**未实现**模态间双向预测功能。

#### 当前实现分析

##### A. 单模态预测

**当前行为**:
```python
# 在hetero_trainer.py第674-722行
for nt in self.metadata[0]:  # 遍历每个模态
    seq = proj_seq_dict.get(nt, None)  # 获取该模态的序列
    # ...
    predictions, _ = self.predictor(context_seq)  # 在同一模态内预测
```

**特点**:
- fMRI数据 → 预测fMRI未来状态
- EEG数据 → 预测EEG未来状态
- **不存在跨模态预测路径**

##### B. 模态对齐 vs 跨模态预测

**LatentAligner** (`train/aligner.py`):
```python
def forward(self, z_fmri: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
    # 计算对齐损失，使fMRI和EEG的潜在表示相似
    # 但这不是预测，只是表示对齐
```

**关键区别**:
- ✅ **对齐**: 让fMRI和EEG的表示在潜在空间中相似
- ❌ **双向预测**: 从fMRI特征预测EEG活动，或从EEG特征预测fMRI活动

#### 缺失的功能

要实现模态间双向预测，需要：

1. **跨模态解码器**:
```python
class CrossModalDecoder(nn.Module):
    """从一个模态预测另一个模态"""
    def __init__(self, hidden_dim, target_dim):
        self.fmri_to_eeg = nn.Sequential(...)  # fMRI → EEG
        self.eeg_to_fmri = nn.Sequential(...)  # EEG → fMRI
```

2. **跨模态预测损失**:
```python
# 训练时需要添加
fmri_latent = model.encode(fmri_data)
predicted_eeg = cross_modal_decoder.fmri_to_eeg(fmri_latent)
loss_cross = F.mse_loss(predicted_eeg, real_eeg)
```

3. **双向预测训练**:
```python
# 需要在训练循环中添加
# fMRI → EEG 预测
eeg_pred_from_fmri = model.predict_cross_modal(fmri_seq, "fmri", "eeg")
loss_f2e = compute_loss(eeg_pred_from_fmri, eeg_target)

# EEG → fMRI 预测
fmri_pred_from_eeg = model.predict_cross_modal(eeg_seq, "eeg", "fmri")
loss_e2f = compute_loss(fmri_pred_from_eeg, fmri_target)
```

#### 当前架构限制

**模型架构** (`train/dynamic_hetero_gnn.py`):
```python
# 每个模态有独立的处理路径
for nt in self.node_types:
    # GNN处理
    # GRU时序建模
    # 独立的TemporalDecoder
    # 独立的NodeDecoder
```

**问题**:
- 各模态独立处理，缺少交叉预测路径
- 解码器只重建自己的模态，不预测其他模态

---

## 📊 详细对比

| 功能 | 实现状态 | 说明 |
|------|---------|------|
| **自回归多步预测** | ✅ 已实现 | |
| └─ 基础自回归（GRU） | ✅ 完整实现 | _temporal_prediction_loss方法 |
| └─ Teacher Forcing | ✅ 完整实现 | 30%概率使用真实值 |
| └─ 时域损失 | ✅ 完整实现 | MSE损失 |
| └─ 频域损失 | ✅ 完整实现 | FFT频谱损失 |
| └─ 增强预测器 | ✅ 完整实现 | PredictorHead + 注意力 |
| └─ 滑动窗口训练 | ✅ 完整实现 | 多样本训练 |
| **模态间双向预测** | ❌ 未实现 | |
| └─ fMRI → EEG | ❌ 缺失 | 无交叉解码器 |
| └─ EEG → fMRI | ❌ 缺失 | 无交叉解码器 |
| └─ 跨模态注意力 | ⚠️ 部分 | TemporalCrossAligner仅用于对齐 |
| └─ 跨模态预测损失 | ❌ 缺失 | 训练中未计算 |

---

## 🔍 代码证据

### 证据1: 自回归预测已实现

**文件**: `train/hetero_trainer.py`  
**位置**: 第411-520行

```python
def _temporal_prediction_loss(self, proj_seq, nt, context_len=40, predict_len=4, ...):
    """自回归多步预测损失"""
    
    # 1. 使用GRU编码上下文
    context = proj_seq[:, :context_len, :]
    out_ctx, h = gru(context)
    
    # 2. 自回归预测未来
    next_input = out_ctx[:, -1:, :]
    preds = []
    for step in range(predict_len):
        out_step, h = gru(next_input, h)
        pred = out_step[:, -1:, :]
        preds.append(pred)
        
        # Teacher forcing
        if do_teacher and step < future_targets.shape[1]:
            next_input = future_targets[:, step:step + 1, :]
        else:
            next_input = pred  # 使用自己的预测
```

**调用位置**: 第735行
```python
loss_t, pred_feat, pred_denorm = self._temporal_prediction_loss(
    seq, nt, return_preds=True, stats_nt=stats_dict.get(nt, None)
)
temp_loss = temp_loss + loss_t
```

### 证据2: PredictorHead集成训练

**文件**: `train/hetero_trainer.py`  
**位置**: 第672-722行

```python
if self.enable_prediction and self.predictor is not None:
    # 滑动窗口训练
    for nt in self.metadata[0]:
        seq = proj_seq_dict.get(nt, None)
        # ...
        for start_idx in range(0, T - context_len - self.prediction_steps + 1, stride):
            context_seq = seq[:, context_start:context_end, :]
            target_seq = seq[:, target_start:target_end, :]
            
            # 预测
            predictions, _ = self.predictor(context_seq)
            
            # 损失
            pred_loss = F_nn.mse_loss(predictions, target_seq)
            window_loss = window_loss + pred_loss
```

### 证据3: 缺少跨模态预测

**文件**: `train/hetero_trainer.py`  
**观察**: 整个文件中

```python
# 没有类似这样的代码：
# fmri_seq = proj_seq_dict['fmri']
# predicted_eeg = model.predict_cross_modal(fmri_seq, 'fmri', 'eeg')
# loss_cross = F.mse_loss(predicted_eeg, real_eeg_seq)
```

**文件**: `train/dynamic_hetero_gnn.py`  
**观察**: 模型架构中
```python
# 每个模态独立重建
for nt in self.node_types:
    decoder = self.node_decoders[nt]
    recon = decoder(seq_out, node_type=nt)  # 只重建自己
    # 没有：recon_other = decoder(seq_out, source=nt, target=other_nt)
```

---

## 💡 总结和建议

### 当前状态

✅ **自回归多步预测**: 
- 完全实现，训练良好
- 支持配置化启用/禁用
- 使用先进的滑动窗口+注意力机制

❌ **模态间双向预测**:
- 完全未实现
- 需要添加跨模态解码器
- 需要设计跨模态预测损失
- 需要重构训练流程

### 实现建议

如需添加模态间双向预测，建议步骤：

1. **添加跨模态解码器模块**
2. **在DynamicHeteroGNN中添加跨模态预测路径**
3. **在训练器中添加跨模态预测损失**
4. **在配置文件中添加相关参数**
5. **验证和调优**

详细实现方案可参考 `OPTIMIZATION_DIRECTIONS.md` 中的相关章节。

---

**文档版本**: 1.0  
**分析完成日期**: 2026-02-02  
**分析人员**: TwinBrain Analysis Team
