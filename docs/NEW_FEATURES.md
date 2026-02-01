# TwinBrain 新功能说明

## 🎯 已实现的优化功能

本文档介绍 TwinBrain 系统最新实现的优化功能（2026-02-01）。

---

## 1. 增强训练监控系统 📊

### 功能概述

全新的训练指标追踪系统，提供完整的训练可见性和自动化监控。

### 核心组件

#### MetricsTracker

位置：`utils/metrics_tracker.py`

**功能**：
- 自动记录所有训练指标历史
- 保存损失分量（重构、时序、对齐等）
- 记录梯度统计信息
- 导出 JSON 格式的指标历史
- 自动生成训练摘要报告

**配置**：
```yaml
# config/default.yaml
metrics:
  enabled: true  # 启用指标追踪
  output_dir: "metrics"  # 保存目录
```

**输出示例**：

训练过程中自动记录：
```
[Epoch  10] total=1.2345 align=0.3456 temp=0.4567 recon=0.4322
[Epoch  10] relative_error={'fmri': 0.12, 'eeg': 0.15}
```

训练结束后的摘要：
```
================================================================================
Metrics Summary (Last 10 epochs)
================================================================================
loss/total                     | Latest:   1.2345 | Avg:   1.3456 | Std:   0.1234
loss/reconstruction            | Latest:   0.4322 | Avg:   0.4500 | Std:   0.0234
loss/temporal                  | Latest:   0.4567 | Avg:   0.4600 | Std:   0.0123
loss/alignment                 | Latest:   0.3456 | Avg:   0.3400 | Std:   0.0210
rel_error/fmri                 | Latest:   0.1200 | Avg:   0.1250 | Std:   0.0050
rel_error/eeg                  | Latest:   0.1500 | Avg:   0.1550 | Std:   0.0060
================================================================================
```

**JSON 导出**（`results/metrics/metrics_history.json`）：
```json
{
  "loss/total": [
    {"epoch": 1, "value": 2.5},
    {"epoch": 2, "value": 2.3}
  ],
  "loss/reconstruction": [...],
  "rel_error/fmri": [...]
}
```

#### TrainingMonitor

**功能**：
- 监控训练进度
- 检测训练停滞
- 检测异常值（NaN/Inf）
- 自动生成警告

### 使用方法

指标追踪已自动集成到训练流程中，只需在配置文件中启用：

```bash
# 在 config/default.yaml 中设置 metrics.enabled = true
python main.py train --config config/default.yaml
```

训练结束后，查看：
- 控制台输出：训练摘要
- 文件：`results/metrics/metrics_history.json`

---

## 2. 多步未来预测 🔮

### 功能概述

基于 GRU 和注意力机制的未来状态预测模块，可以预测大脑未来多个时间步的状态。

### 核心组件

#### PredictorHead

位置：`train/predictor.py`

**特性**：
- GRU 时序建模
- 多头注意力机制
- 自回归预测
- 可配置预测步数

**架构**：
```python
class PredictorHead(nn.Module):
    """多步未来状态预测器"""
    def __init__(
        self,
        hidden_dim: int,
        n_future_steps: int = 10,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    )
```

**配置**：
```yaml
# config/default.yaml
prediction:
  enabled: true   # 启用预测
  steps: 10       # 预测未来10步
  weight: 0.1     # 预测损失权重
```

#### ConditionalPredictor

**用途**：
- TMS/tACS 刺激响应预测
- 药物效应模拟
- 神经调控效果评估

**特性**：
- 整合外部刺激/条件
- 预测大脑对刺激的响应
- 支持时变刺激模式

### 使用方法

#### 1. 训练时启用预测

```yaml
# config/default.yaml
prediction:
  enabled: true
  steps: 10
  weight: 0.1
```

```bash
python main.py train --config config/default.yaml
```

#### 2. 推理时使用预测

```python
from train.predictor import PredictorHead

# 创建预测器
predictor = PredictorHead(
    hidden_dim=128,
    n_future_steps=10
)

# 历史潜在序列 [batch, seq_len, hidden_dim]
latent_seq = model.encode(brain_data)

# 预测未来
predictions, attention = predictor(
    latent_seq,
    return_attention=True
)
# predictions: [batch, 10, hidden_dim]
```

#### 3. 刺激响应预测

```python
from train.predictor import ConditionalPredictor

# 创建条件预测器
predictor = ConditionalPredictor(
    hidden_dim=128,
    condition_dim=200,  # 脑区数量
    n_future_steps=10
)

# 当前大脑状态
current_state = model.encode(current_brain_data)  # [batch, hidden_dim]

# 刺激模式（TMS 刺激特定脑区）
stimulation = torch.zeros(batch, 10, 200)
stimulation[:, :, [10, 15]] = 0.5  # 刺激脑区10和15

# 预测响应
predicted_response = predictor(current_state, stimulation)
# predicted_response: [batch, 10, hidden_dim]
```

### 预测效果

预测模块可以：
- 预测未来 1-20 步的大脑状态
- 学习时序动力学规律
- 为虚拟刺激实验提供基础
- 支持闭环神经调控

---

## 3. 示例代码

### 完整示例

参见 `example_new_features.py`，包含：
1. 多步预测演示
2. 条件预测演示（刺激响应）
3. 指标追踪演示
4. 训练监控演示

运行示例：
```bash
python example_new_features.py
```

### 快速开始

1. **启用新功能**：
```yaml
# config/default.yaml
prediction:
  enabled: true
  steps: 10
  weight: 0.1

metrics:
  enabled: true
  output_dir: "metrics"
```

2. **运行训练**：
```bash
python main.py train --config config/default.yaml
```

3. **查看结果**：
- 控制台：实时训练日志和摘要
- `results/metrics/metrics_history.json`：完整指标历史
- `results/hetero_gnn_trained.pt`：训练好的模型（含预测器）

---

## 4. 配置参数详解

### prediction 配置

```yaml
prediction:
  enabled: false      # 是否启用预测功能
  steps: 10           # 预测未来的步数（1-50）
  weight: 0.1         # 预测损失在总损失中的权重（0.0-1.0）
```

**建议值**：
- `steps`: 5-15 步（取决于数据时间分辨率）
- `weight`: 0.05-0.2（避免过度影响重建质量）

### metrics 配置

```yaml
metrics:
  enabled: true           # 是否启用指标追踪
  output_dir: "metrics"   # 指标保存的子目录
```

---

## 5. 性能影响

### 预测模块

- **参数增加**：约 +1-2M 参数（取决于 hidden_dim）
- **训练时间**：约 +5-10%（每个 epoch）
- **内存占用**：约 +200-500 MB（GPU）

### 指标追踪

- **性能影响**：< 1%（几乎可忽略）
- **存储占用**：约 1-5 MB（JSON 文件）

---

## 6. 故障排查

### 预测功能无法启用

检查：
1. 配置文件中 `prediction.enabled = true`
2. `train/predictor.py` 文件存在
3. PyTorch 版本 >= 1.10

### 指标未保存

检查：
1. 配置文件中 `metrics.enabled = true`
2. `utils/metrics_tracker.py` 文件存在
3. 输出目录有写入权限

### 训练报错

如果遇到预测相关错误，可以临时禁用：
```yaml
prediction:
  enabled: false
```

---

## 7. 未来计划

### 短期（1-2个月）
- [ ] 优化预测损失权重调度
- [ ] 添加多模态联合预测
- [ ] 实现预测不确定性估计

### 中期（3-6个月）
- [ ] 物理约束的预测（神经动力学）
- [ ] 因果推断集成
- [ ] 实时预测 API

---

## 8. 参考资料

### 相关文档
- `OPTIMIZATION_DIRECTIONS.md`：完整的优化方向说明
- `docs/TwinBrain系统使用指南.md`：系统使用手册
- `CHANGELOG.md`：更新历史

### 相关代码
- `train/predictor.py`：预测模块实现
- `utils/metrics_tracker.py`：指标追踪实现
- `train/hetero_trainer.py`：训练器集成
- `example_new_features.py`：功能演示

---

**版本**: 1.0  
**日期**: 2026-02-01  
**作者**: TwinBrain Development Team
