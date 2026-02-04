# Dynamic Weighting Quick Reference

## 快速启用 (Quick Enable)

```yaml
# config/default.yaml
dynamic_weighting:
  enabled: true  # 改为 true 即可启用
```

## 工作原理 (How It Works)

```
Data → Compute Variability → Map to Weights → Apply to Loss
数据 → 计算变化度      → 映射为权重   → 应用到损失
```

### 1. 计算变化度 (Compute Variability)

**EEG**: 时间方差 + 一阶差分 → 识别活跃通道  
**fMRI**: FC变化 + 低频功率 → 识别状态转换ROI

### 2. 映射权重 (Map to Weights)

`w = softmax(variability / temperature)`

- 低温度 → 尖锐分布 → 聚焦高变化通道
- 高温度 → 平坦分布 → 包含更多通道

### 3. 训练阶段 (Training Stages)

| 阶段 | Epochs | 温度 τ | 目的 |
|------|--------|--------|------|
| Warmup | 5 | 0.1 | 防止塌缩 |
| Main | 60 | 0.1→1.0 | 学习结构 |
| Finetune | 30 | 2.0 | 防止过拟合 |

## 参数调整 (Parameter Tuning)

### 问题: EEG仍然塌缩
```yaml
warmup_temp: 0.05        # 更尖锐
warmup_epochs: 10        # 更长warmup
eeg_use_first_order_diff: true
```

### 问题: 权重过于尖锐
```yaml
warmup_temp: 0.2         # 更平坦
main_temp_start: 0.2
```

### 问题: 权重过于平坦
```yaml
main_temp_end: 0.5       # 更尖锐
finetune_temp: 1.0
```

## 监控训练 (Monitor Training)

查看日志中的:
```
[Epoch 1] Dynamic Weighting: stage=warmup, temperature=0.100
  eeg: weight_range=[0.01, 0.23], mean=0.016, std=0.029
```

**正常情况**:
- weight_range 应该有较大差异（不是全部接近）
- warmup阶段 max weight 应该较大 (>0.1)
- finetune阶段分布应该变平坦

## 消融研究 (Ablation Study)

```yaml
# 完全禁用
dynamic_weighting:
  enabled: false

# 只在warmup启用
dynamic_weighting:
  enabled: true
  disable_in_finetune: true
  main_temp_end: 10.0  # 接近uniform
```

## 关键指标 (Key Metrics)

### EEG塌缩检测
```python
# 检查 EEG 输出是否接近零
if eeg_output.abs().mean() < 0.01:
    print("⚠️ EEG collapse detected!")
```

### 权重有效性
```python
# 检查权重分布
weight_std = weights.std()
if weight_std < 0.001:
    print("⚠️ Weights too uniform!")
if weight_std > 0.05:
    print("✓ Good weight diversity")
```

## 常见配置模式 (Common Patterns)

### 保守模式 (Conservative)
```yaml
warmup_temp: 0.2
main_temp_end: 1.0
finetune_temp: 2.0
```

### 激进模式 (Aggressive)
```yaml
warmup_temp: 0.05
main_temp_end: 0.5
finetune_temp: 1.0
```

### 平衡模式 (Balanced, 默认)
```yaml
warmup_temp: 0.1
main_temp_end: 1.0
finetune_temp: 2.0
```

## 调试命令 (Debug Commands)

```python
# 查看变化度分数
variability = computer.compute_eeg_variability(data)
print(f"Variability: {variability}")

# 查看权重
weights = system.compute_modality_weights(data, 'eeg', epoch=1)
print(f"Weights: min={weights.min():.4f}, max={weights.max():.4f}")

# 检查训练阶段
stage_info = system.get_stage_info(epoch)
print(f"Stage: {stage_info['stage']}, Temp: {stage_info['temperature']}")
```

## 性能优化 (Performance)

```yaml
# 如果内存紧张
eeg_window_size: 30      # 减小窗口
eeg_use_covariance: false  # 禁用协方差

# 如果计算慢
fmri_use_fc_change: false  # 禁用FC计算
```

## 故障排除速查 (Quick Troubleshooting)

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| EEG输出接近零 | 温度过高 | 降低 warmup_temp |
| 只有少数通道有梯度 | 温度过低 | 提高 main_temp_start |
| 所有通道权重相同 | 变化度计算失败 | 检查数据质量 |
| 训练不稳定 | 权重变化太快 | 增加各阶段epochs |

## 文档链接 (Documentation)

- 完整文档: `docs/DYNAMIC_WEIGHTING.md`
- 测试: `python test_dynamic_weighting.py`
- 示例: `python example_dynamic_weighting.py`

## 联系支持 (Support)

遇到问题？
1. 查看日志中的 Dynamic Weighting 输出
2. 运行 `test_dynamic_weighting.py` 验证安装
3. 查阅 `docs/DYNAMIC_WEIGHTING.md` 详细说明

---
**快速参考版本**: v1.0  
**更新日期**: 2026-02-04
