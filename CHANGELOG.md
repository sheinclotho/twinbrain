# TwinBrain 更新历史

## 2026-02-04 (训练稳定性修复和文档清理)

### 相对误差计算修复 🔧

#### 问题描述
- 训练过程中出现警告: "[Train] relative error computation failed for this epoch"
- 错误频繁出现但没有详细的错误信息
- 影响训练指标的完整性

#### 根本原因
- `recon_final_map` 在每个 batch 结束时被删除
- 但在 epoch 结束后需要使用它计算相对误差
- 导致 `NameError` 或 `UnboundLocalError`
- 异常处理不完善，缺少详细的错误信息

#### 解决方案
1. **保留 recon_final_map**: 在 batch 清理时不删除 recon_final_map，保留用于 epoch 级别指标计算
2. **改进错误日志**: 添加详细的异常信息和堆栈跟踪
3. **内存清理**: 在使用后通过 finally 块清理 recon_final_map

### 文档清理 📚

#### 清理内容
- 移除过时的修复总结文档:
  - `CUDA_OOM_FIX_SUMMARY.md`
  - `PREDICTION_FIX_SUMMARY.md`
  - `V2实现总结.md`
  - `完成报告.md`
- 保留核心文档:
  - `README.md` / `README_CN.md`
  - `CHANGELOG.md`
  - `OPTIMIZATION_DIRECTIONS.md`

---

## 2026-02-04 (训练稳定性修复)

### 关键修复：训练静滞问题 🔧

#### 问题描述
- 训练在 "Starting stage: Warmup Stage" 后静滞不动
- 无错误信息，无计算活动（风扇无声）
- 用户需手动终止进程
- 缺乏自检能力和进度反馈

#### 根本原因
1. 缺少超时机制
2. CUDA操作无同步检查
3. 关键操作缺乏错误处理
4. 缺少进度指示器
5. 数据加载或模型初始化的静默失败

#### 解决方案

**1. 训练前验证** (train/hetero_trainer.py, lines 648-690)
```python
# 检查 data_list 非空
# 验证模型和 GraphEncoder 已初始化
# 检查 CUDA 可用性和内存
# 检查第一个 batch 的数据结构
```

**2. 心跳监控系统** (TrainingHeartbeat class)
- 后台线程每30秒记录心跳
- 60秒无活动时发出警告
- 自动检测并报告潜在静滞
- 训练完成或出错时安全清理

**3. 批次级进度日志**
- 第一个 epoch 记录每个 batch
- 后续 epoch 每10个 batch 记录一次
- 显示当前进度 (batch X/Y)
- 每5个 batch 更新心跳

**4. 关键操作错误处理**
```python
# GraphEncoder forward: try-catch + CUDA sync
# Model forward: try-catch + CUDA sync  
# Alignment loss: try-catch
# Prediction loss: try-catch (优雅降级)
# Temporal prediction: try-catch
# Backward pass: try-catch + CUDA sync
```

**5. CUDA 同步检查**
- data.to(device) 后同步
- GraphEncoder forward 后同步
- Model forward 后同步
- Backward pass 后同步
- 确保 GPU 操作完成后再继续

**6. 数据结构检查**
- 训练前检查第一个 batch
- 记录节点类型、边类型
- 记录张量形状和数据类型
- 在计算开始前识别数据问题

#### 使用效果
```
2026-02-04 17:20:04 | workflows.training | INFO | Validating training setup...
2026-02-04 17:20:04 | workflows.training | INFO |   ✓ Data list contains 3 batches
2026-02-04 17:20:04 | workflows.training | INFO |   ✓ Model and GraphEncoder initialized
2026-02-04 17:20:04 | workflows.training | INFO |   ✓ Using CUDA device: NVIDIA GeForce RTX 3080
2026-02-04 17:20:04 | workflows.training | INFO |   ✓ First batch inspection:
2026-02-04 17:20:04 | workflows.training | INFO |     - fmri.x_seq shape: (200, 384, 200)
2026-02-04 17:20:04 | workflows.training | INFO |     - eeg.x_seq shape: (62, 500, 62)
2026-02-04 17:20:04 | workflows.training | INFO | ✓ Heartbeat monitor started
2026-02-04 17:20:04 | workflows.training | INFO | [Epoch 1/5] Processing batch 1/3...
2026-02-04 17:20:04 | workflows.training | INFO | [Epoch 1] Running GraphEncoder...
2026-02-04 17:20:05 | workflows.training | INFO | [Epoch 1] GraphEncoder completed
2026-02-04 17:20:34 | workflows.training | INFO | [Heartbeat] Training active (last activity 3.2s ago)
```

#### 影响
- ✅ 用户可以清楚看到训练进度
- ✅ 自动检测60秒以上的静滞
- ✅ 所有错误都有明确的上下文信息
- ✅ CUDA 操作确保完成
- ✅ 数据问题在训练前被发现
- ✅ 训练更稳定可靠

### 技术细节
- **文件**: `train/hetero_trainer.py`
- **新增类**: `TrainingHeartbeat` (心跳监控)
- **修改方法**: `train()` (添加验证、错误处理、进度日志)
- **兼容性**: 完全向后兼容，无配置更改

---

## 2026-02-01 (深夜更新)

### 重要改进：滑动窗口自回归训练 🔧

#### 优化：PredictorHead 滑动窗口训练 ✅
- **之前的问题**: 每个序列只用最后一段训练一次，数据利用率低
- **改进方案**: 
  - **滑动窗口机制**: 在序列上滑动窗口，生成多个训练样本
  - **自回归学习**: 每个位置都学习预测未来
  - **示例**: 序列长度200，context=50，steps=10，stride=5
    - 窗口1: [0:50] → 预测 [50:60]
    - 窗口2: [5:55] → 预测 [55:65]
    - 窗口3: [10:60] → 预测 [60:70]
    - ... 生成约30个训练样本！
- **效果**: 
  - 数据利用率提升 ~30倍
  - 充分的自回归训练
  - 模型学习更充分

**实现细节**:
```python
stride = max(1, self.prediction_steps // 2)  # 重叠窗口
for start_idx in range(0, T - context_len - steps + 1, stride):
    context = seq[:, start_idx:start_idx+context_len, :]
    target = seq[:, start_idx+context_len:start_idx+context_len+steps, :]
    predictions = self.predictor(context)
    loss += mse_loss(predictions, target)
```

## 2026-02-01 (晚间更新)

### 重要修复和新增优化 🔧

#### 修复：PredictorHead 训练集成 ✅
- **问题**: PredictorHead 模块被创建但从未在训练中使用
- **解决**: 
  - 在训练循环中集成 PredictorHead
  - 使用 context_length 从序列中提取上下文和目标
  - 将预测损失加入总损失（权重：prediction_weight）
  - 将 predictor 参数添加到梯度裁剪
- **影响**: PredictorHead 现在会在每个训练批次中被训练

#### 新增：梯度累积 (Gradient Accumulation) 🆕
- **功能**: 在多个批次上累积梯度，等效于更大的批次大小
- **配置**:
  ```yaml
  training:
    gradient_accumulation_steps: 4  # 累积4个批次再更新
  ```
- **优势**:
  - 在有限GPU内存下模拟大批次训练
  - 更稳定的梯度更新
  - 提高训练效率
- **实现**:
  - 仅在累积周期开始时清零梯度
  - 损失按累积步数缩放
  - 仅在累积完成后执行优化器步骤

**修改文件**:
- `train/hetero_trainer.py`: 集成预测训练 + 梯度累积 + 滑动窗口
- `workflows/training.py`: 传递梯度累积参数
- `config/default.yaml`: 新增 gradient_accumulation_steps 配置

## 2026-02-01 (下午更新)

### 预测功能优化 🔧

#### 多步未来预测 - 上下文长度控制
- **新增参数**: `context_length` 
  - 明确指定"使用多少步预测多少步"
  - 解决了之前不清楚使用多长历史的问题
  - 示例：`context_length: 50, steps: 10` = 用50步预测10步
- **更新模块**:
  - `train/predictor.py`: PredictorHead 增加 context_length 参数
  - `train/hetero_trainer.py`: 传递 context_length 参数
  - `workflows/training.py`: 从配置读取 context_length
  - `config/default.yaml`: 新增 prediction.context_length 配置
- **文档更新**:
  - `docs/NEW_FEATURES.md`: 详细说明 context_length 用法
  - `OPTIMIZATION_DIRECTIONS.md`: 标记优化完成
  - `example_new_features.py`: 演示 context_length 功能

**配置示例**:
```yaml
prediction:
  enabled: true
  context_length: 50  # 使用最后50步作为输入
  steps: 10           # 预测接下来10步
  weight: 0.1
```

## 2026-02-01 (上午)

### 新功能实现 🎉

#### 增强训练监控
- **MetricsTracker**: 完整的训练指标追踪系统
  - 自动记录所有损失分量
  - 保存指标历史到 JSON
  - 生成训练摘要报告
- **TrainingMonitor**: 训练进度监控
  - 检测训练停滞
  - 检测异常值（NaN/Inf）
  - 自动生成警告

#### 多步未来预测
- **PredictorHead**: GRU + 注意力机制的预测模块
  - 支持多步未来状态预测
  - 自回归预测机制
  - 可配置预测步数
- **ConditionalPredictor**: 条件化预测器
  - 支持基于外部刺激的预测
  - 适用于 TMS/tACS 效应模拟

#### 配置系统增强
- 新增 `prediction` 配置节
  - `enabled`: 启用/禁用预测
  - `steps`: 预测步数
  - `weight`: 预测损失权重
- 新增 `metrics` 配置节
  - `enabled`: 启用/禁用指标追踪
  - `output_dir`: 指标保存目录

### 代码优化
- 集成 PredictorHead 到 DynamicHeteroTrainer
- 集成 MetricsTracker 到训练流程
- 更新 workflows/training.py 以支持新功能

### 文档更新
- 更新 OPTIMIZATION_DIRECTIONS.md，标记已实现的优化
- 添加详细的使用说明和代码示例
- 更新文档版本到 1.1

### 代码清理
- 移除废弃的 stim_align.py 模块及相关代码调用
- 简化训练工作流，stimulus 数据处理改为可选

### 文档重组
- 整合文档结构，保留3类核心文档
- 创建 docs/ 文件夹存放用户指南
- 统一优化方向文档命名

## 2026-01-30

### 重大重构
- 完成代码架构重构，新增约900行代码，删除约540行冗余代码
- 创建基于YAML的配置系统（utils/config.py）
- 实现统一日志系统（utils/logging_utils.py）
- 模块化训练和导出工作流（workflows/）
- 创建统一入口 main.py，支持命令行参数

### 新功能
- Unity前端集成模块
  - brain_state_exporter.py: JSON格式脑状态导出
  - stimulation_simulator.py: 虚拟刺激模拟器（4种刺激模式）
  - realtime_server.py: WebSocket实时服务器
- 数据缓存管理
- 诊断系统集成
- 配置驱动的训练流程

### 文档完善
- 创建完整的中文系统使用指南（25,000+字）
- 创建优化方向和研究思路文档（43,000+字）
- 更新 README.md 和 README_CN.md

## 历史版本

### v3 (Legacy)
- 基础训练功能
- 多模态数据处理
- 异构图神经网络实现
