# TwinBrain 性能优化指南

## 概述

本指南提供了优化TwinBrain系统性能的最佳实践和技巧，涵盖训练、推理和Unity可视化的各个方面。

## 目录

1. [训练性能优化](#训练性能优化)
2. [内存优化](#内存优化)
3. [数据处理优化](#数据处理优化)
4. [Unity可视化优化](#unity可视化优化)
5. [GPU优化](#gpu优化)

---

## 训练性能优化

### 使用缓存加速数据加载

**配置**：`config/default.yaml`

```yaml
data:
  use_cache: true           # 启用缓存（强烈推荐）
  cache_dir: "cache"        # 缓存目录
```

**效果**：
- 首次运行：正常速度（需要预处理）
- 后续运行：**快10-50倍**（直接加载缓存）

**缓存文件**：
- `cache/eeg_data.pt` - EEG预处理数据
- `cache/hetero_graphs.pt` - 异构图数据

### 梯度累积

适用于GPU内存不足时：

```yaml
training:
  batch_size: 4                      # 较小的批次大小
  gradient_accumulation_steps: 8    # 累积8步 = 实际批次32
```

**优势**：
- 内存占用降低
- 等效于更大的批次大小
- 训练稳定性提升

### 学习率调度

```yaml
training:
  warmup_epochs: 5           # 预热阶段
  warmup_learning_rate: 0.0001  # 较低的初始学习率
  main_epochs: 60            # 主训练阶段
  finetune_epochs: 30        # 微调阶段
  finetune_temp_weight: 5.0  # 增强时间对齐
```

### 禁用非必要功能

**训练时**：

```yaml
diagnostics:
  enabled: false  # 跳过诊断输出（节省时间）

metrics:
  enabled: false  # 不保存详细指标
```

**性能提升**：5-10%

### 多GPU训练

（未来版本支持）

```python
# 示例代码（当前版本需要手动实现）
import torch.nn as nn
model = nn.DataParallel(model)
```

---

## 内存优化

### CUDA缓存清理

**自动清理**：

```yaml
training:
  clear_cache_frequency: 1  # 每个epoch清理一次
```

**手动清理**：

```python
import torch
import gc

# 在训练循环中
if step % 100 == 0:
    torch.cuda.empty_cache()
    gc.collect()
```

### 数据类型优化

使用混合精度训练：

```python
# 在trainer中启用
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**内存节省**：30-50%
**速度提升**：10-30%

### 限制数据大小

```yaml
data:
  max_time_points: 200       # 限制时间点数量
  max_channels: 64           # 限制EEG通道数
  downsample_factor: 2       # 降采样因子
```

### 动态权重计算优化

```yaml
dynamic_weighting:
  enabled: true
  eeg_window_size: 50        # 较小的窗口（节省内存）
  fmri_window_size: 100      # 较小的窗口
  disable_in_finetune: true  # 微调时禁用（节省计算）
```

---

## 数据处理优化

### EEG处理优化

**分块处理**：

```python
# 在 eeg_mapper.py 中已实现
max_time_chunk = 10000  # 每次处理10000个时间点
```

**效果**：
- 避免OOM（Out of Memory）
- 处理长时间序列
- 保持内存稳定

### fMRI预处理优化

**并行处理**：

```python
from joblib import Parallel, delayed

# 并行加载多个任务
results = Parallel(n_jobs=4)(
    delayed(load_single_task)(task) for task in tasks
)
```

### 图构建优化

**稀疏连接**：

```python
# 只保留强连接
threshold = 0.3
mask = connectivity_matrix > threshold
sparse_matrix = connectivity_matrix * mask
```

**内存节省**：50-70%（对于大型网络）

---

## Unity可视化优化

### 连接数限制

已在 `brain_state_exporter.py` 中实现：

```python
max_connections = 10000  # 限制最大连接数
```

**效果**：
- 减少渲染负担
- 提升帧率
- 降低内存使用

### 帧率控制

在Unity中：

```csharp
public class BrainVisualization : MonoBehaviour
{
    public float fps = 10f;  // 降低帧率以提升性能
    
    // 降低更新频率
    public int activityThreshold = 0.3f;  // 只显示活跃脑区
}
```

### LOD（Level of Detail）

```csharp
// 在Unity中实现距离相关的细节层次
public float detailDistance = 100f;

void Update()
{
    float distance = Vector3.Distance(Camera.main.transform.position, transform.position);
    
    if (distance > detailDistance)
    {
        // 使用简化模型
        useSimplifiedModel = true;
    }
}
```

### 对象池

```csharp
// 复用游戏对象而不是频繁创建/销毁
private Queue<GameObject> regionPool = new Queue<GameObject>();

GameObject GetRegion()
{
    if (regionPool.Count > 0)
        return regionPool.Dequeue();
    else
        return Instantiate(regionPrefab);
}

void ReturnRegion(GameObject region)
{
    region.SetActive(false);
    regionPool.Enqueue(region);
}
```

### 批量渲染

```csharp
// 使用Unity的GPU Instancing
Material material;
material.enableInstancing = true;
```

---

## GPU优化

### CUDA设置

**环境变量**：

```bash
# 限制GPU内存增长
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 使用特定GPU
export CUDA_VISIBLE_DEVICES=0

# 禁用CUDA（使用CPU）
export CUDA_VISIBLE_DEVICES=-1
```

### cuDNN基准测试

```python
import torch

# 启用cuDNN自动调优（首次运行慢，后续快）
torch.backends.cudnn.benchmark = True

# 确定性模式（可重复，但较慢）
torch.backends.cudnn.deterministic = True
```

### 内存分配器

```python
# 使用原生分配器（可能更快）
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native'
```

---

## 性能基准测试

### 训练速度

**典型配置**：
- CPU (8 cores): ~30-60 分钟/epoch
- GPU (RTX 3080): ~5-10 分钟/epoch
- GPU + 优化: ~3-5 分钟/epoch

### 内存使用

**典型值**：
- CPU RAM: 8-16 GB
- GPU VRAM: 4-8 GB
- 优化后GPU VRAM: 2-4 GB

### 推理速度

- 单时间点预测: ~10-50 ms
- 序列预测(50步): ~500-1000 ms
- Unity实时可视化: 10-30 FPS

---

## 性能监控

### Python性能分析

```bash
# 使用cProfile
python -m cProfile -o output.prof main.py train --config config/default.yaml

# 查看结果
python -c "import pstats; p = pstats.Stats('output.prof'); p.sort_stats('cumulative').print_stats(20)"
```

### 内存分析

```bash
# 安装memory_profiler
pip install memory_profiler

# 运行分析
python -m memory_profiler main.py train --config config/default.yaml
```

### GPU监控

```bash
# NVIDIA GPU
nvidia-smi -l 1  # 每秒更新

# PyTorch GPU内存
python -c "
import torch
print('GPU memory allocated:', torch.cuda.memory_allocated() / 1024**3, 'GB')
print('GPU memory reserved:', torch.cuda.memory_reserved() / 1024**3, 'GB')
"
```

### 训练监控

```python
# 在训练脚本中添加
import time

start_time = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # 训练代码
    trainer.train(num_epochs=1)
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    print(f"Epoch {epoch}: {epoch_time:.2f}s (total: {total_time/60:.1f}min)")
```

---

## 优化检查清单

### 训练前

- [ ] 启用数据缓存
- [ ] 设置合适的批次大小
- [ ] 配置梯度累积（如果GPU内存不足）
- [ ] 禁用非必要的诊断输出
- [ ] 检查CUDA可用性

### 训练中

- [ ] 监控GPU/CPU使用率
- [ ] 监控内存使用
- [ ] 定期清理CUDA缓存
- [ ] 保存检查点

### 训练后

- [ ] 验证模型文件格式（.pt）
- [ ] 检查模型文件大小是否合理
- [ ] 测试模型加载速度
- [ ] 运行推理性能测试

### Unity集成前

- [ ] 限制连接数量（<10000）
- [ ] 设置合适的帧率
- [ ] 启用对象池
- [ ] 优化脑区数量（如果太多）

---

## 常见性能瓶颈

### 1. 数据加载慢

**原因**：
- 未启用缓存
- 磁盘IO慢
- 数据预处理复杂

**解决**：
- 启用缓存 ✓
- 使用SSD
- 预处理数据并保存

### 2. 训练慢

**原因**：
- GPU未充分利用
- 批次太小
- 学习率不当

**解决**：
- 检查GPU使用率
- 增大批次（使用梯度累积）
- 调整学习率

### 3. 内存不足

**原因**：
- 批次太大
- 数据未释放
- CUDA缓存积累

**解决**：
- 减小批次大小
- 使用梯度累积
- 定期清理缓存

### 4. Unity卡顿

**原因**：
- 连接数太多
- 帧率太高
- 对象频繁创建/销毁

**解决**：
- 限制连接数 ✓
- 降低帧率
- 使用对象池

---

## 最佳实践总结

### 推荐配置（平衡性能和质量）

```yaml
# config/optimized.yaml
data:
  use_cache: true
  cache_dir: "cache"

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  clear_cache_frequency: 1
  warmup_epochs: 5
  main_epochs: 60
  finetune_epochs: 30

diagnostics:
  enabled: false  # 训练时禁用

metrics:
  enabled: true
  save_full_history: false

dynamic_weighting:
  enabled: true
  eeg_window_size: 50
  fmri_window_size: 100
  disable_in_finetune: true
```

### 推荐工作流

1. **首次运行**：
   - 启用诊断
   - 使用小数据集测试
   - 验证所有功能

2. **正式训练**：
   - 禁用诊断
   - 启用缓存
   - 使用完整数据

3. **微调优化**：
   - 调整超参数
   - 监控性能指标
   - 保存最佳模型

4. **Unity部署**：
   - 使用优化的导出设置
   - 限制连接数
   - 测试实时性能

---

## 参考资料

- PyTorch性能优化：https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Unity优化指南：https://docs.unity3d.com/Manual/OptimizingGraphicsPerformance.html
- TwinBrain文档：
  - `使用指南.md`
  - `MODEL_FORMAT.md`
  - `TROUBLESHOOTING.md`

---

**最后更新**: 2024-02
**版本**: 2.4
