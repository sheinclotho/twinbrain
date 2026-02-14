# TwinBrain 问题排查指南

## 概述

本指南帮助您快速诊断和解决TwinBrain项目中的常见问题。

## 目录

1. [训练相关问题](#训练相关问题)
2. [模型加载问题](#模型加载问题)
3. [Unity集成问题](#unity集成问题)
4. [内存和性能问题](#内存和性能问题)
5. [数据处理问题](#数据处理问题)

---

## 训练相关问题

### 问题：训练时出现 "CUDA out of memory"

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate XX.XX MiB
```

**解决方案**：

1. **减小批次大小**：编辑配置文件 `config/default.yaml`
```yaml
training:
  batch_size: 4  # 从32减小到4
```

2. **启用梯度累积**：
```yaml
training:
  gradient_accumulation_steps: 4  # 累积4步后更新
```

3. **清理CUDA缓存**：
```yaml
training:
  clear_cache_frequency: 1  # 每个epoch清理一次
```

4. **使用CPU**：
```bash
python main.py train --config config/default.yaml --no-cuda
```

### 问题：训练损失不下降

**可能原因**：
- 学习率设置不当
- 数据归一化问题
- 模型权重初始化问题

**诊断步骤**：

1. 检查诊断输出：
```bash
# 训练后查看诊断目录
ls test_file3/sub-01/results/diagnostics/
# 查看摘要
cat test_file3/sub-01/results/diagnostics/summary.txt
```

2. 查看训练指标：
```bash
# 查看损失曲线
python -c "
import json
with open('test_file3/sub-01/results/metrics/training_metrics.json') as f:
    metrics = json.load(f)
    print('Loss history:', metrics.get('loss_history', []))
"
```

3. 调整学习率：
```yaml
training:
  warmup_learning_rate: 0.0001  # 降低warmup学习率
  main_epochs: 100  # 增加训练轮次
```

### 问题：训练中断后如何恢复

**解决方案**：

TwinBrain支持从检查点恢复：
```python
# 在training.py中添加
trainer.load_model("test_file3/sub-01/results/hetero_gnn_trained.pt")
trainer.train(num_epochs=50)  # 继续训练
```

---

## 模型加载问题

### 问题：找不到模型文件

**错误信息**：
```
❌ 模型文件不存在: results/best_model.pth
```

**解决方案**：

1. **检查文件位置**：
```bash
# 查找所有.pt模型文件
find . -name "*.pt" -type f

# 常见位置：
# test_file3/sub-01/results/hetero_gnn_trained.pt
```

2. **使用正确路径**：
```bash
# 正确的命令
python unity_startup.py --model test_file3/sub-01/results/hetero_gnn_trained.pt

# 或使用相对路径
cd test_file3/sub-01/results
python ../../../unity_startup.py --model hetero_gnn_trained.pt
```

### 问题：模型文件格式错误

**错误信息**：
```
⚠ 警告: 文件扩展名为 .pth，推荐使用 .pt
```

**解决方案**：

1. **重命名文件**：
```bash
# 简单重命名（内部格式相同）
mv best_model.pth hetero_gnn_trained.pt
```

2. **验证模型格式**：
```python
import torch
checkpoint = torch.load("hetero_gnn_trained.pt", map_location='cpu')
print("Keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Direct state_dict")
```

### 问题：PyTorch版本不兼容

**错误信息**：
```
RuntimeError: Error(s) in loading state_dict
```

**解决方案**：

1. **检查PyTorch版本**：
```bash
python -c "import torch; print(torch.__version__)"
# 推荐: 1.12.0+
```

2. **使用weights_only=False加载**：
```python
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
```

---

## Unity集成问题

### 问题：Unity无法连接到后端

**症状**：
- Unity Console显示 "WebSocket connection failed"
- 后端服务器没有收到连接

**排查步骤**：

1. **确认后端服务器正在运行**：
```bash
python unity_startup.py --model test_file3/sub-01/results/hetero_gnn_trained.pt
# 应该看到: ✓ WebSocket server started on ws://localhost:8765
```

2. **检查端口冲突**：
```bash
# Linux/Mac
lsof -i :8765

# Windows
netstat -ano | findstr :8765

# 使用其他端口
python unity_startup.py --port 8766
```

3. **检查防火墙设置**：
```bash
# 确保端口8765未被防火墙阻止
# Windows: 控制面板 -> 防火墙 -> 允许应用
# Linux: sudo ufw allow 8765
```

### 问题：Unity中看不到大脑模型

**可能原因**：
- OBJ文件缺失
- JSON数据文件格式错误
- BrainVisualization脚本配置错误

**解决方案**：

1. **检查OBJ文件**：
```bash
# 查看OBJ模型是否存在
ls Unity_TwinBrain/Assets/StreamingAssets/OBJ/
# 应该看到: region_0001.obj, region_0002.obj, ...
```

2. **验证JSON数据**：
```python
import json
with open('Unity_TwinBrain/Assets/StreamingAssets/state/brain_state.json') as f:
    data = json.load(f)
    print("Regions:", len(data['brain_state']['regions']))
    print("Version:", data['version'])
```

3. **检查Unity Console日志**：
在Unity Editor中打开Console窗口查看错误信息

### 问题：BrainConfigLoader报错

**错误信息**：
```
NullReferenceException: Object reference not set to an instance of an object
TwinBrain.BrainConfigLoader.LoadConfiguration
```

**解决方案**：

这是由于旧版本中的`predictedColor`属性已被移除。请确保：

1. **使用最新版本的脚本**：
```bash
git pull origin main
# 或重新运行 setup_unity_project.py 生成最新脚本
python setup_unity_project.py --auto-setup
```

2. **更新配置文件**：
```json
{
  "colors": {
    "low_activity": {"r": 0, "g": 0, "b": 255},
    "high_activity": {"r": 255, "g": 0, "b": 0}
    // 移除 "predicted_signal" 配置
  }
}
```

---

## 内存和性能问题

### 问题：程序运行缓慢

**诊断**：

1. **监控资源使用**：
```bash
# Linux/Mac
htop

# Python内存分析
python -m memory_profiler main.py train --config config/default.yaml
```

2. **启用性能日志**：
```bash
python main.py train --config config/default.yaml --log-level DEBUG
```

**优化建议**：

1. **启用缓存**：
```yaml
data:
  use_cache: true  # 启用数据缓存
  cache_dir: "cache"
```

2. **减少诊断输出**：
```yaml
diagnostics:
  enabled: false  # 禁用诊断以加速训练
```

3. **使用更小的数据集**：
```yaml
data:
  max_time_points: 100  # 限制时间点数量
```

### 问题：内存持续增长（内存泄漏）

**解决方案**：

1. **检查张量释放**：
```python
# 在训练循环中定期清理
if epoch % 10 == 0:
    torch.cuda.empty_cache()
    import gc
    gc.collect()
```

2. **使用上下文管理器**：
```python
with torch.no_grad():
    # 推理代码
    predictions = model(data)
```

3. **避免保存整个历史**：
```yaml
metrics:
  save_full_history: false  # 只保存摘要
```

---

## 数据处理问题

### 问题：数据加载失败

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'test_file3/sub-01/eeg/...'
```

**解决方案**：

1. **检查数据目录结构**：
```bash
tree test_file3/sub-01/ -L 2
# 应该包含:
# ├── eeg/
# ├── func/
# └── dwi/
```

2. **验证文件格式**：
```bash
# EEG文件应为 .fif 格式
ls test_file3/sub-01/eeg/*.fif

# fMRI文件应为 .nii.gz 格式
ls test_file3/sub-01/func/*.nii.gz
```

### 问题：缓存文件损坏

**症状**：
```
RuntimeError: Error loading 'cache/eeg_data.pt'
```

**解决方案**：

1. **删除损坏的缓存**：
```bash
rm -rf test_file3/sub-01/results/cache/*.pt
```

2. **重新生成缓存**：
```bash
python main.py train --config config/default.yaml
# 系统会自动重新生成缓存文件
```

3. **禁用缓存进行测试**：
```yaml
data:
  use_cache: false
```

---

## 常用诊断命令

### 系统信息检查

```bash
# 检查Python版本
python --version  # 应该是 3.8+

# 检查依赖包
pip list | grep -E "torch|numpy|mne|nibabel"

# 检查CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 模型验证脚本

创建 `validate_model.py`：

```python
#!/usr/bin/env python3
import torch
from pathlib import Path
import sys

def validate_model(path):
    """验证模型文件格式和内容"""
    path = Path(path)
    
    print(f"验证模型: {path}")
    print("-" * 50)
    
    # 1. 检查文件存在
    if not path.exists():
        print("✗ 文件不存在")
        return False
    print(f"✓ 文件存在 (大小: {path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # 2. 检查扩展名
    if path.suffix != '.pt':
        print(f"⚠ 警告: 扩展名为 {path.suffix}，推荐使用 .pt")
    else:
        print("✓ 文件扩展名正确 (.pt)")
    
    # 3. 尝试加载
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print("✓ 文件可以正常加载")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 4. 检查内容
    if isinstance(checkpoint, dict):
        print(f"✓ Checkpoint是字典格式")
        print(f"  包含的键: {list(checkpoint.keys())}")
        
        if 'model' in checkpoint:
            print("  ✓ 包含 'model' 键")
        elif 'model_state_dict' in checkpoint:
            print("  ✓ 包含 'model_state_dict' 键")
        else:
            print("  ⚠ 未找到标准模型键")
        
        if 'epoch' in checkpoint:
            print(f"  - 训练轮数: {checkpoint['epoch']}")
        if 'best_loss' in checkpoint:
            print(f"  - 最佳损失: {checkpoint['best_loss']:.4f}")
    else:
        print("⚠ Checkpoint不是字典格式")
    
    print("-" * 50)
    print("✓ 验证完成")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python validate_model.py <model_path>")
        sys.exit(1)
    
    validate_model(sys.argv[1])
```

使用方法：
```bash
python validate_model.py test_file3/sub-01/results/hetero_gnn_trained.pt
```

---

## 获取帮助

如果以上方法都无法解决您的问题：

1. **查看日志文件**：
```bash
# 训练日志
cat logs/training.log

# Unity后端日志
cat unity_startup.log
```

2. **启用详细日志**：
```bash
python main.py train --config config/default.yaml --log-level DEBUG
```

3. **提交Issue**：
   - 访问：https://github.com/sheinclotho/twinbrain/issues
   - 提供：错误信息、日志、系统信息、复现步骤

4. **查看文档**：
   - 使用指南：`使用指南.md`
   - 模型说明：`模型说明.md`
   - Unity集成：`Unity一键使用指南.md`
   - 格式规范：`MODEL_FORMAT.md`

---

**最后更新**: 2024-02
**版本**: 2.4
