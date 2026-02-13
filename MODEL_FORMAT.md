# TwinBrain 模型和缓存文件格式规范

## 概述

本文档规定了TwinBrain项目中所有模型和缓存文件的标准格式，以确保训练、推理和Unity集成之间的兼容性。

## 文件扩展名标准

### 模型文件：`.pt`

**所有TwinBrain模型文件必须使用 `.pt` 扩展名**

- ✅ 正确：`hetero_gnn_trained.pt`
- ❌ 错误：`best_model.pth`

**原因**：
- PyTorch官方推荐使用 `.pt` 作为模型文件扩展名
- 与缓存文件格式保持一致
- 避免与Python源文件(.pth被某些系统识别为Python header)混淆

### 缓存文件：`.pt`

**所有数据缓存文件使用 `.pt` 扩展名**

- ✅ `eeg_data.pt`
- ✅ `hetero_graphs.pt`
- ✅ `fmri_cache.pt`

## 文件位置标准

### 训练输出结构

```
test_file3/
└── sub-01/
    └── results/
        ├── hetero_gnn_trained.pt       # 训练完成的模型
        ├── nodes.json                   # 脑区节点配置
        ├── cache/                       # 缓存目录
        │   ├── eeg_data.pt             # EEG数据缓存
        │   └── hetero_graphs.pt        # 异构图缓存
        ├── diagnostics/                 # 诊断输出
        └── metrics/                     # 训练指标
```

### Unity集成使用

Unity后端服务器默认查找模型路径（按优先级）：

1. 用户指定路径：`--model path/to/model.pt`
2. 训练输出路径：`test_file3/sub-01/results/hetero_gnn_trained.pt`
3. 结果目录：`results/hetero_gnn_trained.pt`

## 模型文件格式详细规范

### 标准Checkpoint格式

TwinBrain模型应保存为包含以下键的字典：

```python
checkpoint = {
    "model": model.state_dict(),           # 必需：模型状态字典
    "aligner": aligner_state,              # 推荐：对齐器状态
    "optimizer": optimizer.state_dict(),   # 可选：优化器状态
    "scheduler": scheduler.state_dict(),   # 可选：学习率调度器
    "epoch": current_epoch,                # 推荐：训练轮次
    "best_loss": best_loss,                # 推荐：最佳损失值
    "config": config_dict,                 # 推荐：训练配置
}

# 保存
torch.save(checkpoint, "hetero_gnn_trained.pt")
```

### 加载模型

```python
# 标准加载方式
checkpoint = torch.load("hetero_gnn_trained.pt", map_location=device, weights_only=False)

# 提取模型状态
if isinstance(checkpoint, dict):
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint  # 假设整个字典是模型状态
else:
    model_state = checkpoint  # 直接是state_dict

# 加载到模型
model.load_state_dict(model_state)
```

## 缓存文件格式

### EEG数据缓存 (`eeg_data.pt`)

保存预处理后的EEG数据，格式为字典或直接数据结构：

```python
eeg_data = {
    "task_name": {
        "features": torch.Tensor,      # 特征张量
        "timestamps": np.ndarray,       # 时间戳
        "channels": List[str],          # 通道名称
        # 其他元数据...
    }
}

torch.save(eeg_data, "cache/eeg_data.pt")
```

### 异构图缓存 (`hetero_graphs.pt`)

保存构建好的异构图数据：

```python
# PyTorch Geometric HeteroData对象
hetero_graphs = [hetero_data1, hetero_data2, ...]

torch.save(hetero_graphs, "cache/hetero_graphs.pt")
```

## 版本兼容性

### PyTorch版本

- **推荐**：PyTorch 1.12.0+
- **最低要求**：PyTorch 1.8.0

### 保存选项

使用 `weights_only=False` 以支持复杂对象：

```python
# 保存
torch.save(data, path)  # 默认设置

# 加载
data = torch.load(path, map_location=device, weights_only=False)
```

## 文件命名约定

### 模型文件命名

| 场景 | 命名格式 | 示例 |
|------|---------|------|
| 训练完成 | `hetero_gnn_trained.pt` | 标准输出 |
| 检查点 | `checkpoint_epoch_{N}.pt` | `checkpoint_epoch_050.pt` |
| 最佳模型 | `best_model_epoch_{N}.pt` | `best_model_epoch_075.pt` |

### 缓存文件命名

| 数据类型 | 文件名 |
|---------|--------|
| EEG数据 | `eeg_data.pt` |
| fMRI数据 | `fmri_data.pt` |
| 异构图 | `hetero_graphs.pt` |
| 预处理结果 | `preprocessed_{modality}.pt` |

## 迁移指南

### 从 `.pth` 迁移到 `.pt`

如果你有旧的 `.pth` 文件：

```bash
# 简单重命名即可（内部格式相同）
mv best_model.pth hetero_gnn_trained.pt
```

### 更新代码引用

在所有Python代码中：

```python
# 旧代码
model_path = "results/best_model.pth"  # ❌

# 新代码
model_path = "results/hetero_gnn_trained.pt"  # ✅
```

在Unity集成脚本中：

```bash
# 旧命令
python unity_startup.py --model results/best_model.pth  # ❌

# 新命令
python unity_startup.py --model test_file3/sub-01/results/hetero_gnn_trained.pt  # ✅
```

## 常见问题

### Q: 为什么使用 `.pt` 而不是 `.pth`？

A: 
1. `.pt` 是PyTorch官方推荐的扩展名
2. 与缓存文件格式统一
3. 避免与其他文件类型冲突

### Q: 如何验证模型文件格式？

A: 使用以下代码检查：

```python
import torch
from pathlib import Path

def validate_model_file(path):
    path = Path(path)
    
    # 检查扩展名
    if path.suffix != '.pt':
        print(f"⚠ 警告: 文件扩展名为 {path.suffix}，推荐使用 .pt")
    
    # 尝试加载
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        print(f"✓ 文件可以正常加载")
        
        # 检查内容
        if isinstance(checkpoint, dict):
            print(f"  包含的键: {list(checkpoint.keys())}")
            if 'model' in checkpoint or 'model_state_dict' in checkpoint:
                print(f"  ✓ 包含模型状态")
        return True
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False

# 使用
validate_model_file("hetero_gnn_trained.pt")
```

### Q: 缓存文件损坏怎么办？

A: 
1. 删除 `cache/` 目录下的缓存文件
2. 重新运行训练，系统会自动重新生成缓存

```bash
rm -rf test_file3/sub-01/results/cache/*.pt
python main.py train --config config/default.yaml
```

## 参考资料

- [PyTorch保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch Geometric数据格式](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- TwinBrain项目文档：`使用指南.md`, `模型说明.md`

---

**最后更新**: 2024-02
**版本**: 2.4
