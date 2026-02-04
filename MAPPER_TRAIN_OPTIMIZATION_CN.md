# Mapper 和 Train 文件夹优化建议

## 📋 文档概述

本文档详细分析 `mapper/` 和 `train/` 文件夹中的代码，提供优化建议和具体实施方案。

**分析日期**: 2026-01-30  
**分析范围**: mapper/ (7个文件) + train/ (10个文件)  
**目标**: 识别冗余、优化结构、提升代码质量

---

## 📂 Mapper 文件夹分析与优化

### 当前文件清单

| 文件名 | 行数 | 状态 | 被导入次数 | 说明 |
|--------|------|------|-----------|------|
| multi_modal_mapper.py | ~200 | ✅ 核心 | 高频 | 统一的多模态数据加载 |
| atlas_mapper.py | ~150 | ✅ 核心 | 高频 | 脑图谱处理和区域映射 |
| dti_mapper.py | ~180 | ⚠️ 评估 | 中频 | DTI数据加载 |
| bids_mapper.py | ~250 | ❌ 废弃 | 0次 | BIDS格式（项目未使用） |
| eeg_mapper.py | ~200 | ❌ 废弃 | 0次 | 已被multi_modal_mapper替代 |
| eeg_roi_mapper.py | ~100 | ❌ 废弃 | 0次 | 无任何导入 |
| aligned_latent.py | ~80 | ❌ 废弃 | 0次 | 早期实验代码 |

### 详细分析

#### 1. ✅ multi_modal_mapper.py - 核心映射器

**功能**: 统一的多模态数据加载接口
- fMRI数据加载和预处理
- EEG数据加载和预处理
- DTI连接矩阵加载
- 多模态数据整合

**使用情况**:
```python
# workflows/training.py
from mapper.multi_modal_mapper import MultiModalMapper

mapper = MultiModalMapper(...)
fmri_data = mapper.load_fmri(...)
eeg_data = mapper.load_eeg(...)
dti_matrix = mapper.load_dti(...)
```

**代码质量**: ⭐⭐⭐⭐ (优秀)
- 清晰的接口设计
- 良好的错误处理
- 支持数据缓存

**优化建议**:
1. **添加类型提示**
   ```python
   # 改进前
   def load_fmri(self, func_dir, tasks, atlas_file):
       ...
   
   # 改进后
   def load_fmri(
       self,
       func_dir: Path,
       tasks: List[str],
       atlas_file: Path
   ) -> Dict[str, torch.Tensor]:
       """
       Load and preprocess fMRI data.
       
       Args:
           func_dir: Functional MRI directory
           tasks: List of task names
           atlas_file: Brain atlas file path
           
       Returns:
           Dictionary mapping task names to fMRI tensors
       """
       ...
   ```

2. **改进缓存机制**
   ```python
   # 当前: 简单的文件存在检查
   if cache_file.exists():
       return torch.load(cache_file)
   
   # 建议: 添加版本控制和哈希验证
   def get_cache_key(self, params: dict) -> str:
       """Generate cache key with version and params hash"""
       import hashlib
       param_str = json.dumps(params, sort_keys=True)
       param_hash = hashlib.md5(param_str.encode()).hexdigest()
       return f"v1_{param_hash}"
   ```

3. **统一错误处理**
   ```python
   # 建议: 自定义异常类
   class DataLoadError(Exception):
       """Data loading error"""
       pass
   
   class AtlasMismatchError(DataLoadError):
       """Atlas dimensions don't match"""
       pass
   
   # 使用
   try:
       data = load_nifti(file_path)
   except FileNotFoundError:
       raise DataLoadError(f"File not found: {file_path}")
   except ValueError as e:
       raise AtlasMismatchError(f"Atlas error: {e}")
   ```

---

#### 2. ✅ atlas_mapper.py - 脑图谱处理

**功能**: 脑图谱加载、区域映射、坐标转换
- 支持多种图谱格式 (AAL, Schaefer, etc.)
- 区域名称和索引映射
- 坐标空间转换

**使用情况**:
```python
# 多处导入
from mapper.atlas_mapper import BrainAtlas, load_atlas

atlas = load_atlas("schaefer200")
region_idx = atlas.get_region_index("LH_Vis1")
```

**代码质量**: ⭐⭐⭐⭐ (优秀)

**优化建议**:
1. **添加图谱验证**
   ```python
   def validate_atlas(self) -> bool:
       """Validate atlas integrity"""
       checks = [
           len(self.region_names) == self.n_regions,
           self.label_file.exists(),
           self.nifti_file.exists()
       ]
       return all(checks)
   ```

2. **支持自定义图谱**
   ```python
   def register_custom_atlas(
       self,
       name: str,
       nifti_path: Path,
       label_path: Path
   ):
       """Register a custom brain atlas"""
       # 验证并注册
       self.custom_atlases[name] = {
           'nifti': nifti_path,
           'labels': label_path
       }
   ```

---

#### 3. ⚠️ dti_mapper.py - DTI数据加载

**当前状态**: 功能重叠，需评估

**问题分析**:
```python
# dti_mapper.py
class DTIMapper:
    def load_dti_matrix(self, dti_dir):
        # DTI连接矩阵加载
        ...

# multi_modal_mapper.py 中也有
class MultiModalMapper:
    def load_dti(self, dti_dir):
        # 功能相同
        ...
```

**优化方案 A: 删除 dti_mapper.py**
```python
# 如果 multi_modal_mapper.py 已包含所有功能
# 直接删除 dti_mapper.py，减少180行代码

# 更新所有导入
# 从: from mapper.dti_mapper import DTIMapper
# 到: from mapper.multi_modal_mapper import MultiModalMapper
```

**优化方案 B: 保留为独立模块**
```python
# 如果 dti_mapper 有特殊的DTI处理功能
# 重构为专门的DTI工具模块

# mapper/dti_mapper.py
class DTIProcessor:
    """专门的DTI处理工具"""
    
    @staticmethod
    def load_dti_matrix(dti_path: Path) -> torch.Tensor:
        """加载DTI连接矩阵"""
        ...
    
    @staticmethod
    def compute_fa_map(dti_path: Path) -> np.ndarray:
        """计算FA图"""
        ...
    
    @staticmethod
    def tract_segmentation(dti_path: Path) -> Dict:
        """纤维束分割"""
        ...

# multi_modal_mapper.py 调用
from mapper.dti_mapper import DTIProcessor

class MultiModalMapper:
    def load_dti(self, dti_dir):
        return DTIProcessor.load_dti_matrix(dti_dir)
```

**推荐**: 先检查代码差异，如果功能完全重复则删除，否则重构为工具类。

---

#### 4. ❌ bids_mapper.py - 完全未使用

**证据**:
```bash
# grep 搜索结果
$ grep -r "from.*bids_mapper" .
./main_v4.py:# from mapper.bids_mapper import BIDSMapper  # Unused
./main_export_latent.py:# from mapper.bids_mapper import BIDSMapper  # Unused
./utils/function.py:# from mapper.bids_mapper import BIDSMapper
```

**结论**: 所有导入都被注释，证明完全未使用

**操作**: **立即删除**
```bash
git rm mapper/bids_mapper.py
# 减少 ~250 行代码
```

**理由**:
- 项目不使用BIDS标准数据格式
- 代码中明确标注"Unused"
- 无任何活跃导入
- 保留会增加维护负担

---

#### 5. ❌ eeg_mapper.py - 已被替代

**证据**:
```bash
# grep 搜索结果
$ grep -r "from.*eeg_mapper" .
./main_v4.py:# from mapper.eeg_mapper import EEGMapper  # Unused
./main_export_latent.py:from mapper.eeg_mapper import EEGMapper  # 唯一使用处
./utils/function.py:# from mapper.eeg_mapper import EEGMapper
```

**分析**:
- 仅 `main_export_latent.py` 导入，但该文件是旧版本
- 新架构 `workflows/export_latent.py` 使用 `MultiModalMapper`
- 功能已整合到 `multi_modal_mapper.py`

**操作**: **删除**
```bash
git rm mapper/eeg_mapper.py
# 减少 ~200 行代码
```

**迁移指南**:
```python
# 旧代码 (main_export_latent.py)
from mapper.eeg_mapper import EEGMapper
eeg_mapper = EEGMapper()
eeg_data = eeg_mapper.load_eeg(eeg_dir)

# 新代码 (workflows/export_latent.py)
from mapper.multi_modal_mapper import MultiModalMapper
mapper = MultiModalMapper()
eeg_data = mapper.load_eeg(eeg_dir)
```

---

#### 6. ❌ eeg_roi_mapper.py - 无任何导入

**证据**:
```bash
$ grep -r "eeg_roi_mapper" .
# 无任何结果
```

**结论**: 完全孤立的文件，无任何使用

**操作**: **立即删除**
```bash
git rm mapper/eeg_roi_mapper.py
# 减少 ~100 行代码
```

---

#### 7. ❌ aligned_latent.py - 实验性代码

**证据**:
```bash
$ grep -r "aligned_latent" .
# 无任何导入
```

**推测**: 早期实验代码，未集成到主系统

**操作**: **删除**
```bash
git rm mapper/aligned_latent.py
# 减少 ~80 行代码
```

---

### Mapper 文件夹优化总结

#### 执行方案

**立即删除 (630行代码减少)**:
```bash
cd mapper/
git rm bids_mapper.py       # -250行
git rm eeg_mapper.py         # -200行
git rm eeg_roi_mapper.py     # -100行
git rm aligned_latent.py     # -80行
git commit -m "Remove unused mapper modules"
```

**评估 dti_mapper.py**:
```python
# 步骤1: 比较代码差异
diff mapper/dti_mapper.py mapper/multi_modal_mapper.py

# 步骤2: 检查是否有独特功能
grep -n "def " mapper/dti_mapper.py

# 步骤3: 决定删除或重构
```

**优化保留文件**:
1. `multi_modal_mapper.py`: 添加类型提示、改进缓存
2. `atlas_mapper.py`: 添加验证、支持自定义图谱

#### 优化后的结构

```
mapper/
├── __init__.py
├── multi_modal_mapper.py    # 【优化】添加类型提示、改进缓存
├── atlas_mapper.py           # 【优化】添加验证功能
└── (dti_mapper.py)          # 【可选】评估后决定
```

**代码减少**: 630-810行 (25-32%)

---

## 🧠 Train 文件夹分析与优化

### 当前文件清单

| 文件名 | 行数 | 状态 | 被导入次数 | 说明 |
|--------|------|------|-----------|------|
| hetero_trainer.py | ~400 | ✅ 核心 | 高频 | DynamicHeteroTrainer主训练器 |
| dynamic_hetero_gnn.py | ~350 | ✅ 核心 | 高频 | 异构图神经网络 |
| coder.py | ~200 | ✅ 核心 | 高频 | TemporalDecoder时间解码器 |
| loss_helpers.py | ~150 | ✅ 核心 | 高频 | 损失函数计算 |
| aligner.py | ~180 | ✅ 核心 | 高频 | TemporalAligner时间对齐 |
| align_helper.py | ~100 | ⚠️ 评估 | 中频 | 对齐辅助函数 |
| embed_utils.py | ~80 | ⚠️ 评估 | 低频 | 嵌入向量工具 |
| gnn_trainer.py | ~300 | ❌ 废弃 | 0次 | 旧训练器 |
| embed_analysis.py | ~120 | ❌ 罕用 | 1次 | 嵌入分析 |

### 详细分析

#### 1. ✅ hetero_trainer.py - 核心训练器

**功能**: DynamicHeteroTrainer 主训练器
- 训练循环管理
- 动态损失权重调整
- 检查点保存/加载
- 验证和评估

**使用情况**:
```python
# workflows/training.py
from train.hetero_trainer import DynamicHeteroTrainer

trainer = DynamicHeteroTrainer(
    model=gnn_model,
    data=hetero_data,
    device=device
)
trainer.train(num_epochs=100)
```

**代码质量**: ⭐⭐⭐⭐ (优秀)

**优化建议**:

1. **分离配置管理**
   ```python
   # 当前: 参数直接传入
   trainer = DynamicHeteroTrainer(
       hidden_dim=128,
       decoder_layers=3,
       lr=0.0001,
       temp_weight=5.0,
       # ... 30+ 参数
   )
   
   # 建议: 使用配置对象
   from dataclasses import dataclass
   
   @dataclass
   class TrainerConfig:
       hidden_dim: int = 128
       decoder_layers: int = 3
       learning_rate: float = 0.0001
       temp_weight: float = 5.0
       # ... 其他参数
   
   config = TrainerConfig()
   trainer = DynamicHeteroTrainer(config=config)
   ```

2. **添加训练回调**
   ```python
   class TrainingCallback:
       def on_epoch_start(self, epoch: int):
           pass
       
       def on_epoch_end(self, epoch: int, metrics: dict):
           pass
       
       def on_batch_end(self, batch: int, loss: float):
           pass
   
   class TensorBoardCallback(TrainingCallback):
       def on_epoch_end(self, epoch, metrics):
           self.writer.add_scalars("Loss", metrics, epoch)
   
   # 使用
   trainer.add_callback(TensorBoardCallback())
   trainer.train(num_epochs=100)
   ```

3. **改进检查点管理**
   ```python
   class CheckpointManager:
       def __init__(self, save_dir: Path, keep_top_k: int = 3):
           self.save_dir = save_dir
           self.keep_top_k = keep_top_k
           self.checkpoints = []
       
       def save(self, model, optimizer, epoch, metrics):
           """Save checkpoint and manage history"""
           ckpt_path = self.save_dir / f"epoch_{epoch}.pt"
           torch.save({
               'model': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'epoch': epoch,
               'metrics': metrics
           }, ckpt_path)
           
           # 保留top-k
           self._prune_checkpoints()
   ```

---

#### 2. ✅ dynamic_hetero_gnn.py - 图神经网络

**功能**: 异构图神经网络架构
- 多关系消息传递
- 支持 fMRI-fMRI, fMRI-EEG 等边类型
- 节点特征编码和解码

**代码质量**: ⭐⭐⭐⭐⭐ (优秀)

**优化建议**:

1. **模块化层定义**
   ```python
   # 当前: 所有层在一个类中
   class DynamicHeteroGNN(nn.Module):
       def __init__(self):
           self.conv1 = ...
           self.conv2 = ...
           self.decoder = ...
   
   # 建议: 分离层定义
   class HeteroConvLayer(nn.Module):
       """Single heterogeneous convolution layer"""
       def __init__(self, in_dim, out_dim, edge_types):
           super().__init__()
           self.convs = nn.ModuleDict({
               edge_type: SAGEConv(in_dim, out_dim)
               for edge_type in edge_types
           })
   
   class DynamicHeteroGNN(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.layers = nn.ModuleList([
               HeteroConvLayer(config.hidden_dim, config.hidden_dim, config.edge_types)
               for _ in range(config.num_layers)
           ])
   ```

2. **支持不同聚合方式**
   ```python
   class AggregationModule(nn.Module):
       def __init__(self, agg_type='mean'):
           super().__init__()
           self.agg_type = agg_type
       
       def forward(self, xs: List[torch.Tensor]):
           if self.agg_type == 'mean':
               return torch.stack(xs).mean(dim=0)
           elif self.agg_type == 'max':
               return torch.stack(xs).max(dim=0)[0]
           elif self.agg_type == 'attention':
               return self.attention_aggregate(xs)
   ```

---

#### 3. ✅ coder.py - 时间解码器

**功能**: TemporalDecoder 时间序列解码
- 支持 GRU、Transformer、MLP
- fMRI 和 EEG 信号重建

**代码质量**: ⭐⭐⭐⭐ (优秀)

**优化建议**:

1. **工厂模式创建解码器**
   ```python
   class DecoderFactory:
       @staticmethod
       def create(decoder_type: str, **kwargs) -> nn.Module:
           if decoder_type == 'gru':
               return GRUDecoder(**kwargs)
           elif decoder_type == 'transformer':
               return TransformerDecoder(**kwargs)
           elif decoder_type == 'mlp':
               return MLPDecoder(**kwargs)
           else:
               raise ValueError(f"Unknown decoder: {decoder_type}")
   
   # 使用
   decoder = DecoderFactory.create('gru', hidden_dim=128, num_layers=2)
   ```

2. **添加解码器配置**
   ```python
   @dataclass
   class DecoderConfig:
       type: str = 'gru'
       hidden_dim: int = 128
       num_layers: int = 2
       dropout: float = 0.1
       bidirectional: bool = False
   
   class TemporalDecoder(nn.Module):
       def __init__(self, config: DecoderConfig):
           super().__init__()
           self.config = config
           self.decoder = self._build_decoder()
   ```

---

#### 4. ✅ loss_helpers.py - 损失函数

**功能**: 损失函数计算和管理
- 重建损失
- 时间对齐损失
- 正则化损失

**代码质量**: ⭐⭐⭐⭐ (优秀)

**优化建议**:

1. **损失函数注册机制**
   ```python
   class LossRegistry:
       _losses = {}
       
       @classmethod
       def register(cls, name: str):
           def decorator(loss_fn):
               cls._losses[name] = loss_fn
               return loss_fn
           return decorator
       
       @classmethod
       def get(cls, name: str):
           return cls._losses[name]
   
   # 注册
   @LossRegistry.register('recon')
   def reconstruction_loss(pred, target):
       return F.mse_loss(pred, target)
   
   @LossRegistry.register('temporal')
   def temporal_alignment_loss(pred, target):
       return temporal_loss(pred, target)
   
   # 使用
   loss = LossRegistry.get('recon')(pred, target)
   ```

2. **组合损失管理**
   ```python
   class CompositeLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.losses = {}
           self.weights = {}
       
       def add_loss(self, name: str, loss_fn, weight: float = 1.0):
           self.losses[name] = loss_fn
           self.weights[name] = weight
       
       def forward(self, pred, target):
           total_loss = 0.0
           loss_dict = {}
           
           for name, loss_fn in self.losses.items():
               loss_val = loss_fn(pred, target)
               weighted_loss = self.weights[name] * loss_val
               total_loss += weighted_loss
               loss_dict[name] = loss_val.item()
           
           return total_loss, loss_dict
   ```

---

#### 5. ✅ aligner.py - 时间对齐

**功能**: TemporalAligner 跨模态时间对齐
- 可学习的时间对齐
- 交叉相关分析

**代码质量**: ⭐⭐⭐⭐ (优秀)

**优化建议**:

1. **多种对齐策略**
   ```python
   class AlignmentStrategy(ABC):
       @abstractmethod
       def align(self, source, target):
           pass
   
   class CrossCorrelationAlignment(AlignmentStrategy):
       def align(self, source, target):
           # 互相关对齐
           ...
   
   class DTWAlignment(AlignmentStrategy):
       def align(self, source, target):
           # 动态时间规整
           ...
   
   class LearnableAlignment(AlignmentStrategy):
       def __init__(self):
           self.alignment_net = nn.LSTM(...)
       
       def align(self, source, target):
           # 可学习对齐
           ...
   ```

---

#### 6. ⚠️ align_helper.py - 对齐辅助

**当前状态**: 需要评估是否被 aligner.py 使用

**检查依赖**:
```bash
grep -r "from.*align_helper import\|import align_helper" .
grep -r "align_helper\." train/aligner.py
```

**评估结果**:
- 如果 `aligner.py` 导入并使用 → 保留
- 如果功能可整合到 `aligner.py` → 整合后删除
- 如果完全独立且无导入 → 删除

---

#### 7. ⚠️ embed_utils.py - 嵌入工具

**当前状态**: 需要评估使用频率

**检查依赖**:
```bash
grep -r "from.*embed_utils import\|import embed_utils" .
```

**评估结果**:
- 如果被核心模块频繁使用 → 保留
- 如果仅在分析脚本中使用 → 移动到 utils/
- 如果完全无用 → 删除

---

#### 8. ❌ gnn_trainer.py - 旧训练器

**证据**:
```bash
$ grep -r "from.*gnn_trainer import" .
# 无任何结果
```

**分析**:
- 早期的单一GNN训练器
- 被 `hetero_trainer.py` 完全替代
- 不支持异构图
- 功能已过时

**操作**: **立即删除**
```bash
git rm train/gnn_trainer.py
# 减少 ~300 行代码
```

**理由**:
- `DynamicHeteroTrainer` 提供更强大的功能
- 支持异构图、多模态、动态权重
- 无任何活跃使用
- 保留会造成混淆

---

#### 9. ❌ embed_analysis.py - 嵌入分析

**当前状态**: 极少使用，功能应在 utils/

**证据**:
```bash
$ grep -r "from.*embed_analysis import" .
# 仅在个别分析脚本中使用
```

**问题**:
- 分析工具不应在 train/ 目录
- 应该在 utils/analysis.py
- 使用频率极低

**操作**: **迁移或删除**

**方案 A: 迁移到 utils/**
```bash
# 如果有有用的分析函数
git mv train/embed_analysis.py utils/embedding_analysis.py
# 更新导入
sed -i 's/from train.embed_analysis/from utils.embedding_analysis/g' **/*.py
```

**方案 B: 删除**
```bash
# 如果功能已被 utils/analysis.py 覆盖
git rm train/embed_analysis.py
# 减少 ~120 行代码
```

**推荐**: 检查是否有独特功能，如无则删除。

---

### Train 文件夹优化总结

#### 执行方案

**立即删除 (300行代码减少)**:
```bash
cd train/
git rm gnn_trainer.py        # -300行
git commit -m "Remove deprecated GNN trainer"
```

**评估后决定**:
```bash
# 1. 检查 align_helper.py
grep -r "align_helper" train/aligner.py

# 2. 检查 embed_utils.py
grep -r "embed_utils" train/

# 3. 检查 embed_analysis.py
grep -r "embed_analysis" .
# 决定迁移到 utils/ 或删除
```

**优化保留文件**:
1. `hetero_trainer.py`: 添加配置对象、回调机制、检查点管理
2. `dynamic_hetero_gnn.py`: 模块化层定义、支持多种聚合
3. `coder.py`: 工厂模式、配置对象
4. `loss_helpers.py`: 损失注册机制、组合损失
5. `aligner.py`: 多种对齐策略

#### 优化后的结构

```
train/
├── __init__.py
├── hetero_trainer.py         # 【优化】配置对象、回调机制
├── dynamic_hetero_gnn.py     # 【优化】模块化设计
├── coder.py                  # 【优化】工厂模式
├── loss_helpers.py           # 【优化】损失注册
├── aligner.py                # 【优化】多种策略
├── (align_helper.py)        # 【评估】检查依赖
└── (embed_utils.py)         # 【评估】检查使用
```

**代码减少**: 300-540行 (10-18%)

---

## 📊 总体优化统计

### 代码减少统计

| 模块 | 立即删除 | 评估删除 | 总计 | 百分比 |
|------|---------|---------|------|--------|
| **Mapper** | 630行 | 180行 | 810行 | 25-32% |
| **Train** | 300行 | 220行 | 520行 | 10-18% |
| **总计** | 930行 | 400行 | 1330行 | 17-25% |

### 文件数量变化

| 模块 | 当前 | 删除 | 剩余 | 减少比例 |
|------|------|------|------|---------|
| **Mapper** | 7 | 4-5 | 2-3 | 57-71% |
| **Train** | 9 | 1-3 | 6-8 | 11-33% |
| **总计** | 16 | 5-8 | 8-11 | 31-50% |

---

## 🚀 实施计划

### 阶段1: 立即清理 (1小时)

```bash
# 1. 删除明确废弃的文件
cd /home/runner/work/twinbrain/twinbrain

# Mapper
git rm mapper/bids_mapper.py
git rm mapper/eeg_mapper.py
git rm mapper/eeg_roi_mapper.py
git rm mapper/aligned_latent.py

# Train
git rm train/gnn_trainer.py

# 提交
git commit -m "Remove deprecated mapper and train modules

- Remove unused BIDS mapper (250 lines)
- Remove redundant EEG mapper (200 lines)
- Remove unused EEG ROI mapper (100 lines)
- Remove experimental aligned_latent (80 lines)
- Remove deprecated GNN trainer (300 lines)

Total: 930 lines removed"
```

### 阶段2: 评估和决策 (2-4小时)

```bash
# 1. 评估 dti_mapper.py
echo "=== Checking dti_mapper.py ==="
grep -r "from.*dti_mapper import" .
diff mapper/dti_mapper.py <(grep -A 50 "def load_dti" mapper/multi_modal_mapper.py)

# 2. 评估 align_helper.py
echo "=== Checking align_helper.py ==="
grep -r "align_helper" train/aligner.py
grep -r "from.*align_helper import" .

# 3. 评估 embed_utils.py
echo "=== Checking embed_utils.py ==="
grep -r "from.*embed_utils import" .
grep -r "embed_utils\." train/

# 4. 评估 embed_analysis.py
echo "=== Checking embed_analysis.py ==="
grep -r "from.*embed_analysis import" .
```

**决策矩阵**:
| 文件 | 被导入次数 | 功能重复 | 决策 |
|------|-----------|---------|------|
| dti_mapper.py | >0 | 是 | 删除 |
| dti_mapper.py | >0 | 否 | 保留 |
| align_helper.py | >0 | - | 保留 |
| align_helper.py | =0 | - | 删除 |
| embed_utils.py | >2 | - | 保留 |
| embed_utils.py | ≤2 | - | 删除 |
| embed_analysis.py | >0 | - | 迁移到utils |
| embed_analysis.py | =0 | - | 删除 |

### 阶段3: 代码优化 (1-2天)

#### 优化 mapper/multi_modal_mapper.py
```python
# 1. 添加类型提示
# 2. 改进缓存机制
# 3. 统一错误处理
# 4. 添加文档字符串
```

#### 优化 train/hetero_trainer.py
```python
# 1. 配置对象重构
# 2. 添加训练回调
# 3. 改进检查点管理
```

#### 优化其他核心文件
```python
# dynamic_hetero_gnn.py: 模块化层定义
# coder.py: 工厂模式
# loss_helpers.py: 损失注册机制
# aligner.py: 多种对齐策略
```

### 阶段4: 测试和验证 (半天)

```bash
# 1. 单元测试
python -m pytest tests/test_mapper.py
python -m pytest tests/test_train.py

# 2. 集成测试
python main.py train --config config/default.yaml --dry-run

# 3. 完整训练测试
python main.py train --config config/default.yaml
```

---

## 📋 检查清单

### Mapper 优化检查

- [ ] 删除 bids_mapper.py
- [ ] 删除 eeg_mapper.py
- [ ] 删除 eeg_roi_mapper.py
- [ ] 删除 aligned_latent.py
- [ ] 评估 dti_mapper.py (删除或保留)
- [ ] 优化 multi_modal_mapper.py (类型提示、缓存)
- [ ] 优化 atlas_mapper.py (验证功能)
- [ ] 更新所有导入引用
- [ ] 运行测试验证

### Train 优化检查

- [ ] 删除 gnn_trainer.py
- [ ] 评估 align_helper.py
- [ ] 评估 embed_utils.py
- [ ] 评估 embed_analysis.py (迁移或删除)
- [ ] 优化 hetero_trainer.py (配置、回调)
- [ ] 优化 dynamic_hetero_gnn.py (模块化)
- [ ] 优化 coder.py (工厂模式)
- [ ] 优化 loss_helpers.py (注册机制)
- [ ] 优化 aligner.py (多种策略)
- [ ] 运行测试验证

### 文档更新

- [ ] 更新 README.md
- [ ] 更新 PROJECT_MIGRATION_GUIDE_CN.md
- [ ] 创建 CHANGELOG.md
- [ ] 更新 API 文档

---

## 🎯 预期收益

### 代码质量提升

| 指标 | 改善 |
|------|------|
| 代码量 | ↓ 17-25% |
| 文件数 | ↓ 31-50% |
| 维护成本 | ↓ 显著降低 |
| 代码重复 | ↓ 消除冗余 |
| 可读性 | ↑ 更清晰 |
| 可测试性 | ↑ 更易测试 |

### 开发效率提升

- ✅ 更少的文件需要维护
- ✅ 更清晰的代码结构
- ✅ 更容易找到相关代码
- ✅ 更少的导入混淆
- ✅ 更快的代码审查

### 用户体验提升

- ✅ 更简单的项目结构
- ✅ 更少的学习曲线
- ✅ 更清晰的API
- ✅ 更好的文档

---

## 📞 总结

本文档提供了 mapper/ 和 train/ 文件夹的全面分析和优化建议:

**Mapper 文件夹**:
- 删除4个废弃文件 (630行)
- 评估1个重复文件
- 优化2个核心文件

**Train 文件夹**:
- 删除1个废弃文件 (300行)
- 评估3个边缘文件
- 优化5个核心文件

**总体收益**:
- 代码减少 17-25%
- 文件数减少 31-50%
- 维护成本显著降低

建议按照四个阶段实施优化，预计需要2-3天完成全部工作。

---

**文档版本**: 1.0  
**最后更新**: 2026-01-30  
**维护者**: TwinBrain Development Team
