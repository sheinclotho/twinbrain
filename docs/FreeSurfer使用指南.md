# FreeSurfer 表面数据使用指南

## 概述

TwinBrain 现在支持直接使用 FreeSurfer 表面文件（.pial）和注释文件（.annot）进行 Unity 可视化。这使得您可以使用真实的大脑表面模型，而不是生成的合成数据。

## ⚠️ 重要概念：FreeSurfer 与时序数据

### FreeSurfer 文件的作用

**FreeSurfer 文件只提供大脑的空间结构**：
- 🧠 **解剖结构**：脑区的位置、形状、表面网格
- 🏷️ **分区标签**：每个脑区的名称和网络归属
- 🎨 **颜色信息**：脑区的标准颜色
- ❌ **不包含**：脑活动的时间变化数据

### 时序数据需要从其他来源获取

**JSON 导出需要的时序数据来源**：

1. **真实 fMRI 数据**：
   ```python
   # 加载您的 fMRI 时间序列
   fmri_data = load_fmri_timeseries('subject_01_fmri.nii.gz')
   # 形状: [n_regions, n_timepoints, n_features]
   ```

2. **训练模型的输出**：
   ```python
   # 使用训练好的模型生成预测
   predictions = model.predict(initial_state, n_steps=200)
   fmri_data = predictions['fmri']
   ```

3. **缓存的图数据**：
   ```python
   # 从预处理的 HeteroData 加载
   data = torch.load('preprocessed_data.pt')
   fmri_data = data['fmri'].x_seq
   ```

### 正确的数据流

```
┌─────────────────┐     ┌──────────────────┐
│ FreeSurfer 文件  │     │ fMRI/EEG 时序数据│
│ (空间结构)       │     │ (时间变化)        │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │ JSON 导出    │
              │ (时空数据)   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Unity 可视化 │
              └─────────────┘
```

### 当前实现的限制

⚠️ **注意**：当前实现中，`_load_freesurfer_data()` 方法使用**占位符数据**：

```python
# 这是占位符！不是真实数据！
fmri_data = torch.randn(n_regions, n_timepoints, n_features)
```

**您需要替换为真实数据**。参见本文档后面的 [高级用法](#高级用法) 部分。

## 支持的文件类型

### 表面文件（Surface Files）
- **lh.pial**: 左半球表面网格
- **rh.pial**: 右半球表面网格

这些文件包含大脑皮层表面的三维网格数据（顶点和面）。

### 注释文件（Annotation Files）
- **lh.Schaefer2018_200Parcels_7Networks_order.annot**: 左半球脑区分割
- **rh.Schaefer2018_200Parcels_7Networks_order.annot**: 右半球脑区分割

这些文件包含脑区的标签、颜色和网络信息。

## 快速开始

### 方法一：使用 Python API

```python
from unity_integration import run_unity_workflow, WorkflowConfig

# 配置 FreeSurfer 数据源
config = WorkflowConfig(
    data_source='freesurfer',  # 使用 FreeSurfer 数据
    
    # FreeSurfer 文件路径
    freesurfer_lh_surface='path/to/lh.pial',
    freesurfer_rh_surface='path/to/rh.pial',
    freesurfer_lh_annot='path/to/lh.Schaefer2018_200Parcels_7Networks_order.annot',
    freesurfer_rh_annot='path/to/rh.Schaefer2018_200Parcels_7Networks_order.annot',
    
    # 输出配置
    output_dir='output/freesurfer_export',
    export_formats=['json', 'obj'],
    export_surface_mesh=True,  # 导出真实的表面网格
    
    # 其他参数
    start_time=0,
    end_time=200,
    time_step=5,
    export_connectivity=True,
    subject_id='sub-01',
    atlas_name='Schaefer2018_200Parcels_7Networks'
)

# 运行工作流
results = run_unity_workflow(config)

print(f"✅ 完成！生成了 {len(results['output_files'])} 个文件")
```

### 方法二：使用命令行

```bash
# 使用 FreeSurfer 数据导出
python unity_automation.py \
    --mode export \
    --freesurfer \
    --lh-surface data/lh.pial \
    --rh-surface data/rh.pial \
    --lh-annot data/lh.Schaefer2018_200Parcels_7Networks_order.annot \
    --rh-annot data/rh.Schaefer2018_200Parcels_7Networks_order.annot \
    --export-surface \
    --output freesurfer_output
```

### 命令行参数说明

- `--freesurfer`: 启用 FreeSurfer 数据模式
- `--lh-surface`: 左半球表面文件路径
- `--rh-surface`: 右半球表面文件路径
- `--lh-annot`: 左半球注释文件路径
- `--rh-annot`: 右半球注释文件路径
- `--export-surface`: 导出完整的表面网格（可选）
- `--output`: 输出目录

## 输出文件结构

使用 FreeSurfer 数据后，输出目录结构如下：

```
freesurfer_output/
├── json/                              # JSON 格式脑状态
│   ├── brain_state_0000.json
│   ├── brain_state_0005.json
│   ├── ...
│   └── sequence_index.json
├── obj/                               # OBJ 格式模型
│   ├── brain_regions.obj              # 脑区球体模型
│   └── brain_surface_bilateral.obj   # 完整表面网格（如果启用）
├── materials/                         # Unity 材质配置
│   ├── RegionMaterial.json
│   └── ConnectionMaterial.json
├── unity_config.json                  # Unity 项目配置
└── workflow_report.json               # 工作流报告
```

## 数据处理流程

### 1. 加载表面文件
系统使用 `nibabel` 库读取 FreeSurfer 表面文件：
```python
from unity_integration import FreeSurferLoader

loader = FreeSurferLoader()

# 加载左半球
loader.load_surface('lh.pial', hemisphere='lh')
loader.load_annotation('lh.Schaefer2018_200Parcels_7Networks_order.annot', hemisphere='lh')

# 加载右半球
loader.load_surface('rh.pial', hemisphere='rh')
loader.load_annotation('rh.Schaefer2018_200Parcels_7Networks_order.annot', hemisphere='rh')
```

### 2. 提取脑区信息
系统自动从注释文件中提取：
- 脑区标签（region labels）
- 脑区质心位置（region centroids）
- 网络归属（network assignment）
- 颜色信息（color information）

### 3. 转换为 TwinBrain 格式
FreeSurfer 数据被转换为 TwinBrain 标准的 atlas_info 格式：
```python
atlas_info = loader.to_atlas_info(atlas_name="FreeSurfer_Schaefer200")

# atlas_info 结构示例：
{
    'name': 'FreeSurfer_Schaefer200',
    'n_regions': 200,
    'regions': {
        '1': {
            'label': '7Networks_LH_Vis_1',
            'hemisphere': 'lh',
            'xyz': [-10.5, -85.2, 5.1],
            'network': 'Visual',
            'color': [120, 18, 134]
        },
        ...
    }
}
```

### 4. 导出可视化数据
- **JSON 文件**: 包含每个时间点的脑区活动数据
- **OBJ 文件**: 
  - 球体模型：每个脑区用一个球体表示
  - 表面网格：真实的大脑皮层表面（可选）

## 高级用法

### 仅加载 FreeSurfer 数据（不导出）

```python
from unity_integration import load_freesurfer_data

# 加载 FreeSurfer 数据
atlas_info, loader = load_freesurfer_data(
    lh_surface="lh.pial",
    rh_surface="rh.pial",
    lh_annot="lh.Schaefer2018_200Parcels_7Networks_order.annot",
    rh_annot="rh.Schaefer2018_200Parcels_7Networks_order.annot"
)

print(f"加载了 {atlas_info['n_regions']} 个脑区")

# 查看脑区信息
for region_id, region_info in atlas_info['regions'].items():
    print(f"脑区 {region_id}: {region_info['label']}")
    print(f"  位置: {region_info['xyz']}")
    print(f"  网络: {region_info['network']}")
```

### 如何使用真实的 fMRI/EEG 数据

**重要**：FreeSurfer 只定义空间结构，您必须提供时序数据。

#### 方法 1：加载 fMRI 数据文件

```python
import nibabel as nib
import numpy as np
import torch

# 1. 加载 FreeSurfer 图谱
from unity_integration import load_freesurfer_data
atlas_info, loader = load_freesurfer_data(
    lh_surface="lh.pial",
    rh_surface="rh.pial",
    lh_annot="lh.Schaefer2018_200Parcels_7Networks_order.annot",
    rh_annot="rh.Schaefer2018_200Parcels_7Networks_order.annot"
)

# 2. 加载 fMRI 数据（NIfTI 格式）
fmri_img = nib.load('subject_01_task_rest_bold.nii.gz')
fmri_array = fmri_img.get_fdata()  # 形状: [x, y, z, timepoints]

# 3. 提取每个脑区的时间序列
from nilearn.maskers import NiftiLabelsMasker
masker = NiftiLabelsMasker(
    labels_img='Schaefer2018_200Parcels_7Networks.nii.gz',
    standardize=True
)
region_timeseries = masker.fit_transform(fmri_img)
# 形状: [timepoints, n_regions]

# 4. 转换为 TwinBrain 格式
fmri_data = torch.FloatTensor(region_timeseries.T)  # [n_regions, timepoints]
fmri_data = fmri_data.unsqueeze(-1)  # [n_regions, timepoints, 1]

# 5. 使用真实数据运行工作流
from unity_integration import WorkflowManager, WorkflowConfig

# 创建自定义数据字典
brain_data = {
    'fmri': fmri_data,
    'eeg': None  # 如果没有 EEG 数据
}

# 创建工作流管理器
config = WorkflowConfig(
    data_source='example',  # 使用自定义数据
    output_dir='output/real_fmri',
    export_formats=['json', 'obj']
)

manager = WorkflowManager(config=config, atlas_info=atlas_info)
# 手动设置数据
# manager.brain_data = brain_data
# results = manager.run_full_workflow()
```

#### 方法 2：使用训练模型的输出

```python
# 1. 加载训练好的模型
from train.hetero_trainer import HeteroGCNTrainer
model = torch.load('checkpoints/best_model.pt')

# 2. 加载 FreeSurfer 图谱
atlas_info, loader = load_freesurfer_data(...)

# 3. 准备初始状态
initial_state = get_initial_brain_state()  # 您的函数

# 4. 使用模型生成预测
model.eval()
with torch.no_grad():
    predictions = model.predict(
        initial_state,
        n_steps=200,
        use_prediction_head=True
    )

# 5. 提取预测的 fMRI 数据
fmri_predictions = predictions['fmri']
# 形状: [n_regions, n_timepoints, n_features]

# 6. 导出到 Unity
brain_data = {
    'fmri': fmri_predictions,
    'eeg': predictions.get('eeg', None)
}

# 继续工作流...
```

#### 方法 3：使用缓存的图数据

```python
# 1. 加载预处理的数据
data_list = torch.load('data/processed/subject_01_data.pt')

# 2. 提取时间序列
fmri_sequences = []
for data in data_list:
    if 'fmri' in data.node_types:
        fmri_seq = data['fmri'].x_seq  # [n_regions, time, features]
        fmri_sequences.append(fmri_seq)

# 3. 合并所有序列
fmri_data = torch.cat(fmri_sequences, dim=1)  # 沿时间轴合并

# 4. 使用这些数据...
```

#### 方法 4：修改 WorkflowManager 直接集成

**推荐方式**：修改 `unity_integration/workflow_manager.py`

```python
# 在 _load_freesurfer_data() 方法中替换占位符
def _load_freesurfer_data(self) -> tuple:
    """加载FreeSurfer数据并加载真实活动数据"""
    
    # ... FreeSurfer 加载代码 ...
    
    # === 替换这部分 ===
    # 旧代码（占位符）：
    # fmri_data = torch.randn(n_regions, n_timepoints, n_features)
    
    # 新代码（真实数据）：
    from your_data_module import load_subject_fmri
    fmri_data = load_subject_fmri(
        subject_id=self.config.subject_id,
        n_regions=n_regions,
        atlas_info=self.atlas_info
    )
    
    # 确保形状正确
    assert fmri_data.shape[0] == n_regions, "区域数量不匹配"
    if fmri_data.dim() == 2:
        fmri_data = fmri_data.unsqueeze(-1)  # 添加特征维度
    
    # ... 继续其余代码 ...
```

### 导出不同的表面组合

```python
# 导出左右半球为分离的文件
loader.export_surfaces_as_obj(
    output_dir='output/surfaces',
    combine_hemispheres=False  # 分别导出
)
# 生成: brain_surface_lh.obj, brain_surface_rh.obj

# 导出合并的双半球文件
loader.export_surfaces_as_obj(
    output_dir='output/surfaces',
    combine_hemispheres=True  # 合并导出
)
# 生成: brain_surface_bilateral.obj
```

### 与现有模型结合使用

```python
from unity_integration import WorkflowManager, WorkflowConfig

# 加载 FreeSurfer 图谱
atlas_info, loader = load_freesurfer_data(...)

# 创建工作流管理器（使用已加载的图谱）
config = WorkflowConfig(
    data_source='model',  # 使用模型生成数据
    output_dir='output/combined'
)

manager = WorkflowManager(
    config=config,
    atlas_info=atlas_info,  # 使用 FreeSurfer 图谱
    model=your_trained_model  # 使用您的训练模型
)

results = manager.run_full_workflow()
```

## 在 Unity 中使用

### 1. 导入 OBJ 文件
将生成的 OBJ 文件拖入 Unity 项目：
- `brain_regions.obj`: 脑区球体模型
- `brain_surface_bilateral.obj`: 完整大脑表面（可选）

### 2. 加载配置
使用生成的 `unity_config.json` 配置可视化参数。

### 3. 加载动画数据
使用 `json/` 目录中的时间序列数据创建动画。

## 常见问题

### Q: 支持哪些 FreeSurfer 版本？
A: 支持 FreeSurfer 5.x 和 6.x 生成的文件。

### Q: 必须使用 Schaefer 分割吗？
A: 不是。系统支持任何 FreeSurfer 注释文件，但 Schaefer 分割是推荐的标准。

### Q: 如何处理自定义分割？
A: 只要是标准的 FreeSurfer .annot 格式，系统都可以加载。网络分配可能需要手动调整。

### Q: 表面文件很大，会影响性能吗？
A: 完整表面文件（.pial）可能包含数万到十万个顶点。如果只需要脑区可视化，可以不使用 `--export-surface` 选项，只导出球体模型。

### Q: 能否混合使用 FreeSurfer 和其他数据？
A: 可以。您可以使用 FreeSurfer 定义脑区位置，然后使用其他来源的活动数据（fMRI, EEG 等）。

## 示例脚本

### 完整示例：从 FreeSurfer 文件到 Unity 可视化

```python
#!/usr/bin/env python3
"""
FreeSurfer to Unity 完整示例
"""
from pathlib import Path
from unity_integration import run_unity_workflow, WorkflowConfig

# 设置文件路径
data_dir = Path('data/freesurfer')
output_dir = Path('output/unity_freesurfer')

# 配置工作流
config = WorkflowConfig(
    # FreeSurfer 数据源
    data_source='freesurfer',
    freesurfer_lh_surface=str(data_dir / 'lh.pial'),
    freesurfer_rh_surface=str(data_dir / 'rh.pial'),
    freesurfer_lh_annot=str(data_dir / 'lh.Schaefer2018_200Parcels_7Networks_order.annot'),
    freesurfer_rh_annot=str(data_dir / 'rh.Schaefer2018_200Parcels_7Networks_order.annot'),
    
    # 输出配置
    output_dir=str(output_dir),
    export_formats=['json', 'obj'],
    export_surface_mesh=True,
    
    # 时间序列参数
    start_time=0,
    end_time=200,
    time_step=5,
    
    # 可视化选项
    export_connectivity=True,
    export_networks=True,
    
    # Unity 配置
    generate_unity_config=True,
    generate_materials=True,
    
    # 主体信息
    subject_id='sub-01',
    atlas_name='Schaefer2018_200Parcels_7Networks'
)

# 运行工作流
print("开始处理 FreeSurfer 数据...")
results = run_unity_workflow(config)

# 打印结果
print("\n" + "="*60)
print("处理完成！")
print("="*60)
print(f"完成步骤: {', '.join(results['steps_completed'])}")
print(f"生成文件数: {len(results['output_files'])}")
print(f"输出目录: {output_dir}")
print("\n生成的文件:")
for file in results['output_files']:
    print(f"  - {file}")
```

## 技术细节

### FreeSurfer 文件格式
- **.pial**: 二进制文件，包含顶点坐标和面索引
- **.annot**: 二进制文件，包含每个顶点的区域标签和颜色表

### 坐标系统
FreeSurfer 使用 RAS（Right-Anterior-Superior）坐标系统：
- X: 右侧为正
- Y: 前方为正
- Z: 上方为正

Unity 使用左手坐标系统，系统会自动处理坐标转换。

### 质心计算
脑区质心通过计算该区域所有顶点的平均位置得到：
```python
centroid = region_vertices.mean(axis=0)
```

## 相关文档

- [Unity工作流说明](Unity工作流说明.md) - Unity集成完整指南
- [系统使用指南](TwinBrain系统使用指南.md) - TwinBrain完整用户手册
- [Unity更新说明](Unity更新说明.md) - Unity功能更新记录

## 联系与支持

如有问题或建议，请通过以下方式联系：
- GitHub Issues: https://github.com/sheinclotho/twinbrain/issues
- 项目主页: https://github.com/sheinclotho/twinbrain

---

**最后更新**: 2026-02-04  
**版本**: 2.3  
**新增功能**: FreeSurfer 表面数据支持
