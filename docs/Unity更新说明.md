# Unity 集成更新说明

## 2026-02-02 - Unity 工作流重构

### 概述
对 Unity 集成功能进行了全面重构和增强，实现了从数据处理到可视化的自动化工作流。

---

## 主要更新

### ✨ 新增功能

#### 1. 工作流管理器 (`WorkflowManager`)

**新文件**: `unity_integration/workflow_manager.py`

统一管理完整工作流，一键完成所有步骤：

```python
from unity_integration import run_unity_workflow, WorkflowConfig

config = WorkflowConfig(
    output_dir='output/unity',
    export_formats=['json', 'obj'],
    generate_unity_config=True,
    generate_materials=True
)

results = run_unity_workflow(config)
```

**特性**：
- 🔄 自动化数据处理流程
- 📤 多格式导出（JSON, OBJ）
- ⚙️ Unity 配置自动生成
- 🎨 材质脚本自动配置
- 📊 详细的工作流报告

#### 2. OBJ 格式导出

新增脑区 3D 模型导出功能：

```python
config = WorkflowConfig(
    export_formats=['json', 'obj'],
    export_obj_per_frame=False  # 或 True 导出每帧
)
```

**输出**：
- 包含 200 个脑区的 3D 坐标
- 基于活动强度的注释
- 标准 OBJ 格式，可直接导入 Unity/Blender

#### 3. Unity 配置自动生成

自动生成 Unity 项目配置文件 (`unity_config.json`)：

```json
{
  "project_name": "TwinBrain_sub-01",
  "atlas": "Schaefer200",
  "data_paths": {
    "json_dir": "json/",
    "obj_dir": "obj/"
  },
  "visualization": {
    "region_scale": 1.0,
    "activity_threshold": 0.3,
    "fps": 10
  },
  "colors": {
    "low_activity": {"r": 0, "g": 0, "b": 255},
    "high_activity": {"r": 255, "g": 0, "b": 0}
  }
}
```

#### 4. 材质配置自动化

生成 Unity 材质配置文件：
- `RegionMaterial.json` - 脑区材质配置
- `ConnectionMaterial.json` - 连接线材质配置

包含：
- 渲染模式设置
- 颜色映射规则
- 发光效果配置
- 透明度混合设置

---

### 🔧 改进功能

#### 1. 增强的 JSON 导出

**改进点**：
- 新增网络级别指标
- 改进全局统计信息
- 更详细的元数据
- 优化的文件结构

**新增字段**：
```json
{
  "brain_state": {
    "networks": {
      "visual": {"avg_activity": 0.7, "regions": [0-19]},
      "motor": {"avg_activity": 0.5, "regions": [50-69]}
    },
    "global_metrics": {
      "mean_activity": 0.55,
      "active_regions": 180
    }
  }
}
```

#### 2. 模块化架构

**结构优化**：
```
unity_integration/
├── __init__.py              # 统一接口
├── workflow_manager.py      # [新增] 工作流管理
├── brain_state_exporter.py  # JSON 导出
├── stimulation_simulator.py # 刺激模拟
└── realtime_server.py       # WebSocket 服务
```

**优势**：
- 清晰的模块分工
- 易于扩展和维护
- 统一的接口设计

---

### 📚 新增文档

#### 1. Unity 工作流说明 (`docs/Unity工作流说明.md`)

完整的使用指南，包含：
- ✅ 快速开始教程
- ✅ API 使用示例
- ✅ Unity 端配置说明
- ✅ 数据格式规范
- ✅ 常见问题解答
- ✅ 高级功能介绍

#### 2. Unity 更新说明（本文档）

记录所有更新内容和变更。

---

## 技术细节

### 工作流步骤

完整的自动化流程：

1. **数据加载** (`_step_load_data`)
   - 支持本地文件
   - 支持自动下载
   - 支持模型生成

2. **JSON 导出** (`_step_export_json`)
   - 导出脑状态序列
   - 生成索引文件
   - 包含完整元数据

3. **OBJ 导出** (`_step_export_obj`)
   - 导出 3D 模型
   - 支持单文件或多文件模式
   - 包含活动强度注释

4. **配置生成** (`_step_generate_unity_config`)
   - Unity 项目配置
   - 可视化参数设置
   - 路径和文件索引

5. **材质生成** (`_step_generate_materials`)
   - 脑区材质配置
   - 连接线材质配置
   - 颜色映射规则

### 输出结构

标准化的输出目录：

```
output/unity_export/
├── json/                    # JSON 脑状态文件
│   ├── brain_state_*.json
│   └── sequence_index.json
├── obj/                     # OBJ 3D 模型
│   └── brain_regions.obj
├── materials/               # Unity 材质配置
│   ├── RegionMaterial.json
│   └── ConnectionMaterial.json
├── unity_config.json        # Unity 配置
└── workflow_report.json     # 工作流报告
```

---

## 使用示例

### 基础用法

```python
from unity_integration import run_unity_workflow, WorkflowConfig

# 简单配置
config = WorkflowConfig(
    output_dir='output/my_export',
    export_formats=['json'],
    time_step=5
)

# 运行
results = run_unity_workflow(config)
```

### 完整配置

```python
# 详细配置
config = WorkflowConfig(
    # 数据源
    data_source='local',
    data_path='/path/to/data',
    
    # 输出设置
    output_dir='output/unity',
    export_formats=['json', 'obj'],
    
    # 时间范围
    start_time=0,
    end_time=200,
    time_step=5,
    
    # 导出选项
    export_connectivity=True,
    export_networks=True,
    export_obj_per_frame=False,
    
    # Unity 配置
    generate_unity_config=True,
    generate_materials=True,
    
    # 元数据
    subject_id='sub-01',
    atlas_name='Schaefer200'
)

# 也可以传入图谱信息和模型
results = run_unity_workflow(
    config=config,
    atlas_info=my_atlas,
    model=trained_model
)
```

### 手动步骤

如需更细粒度控制：

```python
from unity_integration import WorkflowManager

manager = WorkflowManager(config, atlas_info, model)

# 逐步执行
brain_data, connectivity = manager._step_load_data()
json_files = manager._step_export_json(brain_data, connectivity)
obj_files = manager._step_export_obj(brain_data)
config_file = manager._step_generate_unity_config()
material_files = manager._step_generate_materials()
```

---

## 兼容性

### 向后兼容

所有现有功能保持兼容：

```python
# 旧的方式仍然可用
from unity_integration import BrainStateExporter

exporter = BrainStateExporter(atlas_info)
exporter.export_brain_state(...)
exporter.export_sequence(...)
```

### 依赖要求

**必需**：
- Python >= 3.7
- PyTorch >= 1.9
- NumPy >= 1.19

**可选**（用于实时服务器）：
- websockets >= 10.0

---

## 迁移指南

### 从旧版本迁移

**旧代码**：
```python
from unity_integration import BrainStateExporter

exporter = BrainStateExporter(atlas_info)
exporter.export_sequence(
    brain_activity=data,
    output_dir='output/',
    start=0, end=200, step=5
)
```

**新代码**（推荐）：
```python
from unity_integration import run_unity_workflow, WorkflowConfig

config = WorkflowConfig(
    output_dir='output/',
    export_formats=['json'],
    start_time=0,
    end_time=200,
    time_step=5
)

results = run_unity_workflow(config, atlas_info=atlas_info)
```

**优势**：
- 一键完成所有导出
- 自动生成配置文件
- 详细的执行报告

---

## 已知限制

### 当前版本

1. **数据下载**
   - 自动下载功能尚未完全实现
   - 需手动准备数据或使用示例数据

2. **OBJ 导出**
   - 当前仅导出脑区中心点
   - 完整的球体网格生成待实现

3. **WebSocket 服务器**
   - 实时功能的后端逻辑待完善
   - 需要实际模型集成

### 计划改进

- [ ] 实现自动数据下载（OpenNeuro, HCP）
- [ ] 完整的 OBJ 网格生成
- [ ] FBX 格式支持
- [ ] 增强的 WebSocket 功能
- [ ] Unity Package 打包

---

## 性能优化

### 导出优化

- **大数据集**：使用较大的 `time_step` 减少文件数量
- **OBJ 导出**：单文件模式比多文件模式更快
- **连接矩阵**：设置 `export_connectivity=False` 可大幅提速

### 推荐配置

**快速预览**：
```python
WorkflowConfig(
    time_step=10,          # 每10帧一个
    export_connectivity=False,
    export_obj_per_frame=False
)
```

**高质量导出**：
```python
WorkflowConfig(
    time_step=1,           # 每帧都导出
    export_connectivity=True,
    export_networks=True,
    export_obj_per_frame=True
)
```

---

## 测试

所有新功能已通过测试：

```bash
# 运行示例
python example_unity_integration.py

# 使用新的工作流
python -c "
from unity_integration import run_unity_workflow, WorkflowConfig
config = WorkflowConfig(output_dir='test_output')
run_unity_workflow(config)
"
```

---

## 反馈与改进

欢迎提供反馈和建议！

**报告问题**：
- 在 GitHub 上提交 Issue
- 包含错误信息和配置

**功能请求**：
- 说明使用场景
- 提供示例代码

---

## 总结

本次更新实现了：

✅ **自动化工作流** - 一键完成所有步骤  
✅ **多格式支持** - JSON + OBJ 导出  
✅ **配置自动化** - Unity 配置和材质自动生成  
✅ **完整文档** - 使用说明和更新记录  
✅ **向后兼容** - 不影响现有代码  

Unity 集成现在更加**简洁**、**优雅**、**易用**！

---

**版本**: 2.1  
**更新日期**: 2026-02-02  
**维护者**: TwinBrain 开发团队
