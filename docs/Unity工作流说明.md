# TwinBrain Unity 工作流说明

## 概述

本文档说明如何使用 TwinBrain 系统的 Unity 集成功能，实现从数据处理到 3D 可视化的完整工作流。

## 核心功能

### 1. 自动化工作流
- ✅ 数据下载和预处理
- ✅ 脑图转换（200个脑区）
- ✅ 多格式导出（JSON, OBJ）
- ✅ Unity 配置自动生成
- ✅ 材质脚本自动配置

### 2. 前后端集成
- ✅ JSON 格式数据导出
- ✅ WebSocket 实时通信
- ✅ 虚拟刺激模拟
- ✅ 前端输入 → 后端响应

## 快速开始

### 方法一：使用工作流管理器（推荐）

```python
from unity_integration import run_unity_workflow, WorkflowConfig

# 配置工作流
config = WorkflowConfig(
    data_source='local',           # 数据源: 'local', 'download', 'model'
    output_dir='output/unity',     # 输出目录
    export_formats=['json', 'obj'], # 导出格式
    start_time=0,                   # 起始时间点
    end_time=200,                   # 结束时间点
    time_step=5,                    # 时间步长
    export_connectivity=True,       # 导出连接
    generate_unity_config=True,     # 生成Unity配置
    generate_materials=True,        # 生成材质配置
    subject_id='sub-01',            # 被试ID
    atlas_name='Schaefer200'        # 图谱名称
)

# 运行完整工作流
results = run_unity_workflow(config)

print(f"✅ 完成！生成了 {len(results['output_files'])} 个文件")
print(f"📁 输出目录: output/unity/")
```

**输出结构：**
```
output/unity/
├── json/                      # JSON 格式脑状态
│   ├── brain_state_0000.json
│   ├── brain_state_0005.json
│   ├── ...
│   └── sequence_index.json
├── obj/                       # OBJ 格式脑模型
│   └── brain_regions.obj
├── materials/                 # Unity 材质配置
│   ├── RegionMaterial.json
│   └── ConnectionMaterial.json
├── unity_config.json          # Unity 项目配置
└── workflow_report.json       # 工作流报告
```

### 方法二：手动导出

#### 1. 导出单个脑状态

```python
from unity_integration import BrainStateExporter

# 准备数据
brain_activity = {
    'fmri': fmri_tensor,  # [N_regions, T, Features]
    'eeg': eeg_tensor
}

connectivity = {
    'structural': connectivity_matrix  # [N_regions, N_regions]
}

# 创建导出器
exporter = BrainStateExporter(atlas_info)

# 导出
exporter.export_brain_state(
    brain_activity=brain_activity,
    connectivity=connectivity,
    time_point=100,
    output_path='brain_state.json'
)
```

#### 2. 导出时间序列

```python
# 导出动画序列
exporter.export_sequence(
    brain_activity=brain_activity,
    output_dir='output/sequence/',
    start=0,
    end=200,
    step=5,
    connectivity=connectivity
)
```

#### 3. 虚拟刺激模拟

```python
from unity_integration import StimulationSimulator, StimulationConfig

# 创建模拟器
simulator = StimulationSimulator(
    n_regions=200,
    connectivity=connectivity_matrix
)

# 配置刺激
stim_config = StimulationConfig(
    target_regions=[10, 15, 20],  # 目标脑区
    amplitude=0.5,                 # 强度
    duration=20,                   # 持续时间
    pattern='sine',                # 模式
    frequency=10.0                 # 频率(Hz)
)

# 模拟响应
trajectory, metrics = simulator.simulate_response(
    initial_state=current_state,
    config=stim_config,
    n_steps=50
)
```

## Unity 端使用

### 1. 项目设置

**安装依赖：**
- Unity 2020.3 或更高版本
- Newtonsoft.Json 包（通过 Package Manager 安装）

**添加脚本：**
1. 复制 `unity_examples/BrainVisualization.cs` 到 Unity 项目
2. 创建空 GameObject，添加 `BrainVisualization` 组件

### 2. 配置

根据生成的 `unity_config.json` 配置 Unity：

```json
{
  "visualization": {
    "region_scale": 1.0,
    "activity_threshold": 0.3,
    "connection_threshold": 0.5,
    "show_connections": true,
    "fps": 10,
    "auto_play": true
  },
  "colors": {
    "low_activity": {"r": 0, "g": 0, "b": 255},
    "high_activity": {"r": 255, "g": 0, "b": 0}
  }
}
```

### 3. 运行

**静态可视化：**
- JSON Path: `output/unity/json/brain_state_0100.json`
- Load Sequence: ✗

**动画可视化：**
- JSON Path: `output/unity/json/`
- Load Sequence: ✓
- Auto Play: ✓

**控制键：**
- `空格键`: 播放/暂停
- `R`: 重新加载

## 实时通信

### 启动服务器

```python
from unity_integration import BrainVisualizationServer

server = BrainVisualizationServer(
    model=trained_model,
    exporter=exporter,
    simulator=simulator,
    port=8765
)

# 启动
server.start()  # Unity 连接: ws://localhost:8765
```

### Unity 客户端

通过 WebSocket 发送请求：

```csharp
// 获取当前脑状态
{
  "type": "get_state"
}

// 预测未来状态
{
  "type": "predict",
  "n_steps": 10
}

// 模拟刺激
{
  "type": "simulate",
  "stimulation": {
    "target_regions": [10, 15],
    "amplitude": 0.5,
    "pattern": "sine"
  }
}
```

## 数据格式

### JSON 格式说明

```json
{
  "version": "2.0",
  "timestamp": "2026-02-02T08:00:00",
  "metadata": {
    "subject": "sub-01",
    "atlas": "Schaefer200",
    "time_point": 100
  },
  "brain_state": {
    "regions": [
      {
        "id": 0,
        "label": "Visual_1",
        "position": {"x": -5, "y": -85, "z": 5},
        "activity": {
          "fmri": {"amplitude": 0.75, "raw_value": 1.5},
          "eeg": {"amplitude": 0.60, "raw_value": 0.8}
        }
      }
    ],
    "connections": [
      {
        "source": 0,
        "target": 1,
        "strength": 0.65,
        "type": "structural"
      }
    ],
    "global_metrics": {
      "mean_activity": 0.55,
      "std_activity": 0.15,
      "max_activity": 0.95,
      "active_regions": 180
    }
  },
  "stimulation": {
    "active": true,
    "target_regions": [10, 15],
    "amplitude": 0.5
  }
}
```

### OBJ 格式

OBJ 文件包含脑区位置和活动强度：

```obj
# TwinBrain Brain Regions OBJ Export
# Atlas: Schaefer200
v -5.0 -85.0 5.0
# Region 0: activity=0.750
v 10.0 -80.0 -10.0
# Region 1: activity=0.620
...
```

## 高级功能

### 1. 条件预测

```python
# 在给定刺激条件下预测
prediction = model.predict(
    initial_state=current_state,
    stimulation=stim_config,
    n_steps=20
)
```

### 2. 逆向刺激设计

```python
# 设计刺激方案以达到目标状态
optimal_stim = simulator.design_inverse_stimulation(
    initial_state=current_state,
    target_state=desired_state,
    max_amplitude=1.0
)
```

### 3. 网络分析

导出的 JSON 包含网络级别指标：

```python
brain_state['brain_state']['networks']
# {
#   "visual": {"avg_activity": 0.7, "regions": [0-19]},
#   "motor": {"avg_activity": 0.5, "regions": [50-69]},
#   ...
# }
```

## 常见问题

### Q1: 如何处理大数据集？

使用分批导出：

```python
config = WorkflowConfig(
    start_time=0,
    end_time=1000,
    time_step=10,  # 每隔10帧导出一次
    ...
)
```

### Q2: 如何自定义颜色映射？

编辑 `unity_config.json` 中的 colors 配置，或在 Unity 中直接修改材质。

### Q3: 性能优化建议

- 提高 `activity_threshold` 减少显示的脑区数量
- 提高 `connection_threshold` 减少连接线
- 降低帧率 (fps)
- 使用简化的脑区模型

### Q4: 如何导入自己的数据？

```python
# 加载预处理后的数据
from preprocess import FMRI_Preprocessor

preprocessor = FMRI_Preprocessor()
fmri_ts = preprocessor.preprocess('subject_fmri.nii.gz')

# 转换为 tensor
fmri_tensor = torch.from_numpy(fmri_ts).float()

# 导出
config = WorkflowConfig(
    data_source='local',
    data_path='path/to/data',
    ...
)
```

## 工作流步骤详解

完整工作流包含以下步骤：

1. **数据准备**
   - 加载或下载脑成像数据
   - 预处理（标准化、去趋势等）

2. **模型推理**（可选）
   - 使用训练好的模型生成预测
   - 或加载已有的分析结果

3. **格式转换**
   - 导出 JSON（Unity 加载）
   - 导出 OBJ（3D 模型）

4. **配置生成**
   - Unity 项目配置
   - 材质和渲染设置

5. **Unity 可视化**
   - 加载数据
   - 实时渲染
   - 交互控制

## 参考资料

- **完整 API 文档**: 见各模块的 docstring
- **示例代码**: `example_unity_integration.py`
- **Unity 脚本**: `unity_examples/BrainVisualization.cs`
- **系统使用指南**: `docs/TwinBrain系统使用指南.md`

## 联系与支持

遇到问题请参考主 README 或提交 Issue。

---

**版本**: 1.0  
**最后更新**: 2026-02-02  
**维护者**: TwinBrain 开发团队
