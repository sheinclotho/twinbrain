# TwinBrain Unity 架构说明

## 概述

TwinBrain Unity 部分是一个独立的可视化模块，用于3D展示大脑活动数据。该模块设计简单、稳定，与主项目松耦合。

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Unity 前端                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ C# 脚本      │  │ 3D 模型(OBJ) │  │ JSON数据    │ │
│  │ - 数据加载   │  │ - 脑区模型   │  │ - 状态文件  │ │
│  │ - 可视化     │  │              │  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │ WebSocket (可选)
┌──────────────────────┴──────────────────────────────┐
│              Python 后端 (可选)                      │
│  ┌─────────────────────────────────────────────┐   │
│  │ unity_integration/ 模块                      │   │
│  │ - realtime_server.py  (实时服务)            │   │
│  │ - brain_state_exporter.py (JSON导出)        │   │
│  │ - obj_generator.py (OBJ模型生成)            │   │
│  │ - freesurfer_loader.py (FreeSurfer数据加载) │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 核心模块

### 1. 项目初始化（一次性）

**脚本**: `setup_unity_project.py`

**功能**:
- 创建Unity项目文件夹结构
- 从FreeSurfer数据生成3D脑区模型（OBJ格式）
- 生成Unity C#脚本模板
- 创建配置文件

**使用场景**: 新建项目时运行一次

```bash
# 使用FreeSurfer数据生成OBJ模型
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer

# 仅创建基础结构
python setup_unity_project.py --auto-setup
```

### 2. 数据处理模块（unity_integration/）

#### 2.1 brain_state_exporter.py
将脑活动数据导出为JSON格式，供Unity加载。

**核心类**: `BrainStateExporter`

**输出格式**:
```json
{
  "timestamp": "2024-02-13T12:00:00",
  "brain_state": {
    "regions": [
      {
        "id": 1,
        "label": "LH_Vis_1",
        "position": {"x": 10.2, "y": 20.4, "z": 30.8},
        "activity": {
          "fmri": {"amplitude": 0.75}
        }
      }
    ]
  }
}
```

#### 2.2 obj_generator.py
生成脑区3D模型（OBJ格式）。

**核心类**: `OBJGenerator`

**输出**: `region_0001.obj`, `region_0002.obj`, ...

#### 2.3 freesurfer_loader.py
加载FreeSurfer表面数据。

**核心类**: `FreeSurferLoader`

**支持文件**:
- `lh.pial`, `rh.pial` (表面网格)
- `*.annot` (脑区标注)

#### 2.4 realtime_server.py (可选)
提供实时WebSocket服务，支持Unity与后端模型交互。

**核心类**: `BrainVisualizationServer`

**功能**:
- 实时预测
- 虚拟刺激模拟
- 状态更新推送

### 3. Unity前端组件

Unity前端由`setup_unity_project.py`生成的C#脚本组成，包括：

- **BrainDataLoader.cs**: JSON数据加载
- **AnimationController.cs**: 时间序列动画
- **StimulationInput.cs**: 刺激输入界面
- **ModelInterface.cs**: 后端通信接口

## 数据流程

### 离线模式（推荐用于稳定可靠的展示）

```
FreeSurfer数据
    ↓ [setup_unity_project.py]
OBJ 3D模型
    ↓
预处理的脑数据 (.pkl/.npy)
    ↓ [brain_state_exporter.py]
JSON状态文件
    ↓ [Unity加载]
3D可视化
```

### 实时模式（可选，用于交互式演示）

```
Unity前端
    ↓ [WebSocket请求]
Python后端 (realtime_server)
    ↓ [模型预测]
JSON响应
    ↓ [Unity更新]
3D可视化
```

## 文件格式规范

### JSON状态文件

**位置**: `unity_project/brain_data/model_output/brain_state_*.json`

**结构**:
- `timestamp`: 时间戳
- `brain_state.regions[]`: 脑区数组
  - `id`: 脑区ID（1-200）
  - `label`: 脑区标签
  - `position`: 3D坐标
  - `activity`: 活动数据
    - `fmri.amplitude`: fMRI信号强度（0-1）
    - `eeg.amplitude`: EEG信号强度（0-1，可选）

### OBJ模型文件

**位置**: `unity_project/Unity_Assets/Models/region_*.obj`

**命名**: `region_0001.obj` 到 `region_0200.obj`（根据脑图谱不同）

**格式**: 标准Wavefront OBJ格式

## 配置系统

### unity_config.json

**位置**: `unity_project/unity_config.json`

**关键配置项**:
```json
{
  "atlas": "Schaefer200",
  "visualization": {
    "region_scale": 1.0,
    "activity_threshold": 0.3,
    "fps": 10,
    "use_obj_models": true
  },
  "data_paths": {
    "json_dir": "brain_data/model_output",
    "obj_dir": "Unity_Assets/Models"
  }
}
```

## 技术选型

### Python依赖
- **numpy**: 数值计算
- **nibabel**: FreeSurfer数据读取（可选）
- **websockets**: 实时通信（可选）
- **torch**: 模型预测（可选）

### Unity要求
- Unity 2019.1+ 
- Newtonsoft.Json包（用于JSON解析）
- 运行时OBJ加载器（可选，用于动态加载模型）

## 设计原则

1. **简单性**: 核心功能仅需JSON文件即可运行
2. **独立性**: Unity部分不依赖主项目的训练/预测功能
3. **稳定性**: 使用静态JSON文件，避免实时计算的复杂性
4. **可扩展性**: 支持可选的实时模式用于高级用户

## 性能考虑

### 优化建议
- 对于200+脑区，使用LOD（细节层次）系统
- 启用GPU Instancing减少DrawCall
- 限制帧率（推荐10-15 FPS）
- 使用活动阈值过滤低活动脑区

### 性能指标
- **加载时间**: < 5秒（200个脑区）
- **帧率**: 30+ FPS（在中等配置电脑上）
- **内存占用**: < 500MB

## 安全考虑

- JSON文件应放在StreamingAssets目录下，不包含敏感信息
- WebSocket连接应使用本地主机（127.0.0.1）
- 避免在生产环境暴露后端服务端口

## 扩展方向

可能的扩展（当前不在实现范围内）：
- 支持更多脑图谱（AAL, Destrieux等）
- VR/AR支持
- 多被试对比可视化
- 实时EEG信号流

---

**版本**: 2.4  
**更新日期**: 2024-02
