# TwinBrain Unity 架构说明 (v2.5)

## 概述

TwinBrain Unity部分是一个独立的可视化模块，用于3D展示大脑活动数据。该模块设计简单、稳定，与主项目松耦合。

**版本2.5更新**：优化了通信机制、增强了自动化工具、改进了错误处理。

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Unity 前端                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ C# 脚本      │  │ 3D 模型(OBJ) │  │ JSON数据    │ │
│  │ - 数据加载   │  │ - 脑区模型   │  │ - 状态文件  │ │
│  │ - 可视化     │  │              │  │            │ │
│  │ - HTTP通信   │  │              │  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/REST (改进)
┌──────────────────────┴──────────────────────────────┐
│              Python 后端 (可选)                      │
│  ┌─────────────────────────────────────────────┐   │
│  │ unity_integration/ 模块                      │   │
│  │ - realtime_server.py  (实时服务+输入验证)   │   │
│  │ - model_server.py     (模型加载和推理)      │   │
│  │ - brain_state_exporter.py (JSON导出)        │   │
│  │ - obj_generator.py    (OBJ模型生成)         │   │
│  │ - freesurfer_loader.py (FreeSurfer数据加载) │   │
│  │ - stimulation_simulator.py (刺激模拟)       │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │ 工具脚本                                     │   │
│  │ - setup_unity_project.py (项目初始化)       │   │
│  │ - unity_startup.py (服务器启动+智能搜索)    │   │
│  │ - unity_package_installer.py (Unity安装)    │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 核心模块详解

### 1. 项目初始化（一次性）

**脚本**: `setup_unity_project.py`

**功能**:
- 创建标准化的Unity项目文件夹结构
- 从FreeSurfer数据生成3D脑区模型（OBJ格式，每个脑区独立文件）
- 生成Unity C#脚本模板
- 创建配置文件和说明文档

**使用场景**: 新建项目时运行一次

```bash
# 使用FreeSurfer数据生成OBJ模型
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer

# 仅创建基础结构（无FreeSurfer）
python setup_unity_project.py --auto-setup

# 自定义输出目录
python setup_unity_project.py --auto-setup --output-dir my_unity_assets
```

**输出结构**:
```
unity_project/
├── freesurfer_files/         # FreeSurfer数据存放
├── brain_data/
│   ├── cache/                # 预处理缓存(.pkl/.npy)
│   ├── model_output/         # JSON状态文件(Unity读取)
│   └── original/             # 原始数据
├── Unity_Assets/
│   ├── Scripts/              # C#脚本（7个）
│   └── Models/               # 3D脑区模型
└── unity_config.json         # 配置文件
```

### 2. Unity包安装器（新增v2.5）

**脚本**: `unity_package_installer.py`

**功能**:
- 验证Unity项目结构
- 自动安装所有C#脚本到Unity项目
- 创建Assembly Definition文件
- 设置StreamingAssets目录结构
- 生成UPM包定义
- 创建使用指南

**使用方法**:
```bash
# 完整安装
python unity_package_installer.py --unity-project /path/to/UnityProject

# 指定数据源
python unity_package_installer.py --unity-project /path/to/UnityProject \
    --data-dir unity_project

# 仅验证
python unity_package_installer.py --unity-project /path/to/UnityProject \
    --validate-only
```

**验证项**:
- Unity项目有效性（ProjectSettings存在）
- Assets目录结构
- Newtonsoft.Json包安装
- 脚本完整性

### 3. 数据处理模块（unity_integration/）

#### 3.1 brain_state_exporter.py
将脑活动数据导出为Unity可读的JSON格式。

**核心类**: `BrainStateExporter`

**输出格式** (v2.0):
```json
{
  "version": "2.0",
  "timestamp": "2024-02-14T12:00:00",
  "subject_id": "sub-01",
  "brain_state": {
    "regions": [
      {
        "id": 1,
        "label": "LH_Vis_1",
        "position": {"x": 10.2, "y": 20.4, "z": 30.8},
        "activity": {
          "fmri": {"amplitude": 0.75, "confidence": 0.9},
          "eeg": {"amplitude": 0.65, "confidence": 0.8}
        }
      }
    ],
    "connections": [
      {
        "source": 1,
        "target": 2,
        "strength": 0.85,
        "type": "structural"
      }
    ]
  },
  "stimulation": {
    "active": true,
    "target_regions": [10, 20],
    "amplitude": 0.5
  }
}
```

**使用**:
```bash
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output
```

#### 3.2 obj_generator.py
生成脑区3D模型（Wavefront OBJ格式）。

**核心类**: `BrainOBJGenerator`

**输出**: `region_0001.obj`, `region_0002.obj`, ...（每个脑区独立文件）

**特点**:
- 支持多边形简化（减少文件大小）
- 自动生成法线
- 材质支持
- 批量导出

#### 3.3 freesurfer_loader.py
加载FreeSurfer表面数据和标注。

**核心类**: `FreeSurferLoader`

**支持文件**:
- `lh.pial`, `rh.pial` (表面网格)
- `*.annot` (脑区标注，如Schaefer200, AAL, Destrieux)

**功能**:
- 自动识别脑图谱类型
- 提取顶点和面
- 解析脑区标签
- 计算脑区中心坐标

#### 3.4 realtime_server.py (v2.5增强)
提供实时WebSocket服务，支持Unity与后端模型交互。

**核心类**: `BrainVisualizationServer`

**改进**:
- ✅ 完整的输入验证（所有参数）
- ✅ 统一的错误响应格式
- ✅ 详细的日志记录
- ✅ 异常捕获和处理
- ✅ 请求类型日志

**API端点**:

| 端点 | 功能 | 输入验证 |
|-----|------|---------|
| `get_state` | 获取当前大脑状态 | 无参数 |
| `predict` | 请求未来预测 | n_steps: 1-1000 |
| `simulate` | 虚拟刺激模拟 | regions: 0-199, amplitude: 0.01-10.0 |
| `stream_start` | 开始流式传输 | fps: 1-60, duration: 1-3600 |
| `stream_stop` | 停止流式传输 | 无参数 |
| `convert_cache` | Cache转JSON | 路径验证 |

**请求-响应格式** (v2.5标准化):
```python
# 请求
{
    "type": "predict",
    "request_id": "req_123_456789",  # 可选，用于追踪
    "n_steps": 10
}

# 成功响应
{
    "type": "prediction",
    "success": true,
    "n_steps": 10,
    "predictions": [...],
    "request_id": "req_123_456789"
}

# 错误响应
{
    "type": "error",
    "success": false,
    "message": "Invalid n_steps: must be 1-1000",
    "request_id": "req_123_456789"
}
```

#### 3.5 model_server.py
封装模型加载和推理逻辑。

**核心类**: `ModelServer`

**功能**:
- 加载训练好的模型checkpoint
- 执行预测
- 模拟刺激响应
- 自动保存结果为JSON

**优雅降级**:
- 模型加载失败时使用随机数据
- 记录详细错误但不中断服务

#### 3.6 stimulation_simulator.py
模拟虚拟脑刺激效果。

**核心类**: `StimulationSimulator`, `StimulationConfig`

**刺激模式**:
- Sine wave（正弦波）
- Square wave（方波）
- Ramp（斜坡）
- Pulse（脉冲）

**参数**:
- `target_regions`: 目标脑区列表
- `amplitude`: 刺激幅度（0.01-10.0）
- `frequency`: 频率（0.1-100 Hz）
- `duration`: 持续时间（秒）
- `pattern`: 刺激模式

#### 3.7 workflow_manager.py
自动化工作流管理。

**核心类**: `WorkflowManager`

**功能**:
- 端到端数据处理
- 自动cache检测和转换
- 批量处理支持
- 进度追踪

### 4. Unity前端组件

Unity前端由7个C#脚本组成：

#### 4.1 核心脚本

**BrainVisualization.cs** (主可视化)
- 加载JSON状态文件
- 管理脑区GameObject
- 颜色映射（活动强度→颜色）
- 时间序列动画
- 点击交互

**BrainDataStructures.cs** (数据结构)
- 定义JSON反序列化类
- BrainStateData
- RegionData
- ConnectionData
- ActivityData

**BrainConfigLoader.cs** (配置加载)
- 读取unity_config.json
- 应用可视化参数
- 动态配置更新

#### 4.2 通信脚本

**WebSocketClient.cs** (原版，兼容性)
- 基础WebSocket客户端
- 事件系统
- 消息队列

**WebSocketClientImproved.cs** (v2.5新增，推荐) ⭐
- 使用UnityWebRequest实现HTTP通信
- 指数退避重连机制
- 请求ID追踪
- 完整输入验证
- 连接状态管理
- 回调支持
- 详细日志

**主要改进**:
```csharp
// 旧版（stub）
wsClient.SendRequest("predict", parameters);

// 新版（完整实现+回调）
wsClient.RequestPrediction(10, (response) => {
    if (response["success"].Value<bool>()) {
        Debug.Log("预测成功");
    } else {
        Debug.LogError($"错误: {response["message"]}");
    }
});
```

#### 4.3 辅助脚本

**StimulationInput.cs** (刺激输入UI)
- UI按钮和输入框
- 参数验证
- 发送刺激请求

**TimelineController.cs** (时间轴控制)
- 播放/暂停
- 帧跳转
- 速度控制

**CacheToJsonConverter.cs** (自动转换) ⭐
- UI一键转换cache文件
- 进度显示
- 错误处理
- 自动检测cache文件

### 5. 服务器启动（运行时）

**脚本**: `unity_startup.py` (v2.5增强)

**功能**:
- 依赖验证（torch, websockets, nibabel）
- 智能模型文件搜索（不再硬编码路径）
- 自动创建缺失目录
- 启动WebSocket服务器
- 健康检查

**改进** (v2.5):
- ✅ 自动搜索模型文件（多个位置）
- ✅ 更好的错误处理
- ✅ 详细的启动日志
- ✅ 模型信息显示（大小、epoch等）
- ✅ 目录自动创建

**使用方法**:
```bash
# 指定模型
python unity_startup.py --model results/model.pt

# 演示模式（无模型）
python unity_startup.py --demo

# 自定义设置
python unity_startup.py --model model.pt \
    --output unity_project \
    --port 8080 \
    --host 0.0.0.0
```

**模型搜索路径** (自动):
1. `--model` 指定的路径
2. `results/hetero_gnn_trained.pt`
3. `test_file3/sub-*/results/hetero_gnn_trained.pt`
4. 所有子目录中的 `results/*.pt`

## 数据流程

### 离线模式（推荐用于稳定展示）

```
FreeSurfer数据
    ↓ [setup_unity_project.py]
OBJ 3D模型
    ↓
预处理的脑数据 (.pkl/.npy)
    ↓ [brain_state_exporter.py]
JSON状态文件
    ↓ [复制到StreamingAssets]
Unity加载
    ↓ [BrainVisualization.cs]
3D可视化
```

### 在线模式（用于实时预测和交互）

```
Unity前端 (WebSocketClientImproved)
    ↓ [HTTP POST请求]
Python后端 (BrainVisualizationServer)
    ↓ [输入验证]
ModelServer预测
    ↓ [生成JSON]
JSON响应 + 自动保存
    ↓ [HTTP响应]
Unity更新可视化
```

### Cache自动转换流程（新增v2.5）

```
用户操作: 放cache文件到cache/目录
    ↓
Unity UI: 点击"转换Cache到JSON"按钮
    ↓
CacheToJsonConverter发送HTTP请求
    ↓
BrainVisualizationServer.handle_convert_cache()
    ↓ [扫描cache文件]
    ↓ [逐个加载并转换]
    ↓ [保存JSON到model_output/]
HTTP响应 (转换统计)
    ↓
Unity显示结果
```

## 文件格式规范

### JSON状态文件 (v2.0)

**位置**: `unity_project/brain_data/model_output/brain_state_*.json`

**必需字段**:
- `version`: "2.0"
- `timestamp`: ISO 8601格式
- `brain_state.regions[]`: 脑区数组

**可选字段**:
- `subject_id`: 被试ID
- `brain_state.connections[]`: 连接数据
- `stimulation`: 刺激信息
- `metadata`: 额外元数据

### OBJ模型文件

**位置**: `unity_project/Unity_Assets/Models/region_*.obj`

**命名**: `region_0001.obj` 到 `region_0200.obj`

**格式**: 标准Wavefront OBJ
- `v` 顶点坐标
- `vn` 法线
- `f` 面（三角形）

### 配置文件 (unity_config.json)

**位置**: `unity_project/unity_config.json`

**示例**:
```json
{
  "version": "2.5",
  "atlas": "Schaefer200",
  "visualization": {
    "region_scale": 1.0,
    "activity_threshold": 0.3,
    "fps": 10,
    "use_obj_models": true,
    "show_connections": false,
    "connection_threshold": 0.5
  },
  "colors": {
    "low_activity": {"r": 0, "g": 0, "b": 255},
    "high_activity": {"r": 255, "g": 0, "b": 0},
    "prediction": {"r": 255, "g": 255, "b": 0}
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
- **websockets**: WebSocket服务器
- **torch**: 模型推理（可选）

### Unity要求
- **Unity版本**: 2019.1+ (推荐2020/2021 LTS)
- **Newtonsoft.Json**: JSON解析（必需）
- **OBJ加载器**: TriLib或Runtime OBJ Importer（可选）

## 设计原则

1. **简单性**: 核心功能仅需JSON文件即可运行
2. **独立性**: Unity部分不依赖训练/预测（可离线使用）
3. **稳定性**: 优先使用静态JSON，避免实时计算的复杂性
4. **可扩展性**: 支持可选的实时模式
5. **容错性**: 优雅降级，失败不中断（v2.5新增）
6. **可维护性**: 清晰的日志和错误消息（v2.5增强）

## 性能考虑

### 优化建议
- 对于200+脑区，使用LOD（细节层次）系统
- 启用GPU Instancing减少DrawCall
- 限制帧率（推荐10-15 FPS）
- 使用活动阈值过滤低活动脑区
- OBJ模型多边形简化

### 性能指标
- **加载时间**: < 5秒（200个脑区）
- **帧率**: 30+ FPS（中等配置）
- **内存占用**: < 500MB
- **网络延迟**: < 100ms（本地）

### 性能监控

```csharp
// 在BrainVisualization中添加
void Update()
{
    if (showPerformanceStats)
    {
        Debug.Log($"FPS: {1.0f / Time.deltaTime:F1}");
        Debug.Log($"Active Regions: {activeRegionCount}");
        Debug.Log($"Draw Calls: {UnityStats.drawCalls}");
    }
}
```

## 安全考虑

- JSON文件应放在StreamingAssets，不包含敏感信息
- WebSocket连接默认使用本地主机（localhost）
- 生产环境不应暴露后端服务端口
- 输入验证防止恶意请求（v2.5新增）
- 限制请求频率（建议添加rate limiting）

## 扩展方向

### 近期计划
- [ ] 真正的WebSocket支持（集成第三方库）
- [ ] 请求缓存和去重
- [ ] Rate limiting（请求频率限制）
- [ ] 完整的API文档（OpenAPI规范）
- [ ] 自动化测试

### 中期计划
- [ ] 支持更多脑图谱（AAL, Destrieux, DKT等）
- [ ] VR/AR支持
- [ ] 多被试对比可视化
- [ ] 实时EEG信号流

### 长期愿景
- [ ] WebGL在线演示
- [ ] Unity Package Manager发布
- [ ] 移动平台支持（iOS/Android）
- [ ] 云端渲染

## v2.5变更总结

### 主要改进
1. ✅ WebSocket客户端完整实现（HTTP/REST）
2. ✅ 服务器端全面输入验证
3. ✅ 智能模型文件搜索
4. ✅ Unity包自动安装工具
5. ✅ 统一的错误响应格式
6. ✅ 请求ID追踪机制
7. ✅ 连接状态管理
8. ✅ 指数退避重连
9. ✅ 详细日志系统
10. ✅ 优雅降级支持

### Bug修复
- 修复WebSocketClient.cs仅为stub的问题
- 修复服务器端缺少输入验证
- 修复硬编码路径问题
- 修复错误消息不一致

### API变更
- 所有响应包含 `success` 字段
- 统一错误格式
- 添加可选的 `request_id`
- 参数验证和范围限制

---

**版本**: 2.5  
**更新日期**: 2024-02-14  
**作者**: TwinBrain Team

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
