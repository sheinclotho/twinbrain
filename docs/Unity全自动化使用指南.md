# Unity 全自动化工作流程使用指南

## 概述

本系统实现了完全自动化的Unity可视化工作流程，包括：
1. 一键式文件夹和脚本生成
2. FreeSurfer文件自动处理
3. 后端模型加载和推理服务
4. Unity中的交互式按钮控制
5. 虚拟刺激的完整闭环流程

## 🚀 快速开始

### 步骤1: 一键式设置

在项目根目录运行：

```bash
python setup_unity_workflow.py --auto-setup
```

这会自动创建：
```
unity_project/
├── freesurfer_files/          # FreeSurfer文件存放处
├── brain_data/                # 数据文件夹
│   ├── original/              # 原始数据
│   ├── cache/                 # 缓存
│   └── model_output/          # 模型输出（Unity读取）
├── Unity_Assets/              # Unity资源
│   ├── Scripts/               # C#脚本
│   └── unity_config.json      # 配置
├── start_backend_server.py    # 启动脚本
└── README_WORKFLOW.md         # 详细文档
```

### 步骤2: （可选）添加FreeSurfer文件

如果您有FreeSurfer文件，放入 `unity_project/freesurfer_files/`：
```bash
cp /path/to/lh.pial unity_project/freesurfer_files/
cp /path/to/rh.pial unity_project/freesurfer_files/
cp /path/to/lh.*.annot unity_project/freesurfer_files/
cp /path/to/rh.*.annot unity_project/freesurfer_files/
```

然后重新运行设置以生成真实的大脑表面：
```bash
python setup_unity_workflow.py --auto-setup
```

### 步骤3: 启动后端服务器

```bash
# Windows
cd unity_project
start_backend_server.bat

# Linux/Mac
cd unity_project
./start_backend_server.sh

# 或直接使用Python
python unity_project/start_backend_server.py
```

服务器会显示：
```
✓ 服务器端口: 8765
✓ Unity连接地址: ws://localhost:8765
```

### 步骤4: Unity中设置

1. **导入资源**
   - 将 `unity_project/Unity_Assets/` 内容复制到Unity项目的 `Assets/` 目录

2. **创建GameObject**
   - 创建空GameObject，命名 "BrainManager"
   - 添加组件：
     - BrainVisualization
     - BrainDataLoader
     - AnimationController
     - StimulationInput
     - ModelInterface

3. **配置路径**
   - BrainDataLoader → Data Folder Path: `../unity_project/brain_data/model_output`
   - ModelInterface → Server URL: `ws://localhost:8765`
   - ModelInterface → Output Folder: `../unity_project/brain_data/model_output`

4. **创建UI按钮**
   - 数据加载按钮：绑定 `BrainDataLoader.OnLoadDataClicked()`
   - 刷新按钮：绑定 `BrainDataLoader.OnRefreshClicked()`
   - 播放按钮：绑定 `AnimationController.OnPlay()`
   - 暂停按钮：绑定 `AnimationController.OnPause()`
   - 停止按钮：绑定 `AnimationController.OnStop()`
   - 应用刺激按钮：绑定 `StimulationInput.OnApplyStimulation()`

## 📊 完整工作流程

### 流程图

```
┌─────────────────┐
│ FreeSurfer文件  │ ─一次性─→ [Unity 3D结构生成]
└─────────────────┘                 ↓
                            Unity前端准备完成
                                    ↓
┌──────────────┐                    ↓
│ 原始fMRI数据 │ → [预处理] → [训练模型]
└──────────────┘                    ↓
                            [加载训练模型]
                                    ↓
        ┌──────────── Unity中点击 ──────────────┐
        │                                       │
   [加载数据]                              [虚拟刺激]
        ↓                                       ↓
   读取JSON  ←── model_output/ ←── [后端模型预测]
        ↓                                       ↑
   [动画显示]                            发送刺激参数
        ↓
   时间轴演化
```

### 详细步骤说明

#### 场景1：查看现有数据的动画

1. 准备数据文件（JSON格式）到 `brain_data/model_output/`
2. Unity中点击"刷新"按钮扫描文件
3. 点击"加载数据"读取JSON
4. 点击"播放"开始动画
5. 使用时间轴滑块控制进度
6. 脑区颜色表示活动强度

#### 场景2：虚拟刺激实验

1. 确保后端服务器正在运行
2. Unity中点击鼠标选择目标脑区（可多选）
3. 设置刺激参数：
   - 强度（amplitude）: 0-1
   - 模式（pattern）: sine/pulse/ramp/constant
   - 频率（frequency）: Hz
4. 点击"应用刺激"
5. 后端模型计算响应（自动保存到 model_output/）
6. 点击"刷新" → "加载数据"
7. 播放查看刺激响应动画

#### 场景3：请求未来预测

1. 确保后端服务器运行且模型已加载
2. Unity中点击"请求预测"按钮
3. 后端生成50步预测
4. 自动保存到 model_output/
5. 点击"刷新" → "加载数据"
6. 播放预测序列动画

## 🎯 Unity脚本功能详解

### BrainDataLoader.cs
**功能**: 数据加载和管理
- `OnLoadDataClicked()`: 加载下一个JSON文件
- `OnRefreshClicked()`: 重新扫描数据文件夹
- `LoadAllAsSequence()`: 加载所有文件作为时间序列

### AnimationController.cs
**功能**: 动画播放控制
- `OnPlay()`: 开始播放动画
- `OnPause()`: 暂停动画
- `OnStop()`: 停止并重置
- 时间轴滑块：手动跳转到任意时间点

### StimulationInput.cs
**功能**: 虚拟刺激输入
- 点击脑区进行选择/取消选择
- 设置刺激参数（强度、模式、频率）
- `OnApplyStimulation()`: 发送刺激到后端

### ModelInterface.cs
**功能**: 后端通信
- 自动连接WebSocket服务器
- `SendStimulation()`: 发送虚拟刺激
- `RequestPrediction()`: 请求模型预测
- 自动保存响应到数据文件夹

## 🔧 后端模型服务

### ModelServer 功能

1. **模型加载**
   ```python
   from unity_integration import ModelServer
   
   server = ModelServer(
       model_path="results/hetero_gnn_trained.pt",
       output_dir="unity_project/brain_data/model_output"
   )
   ```

2. **未来预测**
   ```python
   predictions = server.predict_future(
       n_steps=50,
       subject_id="prediction"
   )
   # 自动保存到 model_output/prediction_t0000_*.json, ...
   ```

3. **刺激模拟**
   ```python
   responses = server.simulate_stimulation(
       target_regions=[10, 15, 20],
       amplitude=0.5,
       pattern="sine",
       frequency=10.0,
       duration=50
   )
   # 自动保存到 model_output/stimulation_t0000_*.json, ...
   ```

### 支持的刺激模式

- **sine**: 正弦波 `amplitude * sin(2π * frequency * t)`
- **pulse**: 脉冲 （周期性短脉冲）
- **ramp**: 渐变 （从0到amplitude线性增长）
- **constant**: 恒定 （持续的固定强度）

## 📁 数据格式

### JSON脑状态格式

```json
{
  "subject_id": "prediction",
  "time_point": 0,
  "time_second": 0.0,
  "timestamp": "2026-02-04T21:00:00",
  "n_regions": 200,
  "regions": [
    {
      "id": 0,
      "label": "Region_000",
      "activity": 0.65,
      "raw_activity": 0.31,
      "network": "Visual",
      "hemisphere": "left"
    }
  ],
  "stimulation": {  // 可选
    "target_regions": [10, 15],
    "amplitude": 0.5,
    "pattern": "sine"
  }
}
```

### 文件命名规则

- 预测: `prediction_t0000_20260204_210000.json`
- 刺激: `stimulation_t0000_20260204_210000.json`
- 元数据: `prediction_metadata_20260204_210000.json`

## 🎨 可视化效果

### 颜色映射
- **蓝色** (activity < 0.3): 低活动
- **黄色** (0.3 ≤ activity < 0.7): 中等活动
- **红色** (activity ≥ 0.7): 高活动

### 大小映射
- 脑区球体大小随活动强度变化
- `size = 1.0 + activity * 2.0`

### 时间轴
- 水平滑块显示当前时间点
- 文本显示当前时间（秒）和帧号
- 拖动滑块实时跳转

## ⚙️ 高级配置

### 修改模型路径
编辑 `unity_project/start_backend_server.py`:
```python
MODEL_PATH = "/path/to/your/model.pt"
```

### 修改输出目录
```python
OUTPUT_DIR = "/path/to/output"
```

### 修改服务器端口
```python
PORT = 8765  # 改为其他端口
```
同时更新Unity中的 ModelInterface → Server URL。

### 自定义数据处理
编辑 `Unity_Assets/Scripts/BrainDataLoader.cs` 中的 `BrainStateData` 结构。

## 🐛 故障排除

### 问题: Unity无法连接服务器
**检查**:
1. 后端服务器是否运行？`python start_backend_server.py`
2. 端口是否正确？默认8765
3. 防火墙是否允许？

### 问题: 找不到数据文件
**检查**:
1. 路径是否正确？使用相对路径 `../unity_project/brain_data/model_output`
2. JSON文件是否存在？检查文件夹
3. 文件权限是否正确？

### 问题: 动画不播放
**检查**:
1. 数据是否已加载？查看Unity Console
2. 时间轴是否设置？`SetTotalFrames()`
3. FPS设置是否合理？默认10

### 问题: 后端模型加载失败
**检查**:
1. 模型文件是否存在？
2. PyTorch版本是否兼容？
3. 查看错误日志

## 📚 相关文档

- [README_CN.md](README_CN.md) - 项目主文档
- [Unity工作流说明](docs/Unity工作流说明.md) - Unity集成详细说明
- [FreeSurfer使用指南](docs/FreeSurfer使用指南.md) - FreeSurfer文件处理
- [正确工作流程说明](正确工作流程说明.md) - 工作流程理解

## 📞 技术支持

如有问题：
1. 查看 `unity_project/README_WORKFLOW.md` 详细文档
2. 检查Unity Console中的错误信息
3. 查看后端服务器日志
4. 提交GitHub Issue

## 🎉 总结

这套自动化系统实现了：

✅ **一键式设置** - 自动创建所有必需的文件夹和脚本  
✅ **FreeSurfer集成** - 自动处理FreeSurfer文件生成真实大脑结构  
✅ **模型加载** - 后端自动加载训练好的PyTorch模型  
✅ **虚拟刺激** - Unity中交互选择脑区并应用刺激  
✅ **实时预测** - 后端模型实时计算响应  
✅ **自动保存** - 所有输出自动保存为JSON格式  
✅ **动画播放** - Unity中流畅播放时间序列动画  
✅ **完整闭环** - 从输入刺激到查看动画的完整流程

现在您可以：
1. 运行 `python setup_unity_workflow.py --auto-setup` 开始
2. 启动后端服务器
3. 在Unity中享受交互式大脑可视化！

---

**最后更新**: 2026-02-04  
**版本**: v2.4  
**作者**: TwinBrain Development Team
