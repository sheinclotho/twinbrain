# TwinBrain Unity 完整使用指南 v3.0

> **最后更新**: 2024-02-15  
> **兼容版本**: Unity 2019.1+ | Python 3.8+ | TwinBrain 2.5+

---

## 📋 目录

1. [系统要求](#系统要求)
2. [快速开始](#快速开始)
3. [详细安装流程](#详细安装流程)
4. [数据准备](#数据准备)
5. [Unity场景配置](#unity场景配置)
6. [使用功能](#使用功能)
7. [常见问题](#常见问题)

---

## 系统要求

### 必需组件
- **Unity**: 2019.1+ (推荐 2020 LTS 或 2021 LTS)
- **Python**: 3.8+
- **操作系统**: Windows 10+, macOS 10.14+, 或 Linux (Ubuntu 18.04+)
- **硬件**: 
  - CPU: 4核心以上
  - RAM: 8GB以上 (推荐16GB)
  - GPU: 支持DirectX 11或OpenGL 4.5 (可选，用于加速)

### 可选组件
- FreeSurfer 7.0+ (用于生成真实3D脑区模型)
- 训练好的TwinBrain模型文件 (用于实时预测)

---

## 快速开始

### 三步完成基础设置

```bash
# 1. 克隆并进入TwinBrain项目
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 2. 生成Unity资源文件
python setup_unity_project.py --auto-setup

# 3. 在Unity Hub中创建新的3D项目
# 项目名称: TwinBrainDemo
# 位置: 任意位置

# 4. 安装TwinBrain到Unity项目
python unity_package_installer.py --unity-project /path/to/TwinBrainDemo
```

安装完成后，在Unity中完成手动配置步骤（详见下文）。

---

## 详细安装流程

### 阶段1: 生成Unity资源文件

#### 1.1 基础设置（无FreeSurfer数据）

```bash
cd twinbrain
python setup_unity_project.py --auto-setup
```

这将创建以下目录结构：

```
unity_project/
├── freesurfer_files/         # FreeSurfer数据存放位置（可选）
├── brain_data/
│   ├── cache/                # 预处理缓存文件 (.pt/.pth PyTorch格式)
│   ├── model_output/         # JSON状态文件
│   └── original/             # 原始数据
├── Unity_Assets/
│   ├── Scripts/              # Unity C#脚本
│   ├── obj/                  # 3D脑区模型（如果使用FreeSurfer）
│   └── unity_config.json     # 配置文件
```

**重要说明**: 
- Cache文件格式为 `.pt` 或 `.pth` (PyTorch格式)，**不是** `.pkl` (pickle格式)
- 这些是训练过程自动生成的PyTorch tensor缓存

#### 1.2 使用FreeSurfer生成真实3D脑区模型

如果您有FreeSurfer处理的脑表面数据：

```bash
# 将FreeSurfer文件准备好
# 需要的文件:
# - lh.pial (左半球表面)
# - rh.pial (右半球表面)
# - lh.Schaefer2018_200Parcels_7Networks_order.annot (左半球标注)
# - rh.Schaefer2018_200Parcels_7Networks_order.annot (右半球标注)

# 运行设置脚本，指定FreeSurfer目录
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer/files
```

这将自动：
1. 读取FreeSurfer表面数据
2. 根据图谱标注分割脑区
3. **为每个脑区生成独立的OBJ文件** (region_0001.obj, region_0002.obj, ...)
4. 生成配置文件和材质定义
5. 将所有文件输出到 `unity_project/Unity_Assets/obj/`

**生成的OBJ文件数量**: 通常为200个（基于Schaefer200图谱）

### 阶段2: 安装到Unity项目

#### 2.1 创建Unity项目

1. 打开Unity Hub
2. 点击"新建项目"
3. 选择"3D"模板
4. 项目名称: `TwinBrainDemo` (或任意名称)
5. 选择保存位置
6. 点击"创建项目"

#### 2.2 使用自动安装工具

```bash
# 安装TwinBrain包到Unity项目
python unity_package_installer.py --unity-project /path/to/TwinBrainDemo

# 如果需要同时复制数据文件
python unity_package_installer.py \
    --unity-project /path/to/TwinBrainDemo \
    --data-dir unity_project
```

**自动完成的操作**:
- ✅ 复制所有C#脚本到 `Assets/TwinBrain/Scripts/`
- ✅ 创建Assembly Definition文件
- ✅ 创建StreamingAssets目录结构:
  - `StreamingAssets/brain_states/` - JSON数据文件
  - `StreamingAssets/config/` - 配置文件
  - `StreamingAssets/OBJ/` - 3D模型文件
- ✅ 配置Newtonsoft.Json依赖（自动下载）
- ✅ 生成使用指南文档

**仍需手动完成的操作** (Unity编辑器限制):
- 🔧 创建场景对象和添加组件
- 🔧 导入OBJ模型并创建预制体
- 🔧 配置组件参数

#### 2.3 导入OBJ模型到Unity（如果使用FreeSurfer）

**重要**: OBJ文件已生成到 `unity_project/Unity_Assets/obj/`，现在需要导入到Unity项目。

##### 方法A: 批量导入（推荐）

```bash
# 将生成的OBJ文件复制到Unity项目
cp -r unity_project/Unity_Assets/obj/* \
    /path/to/TwinBrainDemo/Assets/StreamingAssets/OBJ/
```

然后在Unity中：
1. Unity会自动检测到新的OBJ文件
2. 等待导入完成（200个OBJ文件可能需要几分钟）
3. 所有OBJ文件将出现在 `Assets/StreamingAssets/OBJ/`

##### 方法B: 使用Unity编辑器导入

1. 在Unity编辑器中: `Assets` > `Import New Asset...`
2. 选择 `unity_project/Unity_Assets/obj/` 目录
3. 选择所有OBJ文件（可以使用Ctrl+A全选）
4. 点击"Import"

**注意**: Unity会为每个OBJ创建材质（Material）和预制体（Prefab），这是正常的。

---

## 数据准备

### 准备Brain State JSON文件

Unity可视化需要JSON格式的大脑状态数据。有三种方式准备这些数据：

#### 方式A: 从Cache转换（推荐）

如果您有预处理的cache文件：

```bash
# 确保cache文件格式正确
# 正确格式: .pt 或 .pth (PyTorch tensor)
# 例如: eeg_data.pt, hetero_graphs.pt, fmri_cache.pt

# 方法1: 使用命令行工具
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output

# 方法2: 使用Unity内的UI转换（需要先配置，见下文）
# 1. 启动后端服务器
python unity_startup.py --demo

# 2. 在Unity中点击"转换Cache"按钮
```

**重要**: Cache文件必须是PyTorch格式 (`.pt` 或 `.pth`)，不是pickle格式 (`.pkl`)。

#### 方式B: 从模型实时生成

如果您有训练好的模型：

```bash
# 启动后端服务器（会自动生成JSON）
python unity_startup.py --model results/hetero_gnn_trained.pt \
    --output unity_project

# JSON文件会自动生成到 unity_project/brain_data/model_output/
```

#### 方式C: 使用示例数据

```bash
# 生成示例数据用于测试
python -c "
from unity_integration import generate_example_data
generate_example_data('unity_project/brain_data/model_output', n_regions=200, n_timepoints=100)
"
```

### 复制数据到Unity项目

```bash
# 复制JSON文件
cp unity_project/brain_data/model_output/*.json \
    /path/to/TwinBrainDemo/Assets/StreamingAssets/brain_states/

# 复制配置文件
cp unity_project/Unity_Assets/unity_config.json \
    /path/to/TwinBrainDemo/Assets/StreamingAssets/config/
```

---

## Unity场景配置

以下步骤必须在Unity编辑器中手动完成（无法通过脚本自动化）。

### 步骤1: 等待依赖包下载

1. 在Unity Hub中打开项目
2. Unity会自动检测到manifest.json的更改
3. 自动开始下载Newtonsoft.Json包
4. 等待进度条完成（通常1-2分钟）
5. 如果Console中出现编译错误，请等待下载完成后会自动解决

**验证安装**:
- `Window` > `Package Manager`
- 搜索 "Newtonsoft"
- 应该看到 "Newtonsoft Json" 已安装

### 步骤2: 创建BrainManager对象

1. 在Hierarchy窗口右键
2. 选择 `Create Empty`
3. 命名为 `BrainManager`
4. 确保Transform位置为 (0, 0, 0)

### 步骤3: 添加核心组件

选中BrainManager对象，点击 `Add Component` 添加以下组件：

#### 组件1: BrainVisualization (必需)

主可视化组件，负责加载和显示大脑活动。

**配置参数**:
- `Json Path`: `brain_states` (相对于StreamingAssets的路径)
- `Region Prefab`: [待创建，见步骤4]
- `Load Sequence`: ✓ (如果有多个JSON文件)
- `Region Scale`: 1.0
- `Activity Threshold`: 0.1
- `Show Connections`: false (可选)
- `Animation FPS`: 10

#### 组件2: WebSocketClientImproved (可选，用于实时通信)

用于与后端服务器通信，实现实时预测和刺激模拟。

**配置参数**:
- `Server URL`: `http://localhost:8765`
- `Auto Connect`: ✓
- `Auto Reconnect`: ✓
- `Reconnect Delay`: 2.0
- `Max Reconnect Attempts`: 10

#### 组件3: BrainConfigLoader (可选，用于配置加载)

从JSON文件加载配置。

**配置参数**:
- `Config Path`: `config/unity_config.json`
- `Auto Load`: ✓

### 步骤4: 创建脑区预制体

您需要创建一个预制体来表示脑区。根据是否有OBJ模型，有两种方式：

#### 方式A: 使用OBJ模型（如果有FreeSurfer数据）

**如果已导入200个OBJ文件**:

由于有200个独立的脑区OBJ，您需要决定使用方式：

**方案1: 使用单个OBJ作为代表**（推荐用于快速测试）:
1. 在Project窗口，导航到 `Assets/StreamingAssets/OBJ/`
2. 选择任意一个OBJ文件（如 `region_0001.obj`）
3. 拖拽到Hierarchy创建实例
4. 调整Transform Scale（如 `(0.01, 0.01, 0.01)`）
5. 添加Material（可选）
6. 拖拽回Project窗口创建Prefab，命名 `BrainRegion`
7. 删除Hierarchy中的实例
8. 将Prefab赋值给BrainVisualization的 `Region Prefab`

**方案2: 动态加载所有OBJ**（推荐用于生产）:
1. 不创建预制体
2. 在BrainVisualization中设置 `Use Obj Models`: ✓
3. 设置 `Obj Directory`: `OBJ`
4. 运行时会自动加载所有region_XXXX.obj文件

#### 方式B: 使用简单球体（无FreeSurfer数据）

1. `Hierarchy` > 右键 > `3D Object` > `Sphere`
2. 命名为 `BrainRegion`
3. 设置Transform:
   - Position: (0, 0, 0)
   - Rotation: (0, 0, 0)
   - Scale: (0.5, 0.5, 0.5)
4. 添加Material:
   - 在Project窗口右键 > `Create` > `Material`
   - 命名为 `RegionMaterial`
   - 拖拽到Sphere
5. 拖拽Sphere到Project窗口创建Prefab
6. 删除Hierarchy中的Sphere
7. 将Prefab赋值给BrainVisualization的 `Region Prefab`

### 步骤5: 设置Camera

1. 选中Main Camera
2. 设置Position: (0, 5, -10)
3. 设置Rotation: (30, 0, 0)
4. 添加Camera控制脚本（可选）

### 步骤6: 添加Cache转换UI（可选，但推荐）

如果您想在Unity中直接转换cache文件：

1. **创建Canvas**:
   - `Hierarchy` > `UI` > `Canvas`
   - Canvas Scaler设置为 `Scale With Screen Size`

2. **创建转换按钮**:
   - 右键Canvas > `UI` > `Button`
   - 命名: `ConvertCacheButton`
   - Text: "转换Cache到JSON"
   - 位置: 屏幕左下角

3. **创建状态文本**:
   - 右键Canvas > `UI` > `Text`
   - 命名: `StatusText`
   - 位置: 按钮上方

4. **创建进度条**（可选）:
   - 右键Canvas > `UI` > `Slider`
   - 命名: `ProgressSlider`
   - 位置: 状态文本下方

5. **添加CacheToJsonConverter组件**:
   - 选中Canvas或创建新的Empty GameObject
   - `Add Component` > `CacheToJsonConverter`
   - 配置:
     - `Convert Button`: 拖入ConvertCacheButton
     - `Status Text`: 拖入StatusText
     - `Progress Slider`: 拖入ProgressSlider
     - `Cache Directory`: `../unity_project/brain_data/cache` (相对路径)
     - `Output Directory`: `../unity_project/brain_data/model_output`
     - `Backend Url`: `http://localhost:8765`

### 步骤7: 保存场景

1. `File` > `Save Scene As...`
2. 命名为 `BrainVisualization`
3. 保存到 `Assets/Scenes/`

---

## 使用功能

### 基础可视化

1. 确保所有配置步骤已完成
2. 点击Unity编辑器顶部的 **Play** 按钮
3. 应该看到脑区出现并根据数据显示活动
4. 颜色变化表示活动强度

**键盘控制** (在BrainVisualization脚本中定义):
- `Space`: 播放/暂停动画
- `R`: 重新加载数据
- `←`: 上一帧
- `→`: 下一帧
- `+/-`: 调整播放速度

### 实时预测（需要后端服务器）

#### 启动后端服务器

```bash
# 方式1: 使用训练好的模型
python unity_startup.py --model results/hetero_gnn_trained.pt

# 方式2: 演示模式（无模型）
python unity_startup.py --demo

# 方式3: 自定义配置
python unity_startup.py \
    --model results/hetero_gnn_trained.pt \
    --output unity_project \
    --port 8765
```

#### 在Unity中使用

服务器启动后，WebSocketClientImproved会自动连接。在C#代码中使用：

```csharp
using TwinBrain;

public class MyBrainController : MonoBehaviour
{
    private WebSocketClientImproved wsClient;
    
    void Start()
    {
        wsClient = GetComponent<WebSocketClientImproved>();
        
        // 订阅事件
        wsClient.OnConnected += HandleConnected;
        wsClient.OnBrainStateReceived += HandleBrainState;
        wsClient.OnError += HandleError;
    }
    
    void HandleConnected()
    {
        Debug.Log("✓ 已连接到TwinBrain后端");
        RequestBrainState();
    }
    
    void HandleBrainState(BrainStateData state)
    {
        Debug.Log($"收到大脑状态: {state.brain_state.regions.Count} 个脑区");
        // 处理大脑状态数据
    }
    
    void HandleError(string error)
    {
        Debug.LogError($"WebSocket错误: {error}");
    }
    
    // 请求当前大脑状态
    public void RequestBrainState()
    {
        wsClient.GetBrainState((response) => {
            Debug.Log("获取大脑状态成功");
        });
    }
    
    // 请求预测
    public void RequestPrediction(int steps)
    {
        wsClient.RequestPrediction(steps, (response) => {
            if (response["success"].Value<bool>())
            {
                Debug.Log($"预测{steps}步成功");
            }
        });
    }
    
    // 模拟刺激
    public void SimulateStimulation()
    {
        int[] targetRegions = { 10, 20, 30 };
        float amplitude = 0.5f;
        string waveform = "sine";  // "sine", "square", "ramp"
        
        wsClient.SimulateStimulation(targetRegions, amplitude, waveform, (response) => {
            Debug.Log("刺激模拟完成");
        });
    }
}
```

### Cache文件转换

如果您配置了转换UI：

1. 将cache文件（`.pt`或`.pth`格式）放入 `unity_project/brain_data/cache/`
2. 确保后端服务器正在运行
3. 在Unity中点击Play
4. 点击"转换Cache到JSON"按钮
5. 等待转换完成
6. JSON文件会自动生成到 `model_output/`

或使用代码调用：

```csharp
void ConvertCacheFiles()
{
    var converter = GetComponent<CacheToJsonConverter>();
    converter.StartConversion();
}
```

---

## 常见问题

### Q: 找不到Newtonsoft.Json类型

**A**: 
1. 确认Package Manager中已安装Newtonsoft.Json:
   - `Window` > `Package Manager`
   - 搜索 "Newtonsoft"
2. 如果没有，手动安装:
   - Package Manager > `+` > `Add package from git URL`
   - 输入: `com.unity.nuget.newtonsoft-json`
3. 重启Unity编辑器

### Q: Cache文件格式错误

**A**: Cache文件必须是PyTorch格式 (`.pt` 或 `.pth`)，不是pickle格式 (`.pkl`)。

**检查文件**:
```bash
# 查找cache文件
ls -la unity_project/brain_data/cache/

# 应该看到类似:
# eeg_data.pt
# hetero_graphs.pt
# fmri_cache.pt
```

**如果是.pkl文件**:
这是错误的格式。cache文件应该由TwinBrain训练过程自动生成，格式为PyTorch tensor。如果您有.pkl文件，需要重新运行预处理或训练流程。

### Q: WebSocket连接失败

**A**: 
1. 检查后端服务器是否运行:
   ```bash
   python unity_startup.py --demo
   ```
2. 检查端口是否正确（默认8765）
3. 检查防火墙设置
4. 查看Unity Console和Python终端的错误信息
5. 确认WebSocketClientImproved的Server URL设置正确

### Q: JSON文件加载失败

**A**: 
1. 确认文件在正确位置:
   ```
   Assets/StreamingAssets/brain_states/
   ```
2. 检查文件格式（必须是有效的JSON）
3. 检查BrainVisualization的Json Path设置（应为相对路径，如 `brain_states`）
4. 查看Unity Console的具体错误信息

### Q: 场景中看不到脑区

**A**: 
1. 检查BrainManager GameObject是否激活
2. 检查BrainVisualization组件是否启用
3. 确认Region Prefab已赋值
4. 确认JSON数据已加载成功（查看Console）
5. 检查Camera位置和方向
6. 检查Region Scale是否太小

### Q: 没有FreeSurfer数据怎么办？

**A**: 使用简单球体代替：
1. 创建Sphere预制体（见步骤4方式B）
2. 在BrainVisualization中:
   - `Use Obj Models`: 不勾选
   - `Region Prefab`: 赋值Sphere预制体

### Q: OBJ文件太多，如何批量导入？

**A**: 
1. 使用命令行复制:
   ```bash
   cp -r unity_project/Unity_Assets/obj/* \
       /path/to/UnityProject/Assets/StreamingAssets/OBJ/
   ```
2. Unity会自动检测并导入
3. 或在BrainVisualization中使用 `Use Obj Models`选项动态加载

### Q: 模型文件未找到

**A**: 
```bash
# 查找模型文件
find . -name "*.pt" -type f

# 常见位置:
# - results/hetero_gnn_trained.pt
# - test_file3/sub-XX/results/hetero_gnn_trained.pt

# 使用找到的路径:
python unity_startup.py --model path/to/found/model.pt

# 或使用演示模式（无需模型）:
python unity_startup.py --demo
```

### Q: 性能很慢/卡顿

**A**: 优化建议：
1. 减少可见脑区数量（设置Activity Threshold）
2. 降低Animation FPS（推荐10-15）
3. 如果使用OBJ模型:
   - 减少模型多边形数
   - 使用LOD (Level of Detail)系统
4. 启用GPU Instancing（在Material中）
5. 限制同时显示的连接线数量

---

## 完整工作流示例

### 示例1: 基础可视化（离线模式）

```bash
# 1. 生成Unity资源（使用FreeSurfer）
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer

# 2. 转换预处理数据为JSON
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output

# 3. 安装到Unity项目
python unity_package_installer.py \
    --unity-project /path/to/TwinBrainDemo \
    --data-dir unity_project

# 4. 复制数据到StreamingAssets
cp unity_project/brain_data/model_output/*.json \
    /path/to/TwinBrainDemo/Assets/StreamingAssets/brain_states/
cp unity_project/Unity_Assets/obj/*.obj \
    /path/to/TwinBrainDemo/Assets/StreamingAssets/OBJ/

# 5. 在Unity中完成场景配置并运行
```

### 示例2: 实时预测（在线模式）

```bash
# 1. 准备Unity项目（同上）
python setup_unity_project.py --auto-setup
python unity_package_installer.py --unity-project /path/to/TwinBrainDemo

# 2. 启动后端服务器
python unity_startup.py --model results/hetero_gnn_trained.pt

# 3. 在Unity中运行场景
# - WebSocketClientImproved自动连接
# - 通过代码请求预测和模拟刺激
```

### 示例3: 使用Cache转换UI

```bash
# 1. 设置Unity项目
python setup_unity_project.py --auto-setup
python unity_package_installer.py --unity-project /path/to/TwinBrainDemo

# 2. 将cache文件放入指定目录
cp path/to/eeg_data.pt unity_project/brain_data/cache/
cp path/to/hetero_graphs.pt unity_project/brain_data/cache/

# 3. 启动后端服务器
python unity_startup.py --demo

# 4. 在Unity中配置Cache转换UI（见步骤6）

# 5. 运行Unity，点击"转换Cache"按钮
```

---

## 技术细节

### 数据格式说明

#### Cache文件格式

**正确格式**: PyTorch tensor文件 (`.pt` 或 `.pth`)

```python
# 生成cache文件示例（在预处理或训练时）
import torch

# EEG数据cache
eeg_data = {
    'data': torch.tensor(...),  # shape: [n_samples, n_channels, n_timepoints]
    'labels': torch.tensor(...),
    'metadata': {...}
}
torch.save(eeg_data, 'cache/eeg_data.pt')

# 异构图cache
hetero_graphs = [graph1, graph2, ...]  # PyTorch Geometric HeteroData objects
torch.save(hetero_graphs, 'cache/hetero_graphs.pt')
```

**错误格式**: Pickle文件 (`.pkl`)
- TwinBrain不使用pickle格式存储cache
- 如果您的文档中看到.pkl，这是文档错误，应为.pt

#### JSON文件格式

```json
{
  "timestamp": "2024-02-15T10:30:00",
  "n_regions": 200,
  "brain_state": {
    "regions": [
      {
        "id": 1,
        "name": "lh_Default_PFC_1",
        "activity": 0.75,
        "position": [10.5, 20.3, 5.2]
      },
      ...
    ]
  }
}
```

### 目录结构详解

```
TwinBrainDemo/                          # Unity项目根目录
├── Assets/
│   ├── TwinBrain/
│   │   ├── Scripts/                    # TwinBrain C#脚本
│   │   │   ├── BrainVisualization.cs
│   │   │   ├── WebSocketClientImproved.cs
│   │   │   ├── CacheToJsonConverter.cs
│   │   │   └── ...
│   │   ├── TwinBrain.Scripts.asmdef    # Assembly Definition
│   │   ├── package.json                # Package定义
│   │   └── USAGE_GUIDE.md             # 使用指南
│   ├── StreamingAssets/
│   │   ├── brain_states/              # JSON数据文件
│   │   │   ├── brain_state_001.json
│   │   │   ├── brain_state_002.json
│   │   │   └── ...
│   │   ├── config/                    # 配置文件
│   │   │   └── unity_config.json
│   │   └── OBJ/                       # 3D模型文件
│   │       ├── region_0001.obj
│   │       ├── region_0002.obj
│   │       └── ...
│   ├── Scenes/
│   │   └── BrainVisualization.unity
│   └── ...
├── Packages/
│   └── manifest.json                  # 包含Newtonsoft.Json依赖
└── ProjectSettings/
    └── ...
```

---

## 相关文档

- **[Unity架构说明.md](Unity架构说明.md)** - 技术架构和API文档
- **[项目规范说明书.md](项目规范说明书.md)** - 项目完整技术规范
- **[MODEL_FORMAT.md](MODEL_FORMAT.md)** - 模型文件格式说明
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - WebSocket API文档
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 故障排除指南

---

## 更新日志

### v3.0 (2024-02-15)
- ✅ 完全重写指南，修正所有错误
- ✅ 修正cache文件格式说明（.pt/.pth，不是.pkl）
- ✅ 添加OBJ文件批量导入说明
- ✅ 明确区分自动化和手动步骤
- ✅ 添加完整的工作流示例
- ✅ 改进故障排除部分
- ✅ 参考项目规范说明书确保准确性

### v2.5 (2024-02-14)
- 新增WebSocketClientImproved
- 新增unity_package_installer.py
- 改进文档结构

---

**版本**: 3.0  
**作者**: TwinBrain Team  
**许可**: MIT  
**支持**: https://github.com/sheinclotho/twinbrain/issues
