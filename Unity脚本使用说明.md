# TwinBrain Unity Scripts - 使用说明

## 概述

本项目提供了完整的Unity C#脚本集，用于实现TwinBrain的3D可视化和交互功能。所有脚本已针对Unity 2019+进行优化，解决了命名空间冲突和C#版本兼容性问题。

## 脚本列表

### 核心脚本

1. **BrainDataStructures.cs** - 数据结构定义
   - 定义所有JSON数据类
   - 包含预测和刺激数据结构
   - Unity 2019+ 兼容

2. **BrainVisualization.cs** - 主可视化控制器
   - 加载和显示大脑状态数据
   - 支持单个/多个OBJ模型
   - **时间序列支持**: 预加载整个序列并计算全局归一化范围
   - **统一颜色映射**: 所有数值（真实/预测）使用相同颜色刻度
   - 点击交互功能
   - 序列动画播放和帧控制

3. **TimelineController.cs** - 时间轴控制器（新增）
   - **进度条滑块**: 拖动到任意时间点
   - **播放/暂停控制**: 控制动画播放
   - **帧信息显示**: 当前帧/总帧数
   - **帧步进**: 前进/后退单帧
   - 与BrainVisualization集成

4. **BrainConfigLoader.cs** - 配置加载器
   - 自动加载unity_config.json
   - 应用可视化设置
   - 配置OBJ模型路径

5. **WebSocketClient.cs** - 后端通信客户端
   - 连接TwinBrain后端服务器
   - 发送预测请求
   - 发送刺激模拟请求
   - 接收大脑状态更新

6. **StimulationInput.cs** - 刺激输入控制器
   - 用户界面控制
   - 选择目标脑区
   - 设置刺激参数
   - 发送刺激请求到后端

## 主要改进

### 1. 命名空间统一

**问题**: 原代码中`BrainVisualization.cs`使用`using TwinBrain;`但自身不在该命名空间中。

**解决方案**: 所有脚本现在都在`TwinBrain`命名空间下：

```csharp
namespace TwinBrain
{
    public class BrainVisualization : MonoBehaviour
    {
        // ...
    }
}
```

### 2. C# 6.0+ 兼容性

**问题**: 原代码使用了Unity 2017以下版本不支持的`?.`和`??`操作符。

**解决方案**: 替换为传统的null检查：

```csharp
// 旧代码 (C# 6.0+)
float activity = region.activity.fmri?.amplitude ?? 0f;

// 新代码 (Unity 2019+兼容)
float activity = 0f;
if (region.activity != null && region.activity.fmri != null)
{
    activity = region.activity.fmri.amplitude;
}
```

### 3. 多OBJ模型支持

**功能**: 支持从FreeSurfer .lh文件生成的独立脑区OBJ模型。

**实现**:
- `useObjModels` 选项控制是否使用OBJ模型
- `objDirectory` 指定OBJ文件目录
- 自动加载 `region_0001.obj`, `region_0002.obj` 等文件
- 回退到Sphere预制体或Primitive

**代码示例**:

```csharp
if (useObjModels)
{
    string objPath = Path.Combine(objDirectory, string.Format("region_{0:D4}.obj", region.id));
    regionObj = LoadObjModel(objPath);
}
```

### 4. 点击交互

**功能**: 用户可以点击脑区查看详细信息并选择目标区域。

**实现**:
- 射线检测鼠标点击
- 高亮选中的脑区
- 触发事件通知其他组件
- 支持多选

**代码示例**:

```csharp
if (enableInteraction && Input.GetMouseButtonDown(0))
{
    Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
    RaycastHit hit;
    
    if (Physics.Raycast(ray, out hit))
    {
        // 处理点击...
    }
}
```

### 5. 时间序列可视化与颜色映射

**重要**: 颜色映射基于整个时间序列的归一化，不区分真实/预测信号。

**功能**: 
- 加载整个时间序列并预计算全局最小/最大值
- 所有数值（真实和预测）使用统一的颜色刻度
- 便于观测不同时刻的状态变化

**实现**:
- `lowActivityColor` (蓝色) - 序列中的最小值
- `highActivityColor` (红色) - 序列中的最大值
- 每个时间点的颜色反映其在全局范围内的相对位置

**归一化公式**:
```csharp
// 计算整个序列的全局范围
globalMinActivity = min(所有时间点所有脑区的活动值)
globalMaxActivity = max(所有时间点所有脑区的活动值)

// 归一化每个值
normalizedActivity = (activity - globalMinActivity) / (globalMaxActivity - globalMinActivity)

// 映射到颜色
color = Color.Lerp(蓝色, 红色, normalizedActivity)
```

**优势**:
- 颜色一致性：相同活动值在所有时间点显示相同颜色
- 便于比较：可以直观比较不同时刻的活动模式
- 统一刻度：真实和预测数据使用相同标准

### 6. 时间轴控制（TimelineController.cs）

**功能**: 提供交互式时间轴控制

**UI组件**:
- **进度条滑块**: 拖动到任意时间点，实时更新可视化
- **播放/暂停按钮**: 控制动画自动播放
- **帧信息显示**: 显示 "Frame: 5 / 100"
- **步进按钮**: 前进/后退单帧（可选）

**代码示例**:
```csharp
// 跳转到特定帧
brainVis.SetFrame(50);

// 获取信息
int currentFrame = brainVis.GetCurrentFrame();
int totalFrames = brainVis.GetTotalFrames();
```

### 7. 虚拟刺激输入
### 6. 虚拟刺激输入

**功能**: 提供UI界面输入刺激参数并发送到后端。

**参数**:
- 目标脑区ID列表
- 刺激振幅 (0-5)
- 刺激模式 (constant, sine, pulse, ramp)

**使用**:
1. 点击脑区选择目标
2. 调整振幅滑块
3. 选择刺激模式
4. 点击发送按钮

## Unity场景设置

### 1. 创建BrainManager对象

```
Hierarchy > Create Empty > "BrainManager"
```

### 2. 添加核心组件

```
Add Component > Brain Visualization
Add Component > Brain Config Loader
Add Component > WebSocket Client
```

### 3. 创建UI Canvas

```
Hierarchy > UI > Canvas
```

添加StimulationInput组件到Canvas或子对象。

### 4. 配置组件属性

**BrainVisualization**:
- JSON Path: `StreamingAssets/state`
- Use Obj Models: ✓ (如果有OBJ文件)
- Obj Directory: `StreamingAssets/OBJ`
- Region Prefab: 拖入Sphere预制体（可选）
- Enable Interaction: ✓

**BrainConfigLoader**:
- Config Path: `StreamingAssets/unity_config.json`
- Auto Load: ✓

**WebSocketClient**:
- Server URL: `http://localhost:8765`
- Auto Connect: ✓

**StimulationInput**:
- 链接UI控件（Input Field, Slider, Dropdown, Button, Text）
- Brain Vis: 拖入BrainVisualization组件

## 文件结构

```
Unity项目/
├── Assets/
│   ├── Scripts/
│   │   ├── BrainDataStructures.cs
│   │   ├── BrainVisualization.cs
│   │   ├── BrainConfigLoader.cs
│   │   ├── WebSocketClient.cs
│   │   └── StimulationInput.cs
│   └── StreamingAssets/
│       ├── unity_config.json
│       ├── state/
│       │   ├── brain_state_t0000.json
│       │   └── sequence_index.json
│       └── OBJ/
│           ├── region_0001.obj
│           ├── region_0002.obj
│           └── ...
```

## 配置文件示例

**unity_config.json**:

```json
{
  "project_name": "TwinBrain Unity Project",
  "atlas": "Schaefer200",
  "data_paths": {
    "json_dir": "state",
    "obj_dir": "OBJ",
    "materials_dir": "Materials"
  },
  "visualization": {
    "region_scale": 1.0,
    "activity_threshold": 0.3,
    "connection_threshold": 0.5,
    "show_connections": true,
    "fps": 10,
    "auto_play": true,
    "use_obj_models": true
  },
  "colors": {
    "low_activity": {"r": 0, "g": 0, "b": 255},
    "high_activity": {"r": 255, "g": 0, "b": 0},
    "predicted_signal": {"r": 0, "g": 255, "b": 0}
  }
}
```

## 使用流程

### 基础可视化

1. 启动Unity项目
2. 确保JSON数据在StreamingAssets/state目录
3. 按Play键
4. 使用空格键控制动画播放/暂停
5. 使用R键重新加载数据

### 交互式刺激

1. 启动后端服务器:
```bash
python -m unity_integration.realtime_server
```

2. Unity自动连接到后端

3. 点击脑区选择目标

4. 设置刺激参数

5. 点击"发送刺激"按钮

6. 观察预测结果（绿色显示）

## 故障排除

### 问题：编译错误 "type or namespace name 'TwinBrain' could not be found"

**解决**: 确保所有脚本都在同一个Assembly中，或者在项目设置中正确配置Assembly Definition。

### 问题：OBJ模型未显示

**解决**: 
1. 检查`objDirectory`路径是否正确
2. 确保OBJ文件名格式为`region_XXXX.obj`
3. 考虑使用Unity Editor预先导入OBJ文件作为资产

### 问题：WebSocket连接失败

**解决**:
1. 确认后端服务器正在运行
2. 检查URL和端口配置
3. 查看Unity Console的错误日志

### 问题：点击无反应

**解决**:
1. 确保脑区GameObject有Collider组件
2. 启用`enableInteraction`选项
3. 检查Camera和EventSystem是否正确配置

## 性能优化建议

### 对于大量脑区（200+）

1. **使用LOD系统**: 根据相机距离显示不同细节级别
2. **GPU Instancing**: 对相同几何体使用GPU实例化
3. **禁用不活跃脑区**: 活动值低于阈值的脑区不显示
4. **降低球体分辨率**: 减少顶点数量

### 对于实时更新

1. **限制帧率**: 使用较低的fps值（10-15）
2. **批量更新**: 不要每帧都更新所有脑区
3. **异步加载**: 使用协程加载大型数据文件

## 扩展开发

### 添加自定义可视化

继承`BrainVisualization`类或监听事件：

```csharp
public class CustomVisualizer : MonoBehaviour
{
    void Start()
    {
        BrainVisualization brainVis = GetComponent<BrainVisualization>();
        brainVis.OnRegionClicked += HandleRegionClick;
    }
    
    void HandleRegionClick(int regionId, RegionData data)
    {
        // 自定义处理逻辑
    }
}
```

### 添加新的刺激模式

在`StimulationInput.cs`中添加新模式：

```csharp
public string[] patterns = new string[] { 
    "constant", "sine", "pulse", "ramp", "custom" 
};
```

## 更新日志

### 2024-02-13
- 修复命名空间错误
- 移除C# 6.0+特性以兼容Unity 2019+
- 添加多OBJ模型支持
- 添加点击交互功能
- 添加虚拟刺激输入UI
- 改进颜色映射系统
- 添加预测信号可视化

## 联系与支持

- GitHub Issues: https://github.com/sheinclotho/twinbrain/issues
- 文档: 查看项目中的其他.md文件

---

**注意**: 本文档随代码更新而更新。请始终参考最新版本。
