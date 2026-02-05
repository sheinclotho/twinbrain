# TwinBrain Unity 一键使用指南

## 🎯 目标

让你在**5分钟内**完成从零到Unity可视化大脑活动的全部设置。

## ⚡ 快速开始（3步完成）

### 步骤1: 运行一键式设置脚本

```bash
# 克隆仓库（如果还没有）
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 运行一键设置（自动生成所有必需文件）
python setup_unity_project.py
```

这个命令会：
- ✅ 创建完整的Unity项目结构
- ✅ 复制所有必需的C#脚本
- ✅ 生成示例大脑数据（100个脑区）
- ✅ 生成配置文件
- ✅ 生成详细的使用说明

完成后，你会在 `Unity_TwinBrain/` 目录下看到所有文件。

### 步骤2: 在Unity中创建项目

1. **打开Unity Hub**
2. **点击"New Project"**
3. **选择"3D"模板**
4. **项目名称**: 任意（例如：TwinBrain_Demo）
5. **点击"Create"**

### 步骤3: 导入文件到Unity

#### 3.1 导入C#脚本

1. 在Unity项目中，找到 `Assets/` 文件夹
2. 将 `Unity_TwinBrain/Scripts/` **整个文件夹**复制到 `Assets/`
3. Unity会自动编译脚本

#### 3.2 安装JSON库

在Unity中安装Newtonsoft.Json：

**方法1：Package Manager（推荐）**
1. 打开 Window > Package Manager
2. 点击左上角的 "+" 按钮
3. 选择 "Add package from git URL"
4. 输入：`com.unity.nuget.newtonsoft-json`
5. 点击 Add

**方法2：手动下载**
- 从 https://github.com/jilleJr/Newtonsoft.Json-for-Unity/releases 下载
- 导入到Unity项目

#### 3.3 导入数据文件

1. 在Unity的 `Assets/` 目录下创建 `StreamingAssets` 文件夹（如果不存在）
2. 将 `Unity_TwinBrain/BrainData/` **整个文件夹**复制到 `Assets/StreamingAssets/`

## 🎮 在Unity中设置场景

### 1. 创建脑管理器

1. 在Hierarchy窗口，右键 > Create Empty
2. 命名为 `BrainManager`
3. 选中它

### 2. 添加脚本组件

在Inspector窗口，点击 "Add Component"：

1. **添加 "Brain Visualization"**
   - Json Path: `StreamingAssets/BrainData/JSON`
   - Load Sequence: ✓ 勾选
   - Auto Play: ✓ 勾选（可选）

2. **添加 "Brain Config Loader"**
   - Config Path: `StreamingAssets/BrainData/Config/unity_config.json`
   - Auto Load: ✓ 勾选

### 3. 创建脑区预制体（可选，但推荐）

1. Hierarchy中：GameObject > 3D Object > Sphere
2. 在Inspector中，设置Scale为 (0.1, 0.1, 0.1)
3. 将Sphere从Hierarchy拖到Project窗口（创建Prefab）
4. 选中BrainManager，将Prefab拖到Brain Visualization的"Region Prefab"字段
5. 删除Hierarchy中的Sphere

### 4. 创建连接材质（可选）

1. Project窗口：右键 > Create > Material
2. 命名为 "ConnectionMaterial"
3. 选中BrainManager，将材质拖到Brain Visualization的"Connection Material"字段

## 🚀 运行！

点击Unity的**播放按钮**（▶️），你应该能看到：
- 100个脑区以球形排列
- 颜色表示活动强度（蓝色=低，红色=高）
- 连接线显示脑区间的关系

## ⌨️ 快捷键

- **空格键**: 播放/暂停动画
- **R键**: 重新加载数据

## 🎨 调整可视化效果

选中BrainManager，在Brain Visualization组件中调整：

| 参数 | 说明 | 建议值 |
|-----|------|--------|
| Region Scale | 脑区大小 | 0.5-2.0 |
| Activity Threshold | 显示阈值 | 0.1-0.5 |
| Show Connections | 显示连接 | 开/关 |
| Connection Threshold | 连接阈值 | 0.3-0.7 |
| FPS | 动画帧率 | 5-30 |
| Low Activity Color | 低活动颜色 | 蓝色 |
| High Activity Color | 高活动颜色 | 红色 |

## 📊 使用你自己的数据

### 从TwinBrain模型导出数据

如果你已经训练了TwinBrain模型：

```bash
# 训练模型（如果还没有）
python main.py train --config config/default.yaml

# 导出JSON数据
python -m unity_integration.brain_state_exporter \
    --model results/hetero_gnn_trained.pt \
    --output unity_data \
    --start 0 \
    --end 100 \
    --step 1
```

然后将 `unity_data/json/` 中的文件复制到Unity的 `StreamingAssets/BrainData/JSON/`

### 使用FreeSurfer数据

如果你有FreeSurfer处理的真实大脑表面数据：

```bash
# 设置时包含FreeSurfer文件
python setup_unity_project.py --with-freesurfer /path/to/freesurfer/files
```

## 🌐 实时连接（高级功能）

### 启动后端服务器

```bash
# 在TwinBrain项目目录
python -m unity_integration.realtime_server
```

服务器将在 `ws://localhost:8765` 启动。

### 在Unity中连接

1. 选中BrainManager
2. 添加组件："WebSocket Client"
3. Server URL: `ws://localhost:8765`
4. Auto Connect: ✓ 勾选

现在你可以：
- 实时获取大脑状态
- 请求模型预测
- 模拟刺激效果

### WebSocket API示例

```csharp
// 获取当前状态
webSocketClient.GetBrainState();

// 请求10步预测
webSocketClient.RequestPrediction(10);

// 模拟刺激
int[] targetRegions = new int[] { 10, 20, 30 };
webSocketClient.SimulateStimulation(targetRegions, 0.5f, "sine");
```

## 📱 构建应用（可选）

### 构建桌面应用

1. File > Build Settings
2. 选择平台（PC, Mac, Linux）
3. 点击"Build"
4. 选择输出目录
5. 完成！

### 构建WebGL应用

1. File > Build Settings
2. 选择"WebGL"
3. 点击"Switch Platform"
4. 点击"Build"
5. 部署到Web服务器

注意：WebGL版本的WebSocket需要额外配置JavaScript插件。

## 🎓 学习资源

### Unity脚本API

**BrainVisualization**
- `LoadSingleState(string path)`: 加载单个状态
- `LoadSequence()`: 加载序列
- `Play()`: 播放动画
- `Pause()`: 暂停动画
- `Reload()`: 重新加载

**WebSocketClient**
- `Connect()`: 连接服务器
- `Disconnect()`: 断开连接
- `GetBrainState()`: 获取状态
- `RequestPrediction(int n)`: 请求预测
- `SimulateStimulation(...)`: 模拟刺激

### 示例代码

查看 `Scripts/` 目录中的注释，每个方法都有详细说明。

## ❓ 常见问题

### Q: 脚本编译错误？
**A**: 
1. 确认已安装Newtonsoft.Json
2. 确认所有4个.cs文件都在Scripts文件夹中
3. 重启Unity

### Q: 看不到任何可视化？
**A**: 
1. 检查Console窗口是否有错误
2. 确认JSON路径正确：`StreamingAssets/BrainData/JSON`
3. 确认"Load Sequence"已勾选
4. 尝试降低"Activity Threshold"

### Q: 连接线不显示？
**A**: 
1. 确认"Show Connections"已勾选
2. 降低"Connection Threshold"
3. 指定或创建Connection Material

### Q: 性能太慢？
**A**: 
1. 减少显示的脑区数量（提高Activity Threshold）
2. 关闭连接显示
3. 减少FPS
4. 优化脑区Prefab（减少面数）

### Q: WebSocket连接失败？
**A**: 
1. 确认后端服务器正在运行
2. 检查防火墙设置
3. 确认URL正确（ws://localhost:8765）

### Q: 如何使用不同的脑图谱？
**A**: 
修改数据生成时的图谱参数，或直接使用包含不同图谱的JSON文件。

## 🔧 高级配置

### 自定义可视化

编辑 `Scripts/BrainVisualization.cs` 中的以下方法：
- `CreateRegion()`: 自定义脑区外观
- `CreateConnection()`: 自定义连接样式
- `UpdateVisualization()`: 自定义更新逻辑

### 自定义颜色映射

修改配置文件 `Config/unity_config.json`:

```json
{
  "colors": {
    "low_activity": {"r": 0, "g": 255, "b": 0},
    "high_activity": {"r": 255, "g": 0, "b": 0}
  }
}
```

### 性能优化

1. **对象池**: 重用脑区GameObject而不是销毁/创建
2. **LOD**: 根据距离使用不同细节级别
3. **遮挡剔除**: 使用Unity的遮挡剔除系统
4. **批处理**: 使用GPU Instancing

## 📞 获取帮助

- **GitHub Issues**: https://github.com/sheinclotho/twinbrain/issues
- **文档**: 查看项目README和其他文档
- **示例**: 运行项目中的example_*.py文件

## 🎉 你完成了！

现在你已经掌握了TwinBrain Unity集成的全部基础知识。

**接下来可以做什么？**
1. 导入自己的脑数据
2. 训练自己的模型
3. 实现自定义可视化
4. 探索实时交互功能
5. 构建独立应用分享给他人

祝你使用愉快！🧠✨
