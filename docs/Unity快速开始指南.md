# TwinBrain Unity 快速开始指南

本指南帮助你在5分钟内运行TwinBrain Unity可视化系统。

## 前提条件

- Python 3.7+
- Unity 2020.3 或更高版本
- 基本的命令行使用经验

## 步骤 1: 安装Python依赖

```bash
cd /path/to/twinbrain
pip install -r requirements.txt
pip install websockets  # 可选，用于实时服务器
```

## 步骤 2: 生成Unity资源

运行一键式自动化脚本：

```bash
python unity_automation.py --mode export
```

这将在 `unity_output/` 目录下生成：

```
unity_output/
├── json/                   # 脑状态JSON文件（40+ 文件）
├── obj/                    # 3D模型文件
├── materials/              # 材质配置
├── UnityScripts/           # Unity C#脚本（7个文件）
├── unity_config.json       # Unity配置
├── unity_scene_config.json # 场景配置
├── unity_prefab_config.json# 预制体配置
└── README_UNITY.md        # 详细说明
```

⏱️ 预计时间：30-60秒

## 步骤 3: 创建Unity项目

1. 打开Unity Hub
2. 新建3D项目，命名为 `TwinBrain_Visualization`
3. 等待项目创建完成

## 步骤 4: 安装Newtonsoft.Json包

在Unity中：

1. Window → Package Manager
2. 点击 "+" → Add package from git URL
3. 输入: `com.unity.nuget.newtonsoft-json`
4. 等待安装完成

⏱️ 预计时间：1-2分钟

## 步骤 5: 导入资源

1. 在Unity项目的Assets文件夹下创建 `Scripts` 和 `Data` 文件夹
2. 将 `unity_output/UnityScripts/` 下所有 `.cs` 文件复制到 `Assets/Scripts/`
3. 将 `unity_output/json/` 复制到 `Assets/Data/JSON/`
4. 将 `unity_output/obj/` 复制到 `Assets/Data/OBJ/`
5. 等待Unity编译脚本

⏱️ 预计时间：1-2分钟

## 步骤 6: 配置场景

1. 在Hierarchy中创建空GameObject，命名为 `BrainSystem`
2. 选中 `BrainSystem`，在Inspector中添加组件：
   - `BrainVisualization`
   - `BrainConfigLoader`
   - `BrainInteractionController`
   - `WebSocketClient` (可选)

3. 配置 `BrainVisualization` 组件：
   - JSON Path: `Assets/Data/JSON/`
   - Load Sequence: ✓ (勾选)
   - FPS: 10
   - Auto Play: ✓ (勾选)
   - Region Scale: 1.0
   - Activity Threshold: 0.3
   - Show Connections: ✓ (勾选)

4. 配置 `BrainConfigLoader` 组件：
   - Config Path: `Assets/Data/unity_config.json` (如果复制了该文件)
   - Auto Load: ✓ (勾选)

⏱️ 预计时间：2-3分钟

## 步骤 7: 运行！

点击Unity的 Play 按钮 ▶️

你应该看到：
- 200个脑区显示为彩色球体
- 颜色表示活动强度（蓝色=低，红色=高）
- 自动播放动画显示脑活动变化
- 连接线显示脑区之间的联系

### 基本控制

- **空格键**: 播放/暂停动画
- **R键**: 重新加载
- **鼠标悬停**: 高亮脑区
- **左键点击**: 选择脑区

## （可选）步骤 8: 启动后端服务器

如果你想使用实时功能（预测、刺激模拟），需要启动后端服务器：

```bash
python unity_automation.py --mode server
```

服务器将在 `ws://localhost:8765` 启动

然后在Unity的 `WebSocketClient` 组件中：
- Server URL: `ws://localhost:8765`
- Auto Connect: ✓

重新运行Unity项目，它将自动连接到服务器。

## 故障排除

### 问题 1: 看不到任何脑区

**解决方案:**
- 降低 Activity Threshold (例如设为 0.1)
- 检查 JSON Path 是否正确
- 查看Unity Console是否有错误

### 问题 2: 脚本编译错误

**解决方案:**
- 确认已安装 Newtonsoft.Json 包
- 重启Unity
- 检查所有7个脚本都已复制

### 问题 3: 连接不显示

**解决方案:**
- 确认 Show Connections 已勾选
- 降低 Connection Threshold
- 确认JSON文件包含connections数据

### 问题 4: 性能问题/卡顿

**解决方案:**
- 提高 Activity Threshold (显示更少脑区)
- 取消勾选 Show Connections
- 降低 FPS
- 使用更简单的Region Prefab

## 进阶功能

### 虚拟刺激

1. 在Hierarchy中为 `BrainSystem` 添加：
   - `BrainRegionSelector`
   - `StimulationController`

2. 创建UI Canvas和控制按钮

3. 连接UI到组件

4. 确保后端服务器运行

5. 在Unity中：
   - 点击选择目标脑区
   - 设置刺激参数
   - 点击"应用刺激"
   - 观察预测的脑活动变化

### 预测对比

1. 添加 `PredictionVisualizer` 组件

2. 按 M 键切换可视化模式：
   - Real Only: 仅显示真实数据
   - Prediction Only: 仅显示预测
   - Side by Side: 并排对比
   - Overlay: 叠加显示

## 下一步

- 阅读 `unity_output/README_UNITY.md` 了解详细功能
- 自定义颜色和材质
- 添加UI控制面板
- 尝试VR模式
- 导出可视化视频

## 获取帮助

- 查看 [详细文档](unity_output/README_UNITY.md)
- 查看 [系统使用指南](docs/TwinBrain系统使用指南.md)
- 查看 [Unity工作流说明](docs/Unity工作流说明.md)
- 提交 [GitHub Issue](https://github.com/sheinclotho/twinbrain/issues)

## 完成时间

- 基础可视化：约 5-10 分钟
- 包含实时功能：约 10-15 分钟

祝你使用愉快！🧠✨
