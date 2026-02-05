# TwinBrain 数字孪生脑系统（中文）

基于多模态脑成像数据的数字孪生脑系统，用于大脑活动建模、预测和可视化。

## ⚠️ 重要：正确理解 FreeSurfer 工作流

👉 **[正确工作流程说明](正确工作流程说明.md)** - **必读**：FreeSurfer 与数据加载的正确理解

**关键概念**：
- **FreeSurfer 文件** = Unity 前端结构（一次性创建）
- **数据文件夹** = 内容（可随时更换）
- **不需要两者同时存在**

```
步骤1: FreeSurfer → Unity 前端结构 (一次性)
步骤2: 数据文件夹 → JSON → 加载到前端 (可重复)
步骤3: 交互刺激 → 后端预测 → 实时更新 (可选)
```

## 🚀🚀 NEW! Unity 完全自动化工作流 🆕🆕

**一键式自动化设置，无需手动配置！**

### 快速开始（3步完成）

```bash
# 步骤1: 一键式自动化设置
python setup_unity_workflow.py --auto-setup

# 步骤2: 启动后端服务器
cd unity_project
python start_backend_server.py

# 步骤3: 在Unity中导入生成的资源
# 将 unity_project/Unity_Assets/ 复制到Unity项目的Assets目录
```

**自动生成内容**:
- ✅ 完整的文件夹结构（FreeSurfer文件、数据文件、模型输出）
- ✅ 7个Unity C#脚本（包含交互按钮功能）
- ✅ 后端模型服务器（支持加载训练模型）
- ✅ 虚拟刺激完整闭环流程
- ✅ 跨平台启动脚本（Windows/Linux/Mac）
- ✅ 完整中文文档

**Unity中的按钮功能**:
- 📁 **加载数据** - 从文件夹读取JSON并播放动画
- ⏱️ **时间轴控制** - 播放/暂停/停止，显示演化过程
- 🎯 **选择脑区** - 点击选择目标脑区
- ⚡ **虚拟刺激** - 设置参数并应用刺激
- 🔮 **请求预测** - 后端模型生成预测
- 🔄 **自动保存** - 结果自动保存到数据文件夹

👉 **[Unity全自动化使用指南](docs/Unity全自动化使用指南.md)** - **完整教程** 🔥🔥

## 📚 完整文档

👉 **[文档索引](docs/文档索引.md)** - 按需求快速查找文档

### 主要文档

- **[Unity全自动化使用指南](docs/Unity全自动化使用指南.md)** - **一键式自动化完整教程** 🔥🔥🆕🆕🆕
  - 一键式设置和文件夹生成
  - 后端模型服务器配置
  - Unity交互按钮完整说明
  - 虚拟刺激完整闭环流程
  - 故障排除和高级配置

- **[正确工作流程说明](正确工作流程说明.md)** - **推荐首先阅读** 🆕🆕🆕
  - 正确理解 FreeSurfer 的作用
  - 完整工作流程图
  - 步骤式教程
  - 常见误解解答

- **[Unity工作流说明](docs/Unity工作流说明.md)** - Unity集成完整指南 🆕
  - 自动化工作流
  - 一键导出（JSON + OBJ）
  - Unity配置自动生成
  - 前后端实时通信
  - 使用示例和最佳实践

- **[FreeSurfer使用指南](docs/FreeSurfer使用指南.md)** - FreeSurfer表面数据支持 🆕🆕
  - FreeSurfer = 前端结构定义
  - 加载 FreeSurfer 表面文件（.pial）
  - 加载注释文件（.annot）
  - 导出真实大脑表面网格
  - Python API 和命令行使用

- **[如何使用多个脑模文件](如何使用多个脑模文件.md)** - FreeSurfer 快速答疑 🆕
  - 直接回答 FreeSurfer 文件使用问题
  - 工作流程图解

- **[系统使用指南](docs/TwinBrain系统使用指南.md)** - 完整用户手册
  - 安装和配置
  - 数据准备
  - 训练流程
  - 预测和推理
  - Unity前端集成
  - 配置参数详解
  - 命令行使用
  - 常见问题

- **[新功能说明](docs/NEW_FEATURES.md)** - 最新功能文档
  - 增强训练监控系统
  - 多步未来预测
  - 配置和使用指南
  - 示例代码

- **[优化方向和研究思路](OPTIMIZATION_DIRECTIONS.md)** - 研究规划
  - ✅ 增强预测能力（已实现）
  - ✅ 训练监控系统（已实现）
  - ✅ Unity工作流自动化（已实现）🆕
  - ✅ Unity完全自动化（已实现）🔥🆕🆕
  - ✅ FreeSurfer表面数据支持（已实现）🆕🆕
  - ✅ 后端模型服务（已实现）🆕🆕
  - 虚拟刺激和扰动
  - 多模态融合优化
  - 因果推断和网络分析
  - Unity可视化增强
  - 意识相关特征提取
  - 计算效率优化
  - 长期研究方向

### 项目文档
- [更新历史](CHANGELOG.md) - 版本更新和变更记录

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认v4配置（推荐）
python main.py train --config config/default.yaml

# 或使用v3配置（复现）
python main.py train --config config/v3_legacy.yaml
```

### 3. Unity 可视化（一键导出） 🆕

```python
from unity_integration import run_unity_workflow, WorkflowConfig

# 简单配置
config = WorkflowConfig(
    output_dir='output/unity_export',
    export_formats=['json', 'obj'],  # 导出 JSON 和 OBJ
    time_step=5
)

# 一键完成：数据处理 → 格式转换 → 配置生成
results = run_unity_workflow(config)

# 自动生成：
# - JSON 脑状态文件（用于 Unity 加载）
# - OBJ 3D 模型（200个脑区）
# - unity_config.json（Unity 项目配置）
# - 材质配置文件
```

**详细说明**: [Unity工作流说明](docs/Unity工作流说明.md)

### 3.5. 使用 FreeSurfer 表面数据 🆕🆕

```python
from unity_integration import run_unity_workflow, WorkflowConfig

# 使用 FreeSurfer 表面文件
config = WorkflowConfig(
    data_source='freesurfer',  # 使用 FreeSurfer 数据源
    
    # FreeSurfer 文件路径
    freesurfer_lh_surface='data/lh.pial',
    freesurfer_rh_surface='data/rh.pial',
    freesurfer_lh_annot='data/lh.Schaefer2018_200Parcels_7Networks_order.annot',
    freesurfer_rh_annot='data/rh.Schaefer2018_200Parcels_7Networks_order.annot',
    
    output_dir='output/freesurfer_export',
    export_formats=['json', 'obj'],
    export_surface_mesh=True,  # 导出真实的大脑表面网格
)

results = run_unity_workflow(config)

# 自动生成：
# - JSON 脑状态文件（基于真实脑区位置）
# - OBJ 球体模型（200个脑区）
# - OBJ 表面网格（真实的大脑皮层表面）
# - unity_config.json（Unity 项目配置）
```

**或使用命令行**:
```bash
python unity_automation.py --freesurfer \
  --lh-surface data/lh.pial \
  --rh-surface data/rh.pial \
  --lh-annot data/lh.Schaefer2018_200Parcels_7Networks_order.annot \
  --rh-annot data/rh.Schaefer2018_200Parcels_7Networks_order.annot \
  --export-surface \
  --output freesurfer_output
```

**详细说明**: [FreeSurfer使用指南](docs/FreeSurfer使用指南.md)

### 4. 虚拟刺激模拟

```python
from unity_integration import StimulationSimulator, StimulationConfig

# 创建模拟器
simulator = StimulationSimulator(n_regions=200, connectivity=connectivity_matrix)

# 配置刺激参数
stim_config = StimulationConfig(
    target_regions=[10, 15, 20],  # 目标脑区
    amplitude=0.5,                 # 刺激强度
    duration=10,                   # 持续时间（时间步）
    pattern="sine",                # 刺激模式
    frequency=10.0                 # 频率（Hz，对于sine模式）
)

# 模拟大脑响应
trajectory, metrics = simulator.simulate_response(
    initial_state=current_brain_state,
    config=stim_config,
    n_steps=50
)
```

### 5. 实时WebSocket服务器

```python
from unity_integration import BrainVisualizationServer

# 创建服务器
server = BrainVisualizationServer(
    model=trained_model,
    exporter=exporter,
    simulator=simulator,
    port=8765
)

# 启动服务器
server.start()  # Unity可通过ws://localhost:8765连接
```

## 💡 核心功能

### 多模态数据融合
- **fMRI**: 功能磁共振，空间分辨率高
- **EEG**: 脑电图，时间分辨率高
- **DTI**: 扩散张量成像，结构连接

### 异构图神经网络
- 节点表示不同模态的脑区
- 边表示结构和功能连接
- 跨模态信息传递和对齐

### 大脑状态预测 🆕
- ✅ **多步未来预测**: GRU + 注意力机制
- ✅ **条件化预测**: 考虑刺激影响
- 自回归预测机制
- 可配置预测步数（1-50步）

### 增强训练监控 🆕
- ✅ **MetricsTracker**: 完整指标追踪
- ✅ **自动化日志**: 损失分量、梯度统计
- ✅ **JSON导出**: 便于分析和可视化
- ✅ **训练摘要**: 自动生成报告

### 虚拟刺激系统
- **多种刺激模式**: 脉冲、正弦、渐变、连续
- **空间效应**: 空间扩散和传播
- **网络效应**: 通过连接传播
- **反向设计**: 给定目标设计刺激方案

### Unity 3D可视化
- **自动化工作流**: 一键完成数据处理到可视化 🆕
- **JSON导出**: 标准格式的大脑状态
- **OBJ导出**: 200个脑区的3D模型 🆕
- **自动配置**: Unity配置和材质自动生成 🆕
- **实时通信**: WebSocket协议
- **交互式展示**:
  - 脑区活跃度热图（颜色映射）
  - 连接强度可视化（线条粗细）
  - 时间序列动画
  - 刺激效应实时预览

## 🎯 应用场景

1. **意识科学研究**
   - 量化意识水平（整合信息Φ）
   - 识别意识神经关联物(NCC)
   - 意识状态转换动力学

2. **神经调控**
   - TMS/tACS刺激方案设计
   - 刺激参数优化
   - 预测刺激效应

3. **脑机接口**
   - 预测大脑响应
   - 优化接口参数
   - 实时反馈控制

4. **临床应用**
   - 异常脑活动检测
   - 疾病模式识别
   - 治疗方案评估

5. **教学演示**
   - 3D可视化大脑活动
   - 交互式教学工具
   - 科普展示

## 📊 系统架构

```
TwinBrain 系统
├─ 数据层
│  ├─ fMRI数据处理
│  ├─ EEG数据处理
│  └─ DTI连接矩阵
│
├─ 模型层
│  ├─ 图构建（异构图）
│  ├─ 图编码器（GNN）
│  ├─ 时序建模（GRU）
│  ├─ 特征解码器
│  └─ 跨模态对齐
│
├─ 预测层
│  ├─ 未来状态预测
│  ├─ 刺激响应模拟
│  └─ 因果推断
│
└─ 可视化层
   ├─ JSON导出
   ├─ WebSocket服务
   └─ Unity前端
```

## 📖 详细文档导航

### 新手入门
1. 阅读 [系统使用指南](docs/TwinBrain系统使用指南.md) 第1-3章
2. 按照快速开始步骤安装和测试
3. 尝试运行训练示例
4. 导出第一个JSON文件

### 进阶使用
1. 配置系统详解（使用指南第4章）
2. 训练流程说明（使用指南第5章）
3. 预测和推理（使用指南第6章）
4. Unity集成（使用指南第7章）

### 研究开发
1. 阅读 [优化方向和研究思路](OPTIMIZATION_DIRECTIONS.md)
2. 了解核心研究方向
3. 查看代码实现示例
4. 参与项目开发

## 🔬 研究方向

详见 [优化方向文档](OPTIMIZATION_DIRECTIONS.md)：

1. **预测能力增强**
   - 多步预测
   - 物理约束
   - 条件预测

2. **因果分析**
   - Granger因果
   - 有效连接
   - 网络拓扑

3. **意识计算**
   - 整合信息理论
   - 神经振荡
   - 意识指标

4. **性能优化**
   - 模型压缩
   - 并行计算
   - 实时推理

## 🛠️ 技术栈

- **深度学习**: PyTorch, PyTorch Geometric
- **脑成像**: Nibabel, MNE-Python
- **科学计算**: NumPy, SciPy
- **可视化**: Matplotlib, NetworkX
- **通信**: WebSocket, JSON
- **前端**: Unity 3D (C#)

## 🤝 贡献

欢迎贡献代码、报告问题、提出建议！

- 提交Issue: 报告bug或请求功能
- Pull Request: 贡献代码改进
- 文档改进: 帮助完善文档

## 📄 许可证

[待添加]

## 📧 联系方式

项目地址: https://github.com/sheinclotho/twinbrain

---

**最后更新**: 2026-02-04  
**版本**: 2.4  
**维护者**: TwinBrain Development Team

> "探索意识的本质，理解大脑的奥秘。"

---

## 🎉 最新更新 (2026-02-04)

### Unity 完全自动化工作流 🔥🔥🆕🆕🆕
- ✅ **一键式设置脚本** (`setup_unity_workflow.py`): 自动创建所有必需文件和文件夹
- ✅ **自动文件夹生成**: FreeSurfer文件夹 + 数据文件夹（原始/缓存/模型输出）
- ✅ **Unity脚本自动生成**: 7个完整的C#脚本，包含交互按钮功能
- ✅ **ModelServer**: 后端模型加载和推理服务
- ✅ **虚拟刺激完整闭环**: 从Unity选择脑区 → 后端计算 → 自动保存 → Unity加载动画
- ✅ **跨平台启动脚本**: Python/Windows批处理/Linux Shell脚本
- ✅ **完整中文文档**: 使用指南和故障排除
- 📚 详见 **[Unity全自动化使用指南](docs/Unity全自动化使用指南.md)** 🔥

**Unity中的按钮功能**:
```
✅ 加载数据按钮 - 从文件夹读取JSON
✅ 刷新按钮 - 重新扫描数据文件
✅ 播放/暂停/停止 - 控制动画
✅ 时间轴滑块 - 显示演化过程
✅ 选择脑区 - 鼠标点击选择
✅ 应用刺激 - 设置参数并发送
✅ 请求预测 - 后端模型生成预测
```

**使用示例**:
```bash
# 一键式设置
python setup_unity_workflow.py --auto-setup

# 启动后端服务器（自动加载训练模型）
cd unity_project
python start_backend_server.py

# 在Unity中导入生成的资源
# 将 unity_project/Unity_Assets/ 复制到Unity项目
```

### 之前更新 - FreeSurfer 表面数据支持 🆕🆕
- ✅ **FreeSurferLoader**: 加载 FreeSurfer 表面文件（.pial）和注释文件（.annot）
- ✅ **真实大脑表面**: 导出真实的大脑皮层表面网格为 OBJ 格式
- ✅ **自动提取脑区**: 从 FreeSurfer 分割自动提取脑区位置和网络信息
- ✅ **完整集成**: 与现有 Unity 工作流无缝集成
- ✅ **命令行支持**: 通过 unity_automation.py 直接使用 FreeSurfer 数据
- 📚 详见 [FreeSurfer使用指南](docs/FreeSurfer使用指南.md)

**使用示例**:
```python
# 加载 FreeSurfer 表面文件
config = WorkflowConfig(
    data_source='freesurfer',
    freesurfer_lh_surface='lh.pial',
    freesurfer_rh_surface='rh.pial',
    freesurfer_lh_annot='lh.Schaefer2018_200Parcels_7Networks_order.annot',
    freesurfer_rh_annot='rh.Schaefer2018_200Parcels_7Networks_order.annot',
    export_surface_mesh=True  # 导出真实表面网格
)
results = run_unity_workflow(config)
```

### 之前更新 (2026-02-02)

#### Unity 工作流自动化 🆕
- ✅ **WorkflowManager**: 一键完成完整工作流
- ✅ **多格式导出**: JSON + OBJ 同时导出
- ✅ **自动配置**: Unity配置和材质自动生成
- ✅ **前后端集成**: WebSocket 实时通信完整实现
- 📚 详见 [Unity工作流说明](docs/Unity工作流说明.md) 和 [Unity更新说明](docs/Unity更新说明.md)

### 更早更新
- ✅ **多步未来预测**: 实现 PredictorHead 模块
- ✅ **增强监控**: 完整的训练指标追踪系统
- ✅ **自动化日志**: MetricsTracker 和 JSON 导出
- 📚 详见 [新功能说明](docs/NEW_FEATURES.md)
