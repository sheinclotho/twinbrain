# TwinBrain 数字孪生脑系统（中文）

基于多模态脑成像数据的数字孪生脑系统，用于大脑活动建模、预测和可视化。

## 📚 完整文档

### 主要文档
- **[系统使用指南](TwinBrain系统使用指南.md)** - 完整用户手册
  - 安装和配置
  - 数据准备
  - 训练流程
  - 预测和推理
  - Unity前端集成
  - 配置参数详解
  - 命令行使用
  - 常见问题

- **[优化方向和研究思路](TwinBrain优化方向和研究思路.md)** - 研究规划
  - 增强预测能力
  - 虚拟刺激和扰动
  - 多模态融合优化
  - 因果推断和网络分析
  - Unity可视化增强
  - 意识相关特征提取
  - 计算效率优化
  - 长期研究方向

### 技术文档
- [重构总结](REFACTORING_SUMMARY_CN.md) - 代码重构和设计改进
- [重构实施](REFACTORING_IMPLEMENTATION.md) - 实施细节和迁移指南

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

### 3. 导出大脑状态（Unity可视化）

```python
from unity_integration import BrainStateExporter

# 初始化导出器
exporter = BrainStateExporter(atlas_info)

# 导出单个时间点
brain_state = exporter.export_brain_state(
    brain_activity={'fmri': fmri_data, 'eeg': eeg_data},
    time_point=100,
    output_path="brain_state.json"
)

# 导出时间序列（用于动画）
exporter.export_sequence(
    brain_activity={'fmri': fmri_data},
    output_dir="brain_sequence/",
    start=0, end=200, step=5
)
```

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

### 大脑状态预测
- 单步和多步未来预测
- 基于神经动力学约束
- 条件化预测（考虑刺激影响）

### 虚拟刺激系统
- **多种刺激模式**: 脉冲、正弦、渐变、连续
- **空间效应**: 空间扩散和传播
- **网络效应**: 通过连接传播
- **反向设计**: 给定目标设计刺激方案

### Unity 3D可视化
- **JSON导出**: 标准格式的大脑状态
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
1. 阅读 [系统使用指南](TwinBrain系统使用指南.md) 第1-3章
2. 按照快速开始步骤安装和测试
3. 尝试运行训练示例
4. 导出第一个JSON文件

### 进阶使用
1. 配置系统详解（使用指南第4章）
2. 训练流程说明（使用指南第5章）
3. 预测和推理（使用指南第6章）
4. Unity集成（使用指南第7章）

### 研究开发
1. 阅读 [优化方向和研究思路](TwinBrain优化方向和研究思路.md)
2. 了解核心研究方向
3. 查看代码实现示例
4. 参与项目开发

## 🔬 研究方向

详见 [优化方向文档](TwinBrain优化方向和研究思路.md)：

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

**最后更新**: 2026-01-31  
**版本**: 2.0  
**维护者**: TwinBrain Development Team

> "探索意识的本质，理解大脑的奥秘。"
