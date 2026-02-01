# TwinBrain - 数字孪生脑系统

基于多模态脑成像数据（fMRI、EEG、DTI）的数字孪生脑系统，使用异构图神经网络进行脑信号重建、预测和Unity可视化。

**核心特性**:
- 🧠 多模态脑成像数据整合（fMRI、EEG、DTI）
- 🔮 大脑未来状态预测（✅ 新增）
- 📊 增强训练监控系统（✅ 新增）
- 🎯 虚拟刺激和扰动模拟
- 🌐 Unity前端实时可视化
- 📈 异构图神经网络建模

## 📚 文档

### 中文文档（推荐）
- **[系统使用指南](docs/TwinBrain系统使用指南.md)** - 完整的用户指南：安装、配置、训练、预测、Unity集成
- **[新功能说明](docs/NEW_FEATURES.md)** - 最新功能详解（2026-02-01）🆕
- **[优化方向和研究思路](OPTIMIZATION_DIRECTIONS.md)** - 研究方向、优化策略、未来规划

### 项目文档
- [更新历史](CHANGELOG.md) - 版本更新和变更记录

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认配置训练（推荐）
python main.py train --config config/default.yaml
```

### 3. 导出大脑状态（Unity可视化）

```python
from unity_integration import BrainStateExporter

# 创建导出器
exporter = BrainStateExporter(atlas_info, model_version="v4")

# 导出当前状态
brain_state_json = exporter.export_brain_state(
    brain_activity={'fmri': fmri_data, 'eeg': eeg_data},
    output_path="brain_state.json"
)

# 导出时间序列
exporter.export_sequence(
    brain_activity={'fmri': fmri_data},
    output_dir="brain_sequence/",
    start=0, end=200, step=5
)
```

### 4. 模拟虚拟刺激

```python
from unity_integration import StimulationSimulator, StimulationConfig

# 创建模拟器
simulator = StimulationSimulator(n_regions=200)

# 配置刺激
config = StimulationConfig(
    target_regions=[10, 15, 20],
    amplitude=0.5,
    duration=10,
    pattern="sine",
    frequency=10.0
)

# 模拟响应
trajectory, metrics = simulator.simulate_response(
    initial_state, config, n_steps=50
)
```

详细说明请参考 [系统使用指南](docs/TwinBrain系统使用指南.md)

## 💡 核心功能

### 1. 多模态数据处理
- fMRI: 功能磁共振成像，空间分辨率高
- EEG: 脑电图，时间分辨率高
- DTI: 扩散张量成像，结构连接

### 2. 异构图神经网络
- 节点：不同模态的脑区
- 边：结构连接和功能连接
- 跨模态信息传递

### 3. 未来状态预测
- 单步和多步预测
- 基于神经动力学的预测
- 条件化预测（考虑刺激）

### 4. 虚拟刺激模拟
- 多种刺激模式：脉冲、正弦、渐变、连续
- 空间扩散效应
- 网络传播效应
- 反向设计刺激方案

### 5. Unity可视化
- JSON格式导出
- WebSocket实时通信
- 脑区活动热图
- 连接强度可视化

## 🎯 应用场景

- **意识研究**: 量化意识水平，探索意识神经关联物
- **神经调控**: 设计TMS/tACS刺激方案
- **药物评估**: 模拟药物对大脑的影响
- **脑机接口**: 预测大脑响应，优化接口参数
- **临床诊断**: 识别异常脑活动模式
- **教学演示**: 3D可视化大脑活动

## 📖 详细文档

完整的使用说明、API文档、配置参数详解，请参考：

- [系统使用指南](docs/TwinBrain系统使用指南.md) - 从安装到高级功能
- [优化方向和研究思路](OPTIMIZATION_DIRECTIONS.md) - 研究方向和技术路线

## 🔬 研究方向

我们正在探索以下方向（详见[优化方向文档](OPTIMIZATION_DIRECTIONS.md)）：

1. **增强预测能力**
   - 多步未来预测
   - 物理约束的预测
   - 条件预测

2. **因果推断**
   - Granger因果分析
   - 有效连接计算
   - 网络拓扑分析

3. **意识计算**
   - 整合信息理论(IIT)
   - 全局工作空间理论
   - 神经振荡分析

4. **模型优化**
   - 知识蒸馏
   - 模型剪枝
   - 量化加速

## 📊 项目结构（新架构）

```
twinbrain/
├── main.py                    # 统一入口点
├── config/                    # 配置文件
│   ├── default.yaml           # v4 默认配置
│   ├── v3_legacy.yaml         # v3 兼容配置
│   └── export.yaml            # 导出配置
├── workflows/                 # 工作流模块
│   ├── training.py            # 训练流程
│   └── export_latent.py       # 导出流程
├── unity_integration/         # Unity集成模块（新）
│   ├── brain_state_exporter.py    # JSON导出
│   ├── stimulation_simulator.py   # 刺激模拟
│   └── realtime_server.py         # WebSocket服务器
├── mapper/                    # 数据映射
│   ├── atlas_mapper.py
│   ├── bids_mapper.py
│   ├── eeg_mapper.py
│   └── multi_modal_mapper.py
├── train/                     # 训练模块
│   ├── dynamic_hetero_gnn.py  # 模型定义
│   ├── hetero_trainer.py      # 训练器
│   └── aligner.py             # 对齐器
├── utils/                     # 工具模块
│   ├── config.py              # 配置管理
│   ├── logging_utils.py       # 日志系统
│   └── analysis.py            # 分析工具
└── preprocess/                # 预处理
    ├── eeg_preprocessor.py
    └── fmri_preprocessor.py
```

## 🔧 配置说明

### 配置文件结构

配置文件使用 YAML 格式，包含以下主要部分：

```yaml
# 版本信息
version: "v4"
description: "配置说明"

# 训练参数
training:
  warmup_epochs: 5
  finetune_epochs: 80
  learning_rate: 0.0001

# 模型架构
model:
  hidden_dim: 128
  decoder_layers: 3

# 损失函数权重
loss:
  recon_weight: 1.0
  temp_weight: 5.0

# 更多配置...
```

### 版本差异

#### default.yaml（v4 - 推荐）
- 更长的微调周期（80 epochs）
- 更强的时间对齐权重（temp_weight=5.0）
- 更深的解码器（3层）
- 扩展的预热期（10 epochs）

#### v3_legacy.yaml（遗留）
保留用于向后兼容和实验复现。建议新实验使用 v4 配置。

## 🎯 新架构优势

### 1. 配置驱动
- ✅ 超参数与代码分离
- ✅ 易于实验和调参
- ✅ 版本化配置管理

### 2. 模块化设计
- ✅ 清晰的工作流划分
- ✅ 可重用的组件
- ✅ 易于扩展和维护

### 3. 改进的日志系统
- ✅ 彩色控制台输出
- ✅ 详细的文件日志
- ✅ 阶段计时和错误追踪

### 4. 统一入口点
- ✅ 一致的命令行接口
- ✅ 支持多种工作流
- ✅ Dry-run 模式验证配置

## 🔧 版本说明（旧架构）

### main_v4.py（推荐）
- 更长的微调周期（80 epochs）
- 更强的时间对齐权重（temp_weight=5.0）
- 更深的解码器（3层）
- 扩展的预热期（10 epochs）

### main_v3.py（已移除）
旧版本已在重构中移除。如需使用v3配置，请使用新架构：
```bash
python main.py train --config config/v3_legacy.yaml
```

## 📝 最近更新

查看 [CHANGELOG.md](CHANGELOG.md) 了解详细的版本更新和变更记录。

### 重大更新（2026-02-01）

#### 代码清理
- ✅ **移除废弃代码** - 删除 stim_align.py 及相关调用
- ✅ **简化训练流程** - Stimulus 数据处理改为可选

#### 文档重组
- ✅ **清理冗余文档** - 移除重构相关的临时文档
- ✅ **统一文档结构** - 3类核心文档，清晰明了
- ✅ **创建 docs/ 文件夹** - 用户指南独立目录

### 重大更新（2026-01-30）

#### 完整重构 - 新架构
- ✅ **移除 main_v3.py** - 旧版本代码已清理
- ✅ **配置系统** - 实现 YAML 配置管理
- ✅ **统一入口** - 创建 main.py 统一入口点
- ✅ **工作流模块化** - workflows/ 包含独立工作流
- ✅ **日志系统** - 彩色输出和详细文件日志
- ✅ **向后兼容** - 保留 main_v4.py 和 main_export_latent.py

#### 之前的改进
- ✅ 修复了3个致命运行时错误
- ✅ 修复了2个逻辑错误
- ✅ 消除了130+行重复代码
- ✅ 添加了完善的文档和弃用警告
- ✅ 净减少514行代码

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[许可证信息待添加]

## 📧 联系方式

[联系方式待添加]
