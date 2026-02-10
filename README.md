# TwinBrain - 数字孪生脑系统

基于多模态脑成像数据的数字孪生脑系统，用于大脑活动建模、预测和Unity 3D可视化。

## 📚 文档

本项目包含4个核心文档：

1. **[使用指南.md](使用指南.md)** - 面向无编程基础用户的完整操作指南
2. **[Unity一键使用指南.md](Unity一键使用指南.md)** - Unity集成和可视化详细教程
3. **[模型说明.md](模型说明.md)** - 模型架构、原理和技术细节
4. **[更新日志.md](更新日志.md)** - 版本更新记录和新功能说明

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 安装依赖
pip install -r requirements.txt
```

### 一键生成Unity项目

```bash
# 自动生成完整Unity项目结构
python setup_unity_workflow.py --auto-setup
```

### 使用FreeSurfer数据（可选）

```bash
# 将FreeSurfer文件放入指定位置后
python setup_unity_workflow.py --auto-setup
```

## 📖 详细说明

请查看 **[使用指南.md](使用指南.md)** 获取完整的分步教程。

## ⚡ 主要功能

- ✅ 多模态脑数据集成（fMRI、EEG、DTI）
- ✅ 基于异构图神经网络的大脑建模
- ✅ FreeSurfer表面数据支持
- ✅ Unity 3D实时可视化
- ✅ 数百个独立脑区OBJ模型生成（支持脑膜模拟）
- ✅ 大脑状态预测和刺激模拟
- ✅ WebSocket实时通信
- ✅ 完全自动化的工作流

## 🛠️ 系统要求

- Python 3.8+
- Unity 2017.1+ (推荐 2019/2020 LTS)
- 4GB+ RAM
- （可选）FreeSurfer表面数据文件

## 📦 项目结构

```
twinbrain/
├── unity_integration/     # Unity集成模块
│   ├── obj_generator.py   # OBJ 3D模型生成（支持批量导出）
│   ├── freesurfer_loader.py  # FreeSurfer数据加载
│   ├── brain_state_exporter.py  # JSON数据导出
│   └── workflow_manager.py  # 工作流管理
├── unity_examples/        # Unity C#示例脚本
│   ├── BrainVisualization.cs
│   ├── BrainConfigLoader.cs
│   ├── BrainDataStructures.cs
│   └── WebSocketClient.cs
├── mapper/                # 脑图谱和数据映射
├── workflows/             # 训练和预处理工作流
└── setup_unity_workflow.py  # 一键式Unity项目设置
```

## 🎯 使用场景

1. **神经科学研究** - 可视化和分析大脑活动数据
2. **医学教育** - 交互式大脑结构和功能展示
3. **脑机接口** - 实时大脑状态监测和预测
4. **认知建模** - 数字孪生脑仿真和实验

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

## 📄 许可

[添加许可信息]

## 🔗 相关链接

- [GitHub仓库](https://github.com/sheinclotho/twinbrain)
- [问题反馈](https://github.com/sheinclotho/twinbrain/issues)
- [讨论区](https://github.com/sheinclotho/twinbrain/discussions)

---

*最后更新: 2024-02*
