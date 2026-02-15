# TwinBrain - 数字孪生脑系统

基于多模态脑成像数据的数字孪生脑系统，用于大脑活动建模、预测和Unity 3D可视化。

## 📚 文档

本项目包含完整的文档集：

### 使用指南（根目录）
1. **[使用指南.md](使用指南.md)** - 面向无编程基础用户的完整操作指南
2. **[Unity使用指南.md](Unity使用指南.md)** - Unity可视化完整使用指南（基于一键安装脚本）⭐⭐

### 技术文档（根目录）
4. **[模型说明.md](模型说明.md)** - 模型架构、原理和技术细节
5. **[MODEL_FORMAT.md](MODEL_FORMAT.md)** - 模型和缓存文件格式规范 ⭐
6. **[PERFORMANCE.md](PERFORMANCE.md)** - 性能优化指南 ⭐
7. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 问题排查指南 ⭐
8. **[项目规范说明书.md](项目规范说明书.md)** - 项目开发规范

### 项目更新记录（docs/目录）
9. **[docs/更新日志.md](docs/更新日志.md)** - 版本更新记录和新功能说明
10. **[docs/REORGANIZATION_SUMMARY.md](docs/REORGANIZATION_SUMMARY.md)** - 项目重组总结
11. **[docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md)** - 本次优化总结

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
# 基本使用：创建项目结构
python setup_unity_project.py --auto-setup

# 使用FreeSurfer数据自动生成OBJ模型
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer/files

# 自动安装到Unity项目（v2.5新增）⭐
python unity_package_installer.py --unity-project /path/to/UnityProject

# 启动Unity后端服务（支持实时预测和刺激）
python unity_startup.py --demo
```

## 📖 详细说明

请查看 **[使用指南.md](使用指南.md)** 获取完整的分步教程。

## ⚡ 主要功能

- ✅ 多模态脑数据集成（fMRI、EEG、DTI）
- ✅ 基于异构图神经网络的大脑建模
- ✅ FreeSurfer表面数据支持
- ✅ Unity 3D实时可视化（Unity 2019+兼容）
- ✅ 数百个独立脑区OBJ模型生成（支持脑膜模拟）
- ✅ 点击交互选择脑区
- ✅ 虚拟刺激输入和模拟
- ✅ 大脑状态预测和实时反馈
- ✅ 颜色映射区分真实/预测信号
- ✅ HTTP/REST实时通信（v2.5改进）
- ✅ 完全自动化的工作流
- ✅ 一键式Unity项目设置和安装（v2.5新增）⭐

## 🛠️ 系统要求

- Python 3.8+
- Unity 2017.1+ (推荐 2019/2020 LTS)
- 4GB+ RAM
- （可选）FreeSurfer表面数据文件

## 📦 项目结构

```
twinbrain/
├── docs/                       # 项目更新和重组文档
│   ├── 更新日志.md (v2.5更新)
│   ├── REORGANIZATION_SUMMARY.md
│   └── OPTIMIZATION_SUMMARY.md
├── unity_integration/          # Unity集成模块
│   ├── obj_generator.py        # OBJ 3D模型生成
│   ├── freesurfer_loader.py    # FreeSurfer数据加载
│   ├── brain_state_exporter.py # JSON数据导出
│   ├── realtime_server.py      # WebSocket服务器（v2.5增强）
│   ├── model_server.py         # 模型服务器
│   └── workflow_manager.py     # 工作流管理
├── unity_examples/             # Unity C#脚本示例
│   ├── WebSocketClient.cs      # 原版客户端
│   ├── WebSocketClientImproved.cs # 改进版客户端（v2.5新增）⭐
│   ├── BrainVisualization.cs   # 主可视化脚本
│   ├── CacheToJsonConverter.cs # Cache转换工具（v2.5优化）
│   └── ...                     # 其他C#脚本
├── mapper/                     # 脑图谱和数据映射
├── workflows/                  # 训练和预处理工作流
├── setup_unity_project.py      # Unity项目设置（初始化）
├── unity_startup.py            # Unity后端启动（v2.5增强）
└── unity_package_installer.py  # Unity包安装器（v2.5新增）⭐
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

*最后更新: 2024-02-14 (v2.5)*
