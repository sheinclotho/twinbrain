# TwinBrain - 数字孪生脑系统

基于多模态脑成像数据（fMRI、EEG、DTI）的数字孪生脑系统，使用异构图神经网络进行脑信号重建和分析。

## 📚 文档

- **[重构总结文档](REFACTORING_SUMMARY_CN.md)** - 详细的代码重构说明、设计改进建议和架构优化方案（中文）

## 🚀 快速开始

### 依赖安装

```bash
pip install -r requirements.txt
```

### 使用方法

**训练模型（推荐使用v4）:**
```bash
python main_v4.py
```

**训练模型（v3版本，用于实验复现）:**
```bash
python main_v3.py
```

**导出潜在表征:**
```bash
python main_export_latent.py
```

## 📊 项目结构

```
twinbrain/
├── main_v3.py              # 训练主程序 v3（已弃用）
├── main_v4.py              # 训练主程序 v4（推荐）
├── main_export_latent.py   # 潜在表征导出
├── mapper/                 # 数据映射模块
│   ├── atlas_mapper.py
│   ├── bids_mapper.py
│   ├── eeg_mapper.py
│   └── multi_modal_mapper.py
├── train/                  # 训练模块
│   ├── hetero_trainer.py
│   ├── dynamic_hetero_gnn.py
│   └── aligner.py
├── utils/                  # 工具模块
│   ├── utils.py           # 通用工具
│   ├── function.py        # 数据处理函数
│   ├── analysis.py        # 分析工具
│   └── debug.py           # 调试工具
└── preprocess/            # 预处理模块
    ├── eeg_preprocessor.py
    └── fmri_preprocessor.py
```

## 🔧 版本说明

### main_v4.py（推荐）
- 更长的微调周期（80 epochs）
- 更强的时间对齐权重（temp_weight=5.0）
- 更深的解码器（3层）
- 扩展的预热期（10 epochs）

### main_v3.py（已弃用）
保留用于向后兼容和实验复现。建议新实验使用 v4 版本。

## 📝 最近更新

查看 [REFACTORING_SUMMARY_CN.md](REFACTORING_SUMMARY_CN.md) 了解详细的重构内容和改进建议。

### 主要改进
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
