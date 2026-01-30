# TwinBrain - 数字孪生脑系统

基于多模态脑成像数据（fMRI、EEG、DTI）的数字孪生脑系统，使用异构图神经网络进行脑信号重建和分析。

**最新更新**: 项目已完成完整重构，采用配置驱动的模块化架构。

## 📚 文档

- **[重构总结文档](REFACTORING_SUMMARY_CN.md)** - 详细的代码重构说明、设计改进建议和架构优化方案（中文）

## 🚀 快速开始

### 依赖安装

```bash
pip install -r requirements.txt
```

### 新版本使用方法（推荐）

**使用默认配置训练（v4 推荐）:**
```bash
python main.py train --config config/default.yaml
```

**使用 v3 配置复现旧实验:**
```bash
python main.py train --config config/v3_legacy.yaml
```

**导出潜在表征:**
```bash
python main.py export --config config/export.yaml --subject sub-01
```

**查看配置（不运行）:**
```bash
python main.py train --config config/default.yaml --dry-run
```

### 旧版本使用方法（兼容）

**训练模型（v4）:**
```bash
python main_v4.py
```

**导出潜在表征:**
```bash
python main_export_latent.py
```

## 📊 项目结构（新架构）

```
twinbrain/
├── main.py                 # 统一入口点（新）
├── main_v4.py              # 训练主程序（旧版，仍可用）
├── main_export_latent.py   # 潜在表征导出（旧版）
├── config/                 # 配置文件（新）
│   ├── default.yaml        # v4 默认配置
│   ├── v3_legacy.yaml      # v3 遗留配置
│   └── export.yaml         # 导出配置
├── workflows/              # 工作流模块（新）
│   ├── training.py         # 训练工作流
│   └── export_latent.py    # 导出工作流
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
│   ├── analysis.py        # 分析工具（新）
│   ├── config.py          # 配置管理（新）
│   ├── logging_utils.py   # 日志系统（新）
│   └── debug.py           # 调试工具
└── preprocess/            # 预处理模块
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

查看 [REFACTORING_SUMMARY_CN.md](REFACTORING_SUMMARY_CN.md) 了解详细的重构内容和改进建议。

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
