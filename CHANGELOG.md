# TwinBrain 更新历史

## 2026-02-01

### 新功能实现 🎉

#### 增强训练监控
- **MetricsTracker**: 完整的训练指标追踪系统
  - 自动记录所有损失分量
  - 保存指标历史到 JSON
  - 生成训练摘要报告
- **TrainingMonitor**: 训练进度监控
  - 检测训练停滞
  - 检测异常值（NaN/Inf）
  - 自动生成警告

#### 多步未来预测
- **PredictorHead**: GRU + 注意力机制的预测模块
  - 支持多步未来状态预测
  - 自回归预测机制
  - 可配置预测步数
- **ConditionalPredictor**: 条件化预测器
  - 支持基于外部刺激的预测
  - 适用于 TMS/tACS 效应模拟

#### 配置系统增强
- 新增 `prediction` 配置节
  - `enabled`: 启用/禁用预测
  - `steps`: 预测步数
  - `weight`: 预测损失权重
- 新增 `metrics` 配置节
  - `enabled`: 启用/禁用指标追踪
  - `output_dir`: 指标保存目录

### 代码优化
- 集成 PredictorHead 到 DynamicHeteroTrainer
- 集成 MetricsTracker 到训练流程
- 更新 workflows/training.py 以支持新功能

### 文档更新
- 更新 OPTIMIZATION_DIRECTIONS.md，标记已实现的优化
- 添加详细的使用说明和代码示例
- 更新文档版本到 1.1

### 代码清理
- 移除废弃的 stim_align.py 模块及相关代码调用
- 简化训练工作流，stimulus 数据处理改为可选

### 文档重组
- 整合文档结构，保留3类核心文档
- 创建 docs/ 文件夹存放用户指南
- 统一优化方向文档命名

## 2026-01-30

### 重大重构
- 完成代码架构重构，新增约900行代码，删除约540行冗余代码
- 创建基于YAML的配置系统（utils/config.py）
- 实现统一日志系统（utils/logging_utils.py）
- 模块化训练和导出工作流（workflows/）
- 创建统一入口 main.py，支持命令行参数

### 新功能
- Unity前端集成模块
  - brain_state_exporter.py: JSON格式脑状态导出
  - stimulation_simulator.py: 虚拟刺激模拟器（4种刺激模式）
  - realtime_server.py: WebSocket实时服务器
- 数据缓存管理
- 诊断系统集成
- 配置驱动的训练流程

### 文档完善
- 创建完整的中文系统使用指南（25,000+字）
- 创建优化方向和研究思路文档（43,000+字）
- 更新 README.md 和 README_CN.md

## 历史版本

### v3 (Legacy)
- 基础训练功能
- 多模态数据处理
- 异构图神经网络实现
