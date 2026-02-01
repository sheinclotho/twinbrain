# TwinBrain 更新历史

## 2026-02-01

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
