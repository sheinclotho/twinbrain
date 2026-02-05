# TwinBrain 数字孪生脑系统

基于多模态脑成像数据的数字孪生脑系统，用于大脑活动建模、预测和可视化。

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 安装依赖
pip install -r requirements.txt
```

### 2. Unity可视化（最快开始方式）

```bash
# 一键生成Unity项目（5分钟完成）
python setup_unity_project.py
```

然后按照生成的 `Unity_TwinBrain/README_UNITY.md` 说明操作。

详见：**[Unity一键使用指南](Unity一键使用指南.md)** 📘

### 3. 训练模型

```bash
# 使用默认配置训练
python main.py train --config config/default.yaml
```

详见：**[模型说明](模型说明.md)** 📘

## 📚 核心文档

本项目包含3个核心文档，涵盖所有功能：

1. **[模型说明.md](模型说明.md)** - 模型架构、训练、API使用
   - 模型架构和组件
   - 数据格式说明
   - 训练和预测方法
   - API参考
   - 常见问题

2. **[Unity一键使用指南.md](Unity一键使用指南.md)** - Unity集成完整教程
   - 3步完成Unity设置
   - 详细的分步指南
   - 可视化参数调整
   - 实时连接配置
   - 疑难解答

3. **[更新日志.md](更新日志.md)** - 版本更新和功能路线图
   - 最新更新内容
   - 历史版本记录
   - 未来开发计划

## ⚡ 主要特性

### 🧠 强大的模型
- 异构图神经网络架构
- 多模态数据融合（fMRI、EEG）
- 时序动态建模
- 多步未来预测
- 刺激模拟

### 🎮 Unity可视化
- 一键式自动化设置
- 实时大脑活动可视化
- 交互式参数调整
- WebSocket实时连接
- 支持FreeSurfer数据

### 📊 完整工具链
- 自动化数据预处理
- 训练监控和可视化
- 模型保存和加载
- JSON/OBJ格式导出
- Python API和命令行

## 🎯 使用场景

1. **科研**：大脑活动建模和预测
2. **教学**：大脑功能可视化演示
3. **临床**：神经疾病研究辅助
4. **开发**：脑机接口原型开发

## 🔧 系统要求

### Python环境
- Python 3.8+
- PyTorch 1.10+
- 其他依赖见 requirements.txt

### Unity环境（可视化）
- Unity 2021.3 或更新版本
- Newtonsoft.Json包

### 硬件建议
- GPU：NVIDIA GPU（推荐，训练加速）
- 内存：8GB+（16GB推荐）
- 存储：5GB+

## 📖 快速示例

### 训练模型

```bash
# 基本训练
python main.py train --config config/default.yaml

# 指定输出目录
python main.py train --config config/default.yaml --output results/my_exp

# 使用GPU
python main.py train --config config/default.yaml --device cuda
```

### 预测

```python
from train.trainer import load_model

# 加载模型
model = load_model("results/hetero_gnn_trained.pt")

# 单步预测
next_state = model.predict(current_state)

# 多步预测
future_states = model.predict_multi_step(current_state, n_steps=10)
```

### Unity导出

```bash
# 一键生成Unity项目
python setup_unity_project.py

# 导出自定义数据
python -m unity_integration.brain_state_exporter \
    --model results/hetero_gnn_trained.pt \
    --output unity_data
```

### 实时服务器

```bash
# 启动WebSocket服务器
python -m unity_integration.realtime_server
```

## 📁 项目结构

```
twinbrain/
├── config/              # 配置文件
├── train/               # 训练模块
├── preprocess/          # 数据预处理
├── unity_integration/   # Unity集成
├── unity_examples/      # Unity C#脚本
├── utils/               # 工具函数
├── setup_unity_project.py  # Unity一键设置
├── 模型说明.md          # 模型文档
├── Unity一键使用指南.md  # Unity教程
└── 更新日志.md          # 更新记录
```

## 🎓 学习路径

### 新手入门
1. 阅读本README了解项目
2. 运行 `python setup_unity_project.py` 体验可视化
3. 按照Unity一键使用指南在Unity中查看效果
4. 了解基本概念和术语

### 进阶使用
1. 阅读模型说明，理解模型架构
2. 使用自己的数据训练模型
3. 调整超参数优化模型
4. 自定义可视化效果

### 高级开发
1. 修改模型架构
2. 实现自定义数据加载器
3. 扩展Unity脚本功能
4. 开发新的分析工具

## ❓ 常见问题

### Q: 我没有GPU可以使用吗？
A: 可以，但训练会较慢。可以使用CPU模式：`--device cpu`

### Q: 我没有脑数据怎么办？
A: 项目自动生成示例数据，可以先用来学习和测试。

### Q: Unity可视化需要编程吗？
A: 不需要！使用 `setup_unity_project.py` 自动生成所有文件，按说明操作即可。

### Q: 支持哪些脑图谱？
A: Schaefer（100/200/400区）、AAL、Destrieux等标准图谱，也支持FreeSurfer个性化图谱。

### Q: 可以商用吗？
A: 本项目采用MIT许可证，可以自由使用，包括商业用途。

更多问题参见各核心文档的FAQ部分。

## 🤝 贡献

欢迎贡献代码、文档、bug报告或功能建议！

1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 📞 联系与支持

- **GitHub Issues**: https://github.com/sheinclotho/twinbrain/issues
- **邮件**: 通过GitHub Issues联系
- **文档**: 查看3个核心文档
- **示例**: 项目中的 example_*.py 文件

## 🌟 致谢

感谢所有贡献者和用户的支持！

## 🔗 相关资源

- PyTorch: https://pytorch.org/
- Unity: https://unity.com/
- FreeSurfer: https://surfer.nmr.mgh.harvard.edu/

---

**记住**：有问题先查看3个核心文档，90%的问题都能找到答案！

最后更新: 2024-02-05
