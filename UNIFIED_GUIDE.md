# TwinBrain 完整使用指南

> **版本**: v4.1 (2024-02-15)  
> **一体化指南**: 从安装到使用的完整流程，包含自动化虚拟刺激

---

## 📋 目录

1. [系统要求](#系统要求)
2. [完整安装流程](#完整安装流程)
3. [模型准备与训练](#模型准备与训练)
4. [Unity项目设置](#unity项目设置)
5. [后端服务配置](#后端服务配置)
6. [虚拟刺激使用（全自动）](#虚拟刺激使用全自动)
7. [高级功能](#高级功能)
8. [故障排除](#故障排除)

---

## 系统要求

### 必需软件
- **Python 3.8+** - 后端计算和服务器
- **Unity 2019.1+** - 3D可视化（推荐2020 LTS或2021 LTS）
- **8GB+ RAM** - 推荐16GB用于大规模数据

### 可选组件
- **FreeSurfer 7.0+** - 生成真实脑表面3D模型
- **训练好的模型文件** - 用于实时预测（或使用演示模式）
- **GPU** - 加速模型训练和推理

---

## 完整安装流程

### 第一步：获取代码

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 安装Python依赖
pip install -r requirements.txt
```

**依赖说明**:
- `torch` - PyTorch深度学习框架
- `numpy` - 数值计算
- `nibabel` - 处理神经影像数据（可选，用于FreeSurfer）
- `websockets` - WebSocket通信（可选，用于实时模式）

### 第二步：创建Unity项目

在Unity Hub中创建新的3D项目：
- 项目名称：`TwinBrainDemo`（或任意名称）
- 模板：3D
- Unity版本：2019.1+

### 第三步：运行一键安装

```bash
# 完整安装（带FreeSurfer数据生成3D模型）
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo \
    --freesurfer-dir /path/to/freesurfer

# 或基础安装（使用默认球体模型）
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo
```

**这个脚本会自动**:
- ✅ 生成OBJ脑区模型（如提供FreeSurfer）
- ✅ 复制所有C#脚本到Unity项目
- ✅ 安装Unity Editor自动化工具
- ✅ 创建标准目录结构
- ✅ 配置项目依赖

**输出目录结构**:
```
unity_project/                          # 中间文件目录
├── freesurfer_files/                   # FreeSurfer数据
├── brain_data/
│   ├── cache/                          # 预处理缓存 (.pt格式)
│   ├── model_output/                   # JSON状态文件
│   │   ├── predictions/                # 预测结果（自动生成）
│   │   └── stimulation/                # 刺激结果（自动生成）
│   └── original/                       # 原始数据
└── Unity_Assets/
    ├── Scripts/                        # C#脚本源文件
    └── obj/                            # 生成的OBJ文件

TwinBrainDemo/                          # Unity项目
└── Assets/
    ├── TwinBrain/
    │   ├── Scripts/                    # 已安装的C#脚本
    │   └── Editor/                     # Unity Editor工具
    └── StreamingAssets/
        └── OBJ/                        # 已复制的OBJ文件
```

### 第四步：Unity内自动设置

1. **打开Unity项目** - 在Unity Hub中打开刚创建的项目
2. **等待Newtonsoft.Json包安装** - Unity会自动下载（1-2分钟）
3. **运行自动设置工具**:
   - 菜单: **TwinBrain → 自动设置场景**
   - 勾选所有选项（包括"创建虚拟刺激UI"）
   - 点击"开始自动设置"
   - 等待完成（约30秒-2分钟）

**自动设置完成的内容**:
- ✅ 导入并配置所有OBJ文件（200+个，自动设置缩放）
- ✅ 创建BrainManager GameObject
- ✅ 添加BrainVisualization组件（启用文件监控）
- ✅ 添加WebSocketClient组件（后端通信）
- ✅ 添加StimulationInput组件（刺激控制）
- ✅ 创建完整虚拟刺激UI（左下角面板）
- ✅ 配置摄像机
- ✅ 自动连接所有组件引用

---

## 模型准备与训练

### 选项1: 使用演示模式（无需模型）

如果只是测试或演示，可以跳过模型训练，直接使用演示模式：

```bash
python unity_startup.py --demo
```

演示模式会生成随机的大脑活动数据用于可视化。

### 选项2: 训练自己的模型

#### 准备训练数据

将脑成像数据放置在标准位置：

```
unity_project/brain_data/original/
├── fmri/                               # fMRI数据
│   └── sub-01_task-rest_bold.nii.gz   # NIfTI格式
├── eeg/                                # EEG数据
│   └── sub-01_task-rest_eeg.set       # EEGLAB格式
└── dti/                                # DTI连接矩阵
    └── sub-01_connectivity.npy         # NumPy格式
```

#### 运行训练

```bash
# 基础训练（使用默认参数）
python main.py

# 高级训练（自定义参数）
python main.py \
    --data_dir unity_project/brain_data/original \
    --output_dir results \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

#### 模型文件位置

训练完成后，模型文件会保存在：

```
results/
├── hetero_gnn_trained.pt               # 训练好的模型（主要文件）⭐
├── best_model.pt                       # 最佳模型检查点
├── training_log.txt                    # 训练日志
└── metrics.json                        # 训练指标
```

**重要**: 后端服务器会在以下位置查找模型文件：
1. 命令行指定的路径: `--model /path/to/model.pt`
2. `results/hetero_gnn_trained.pt`（默认）
3. `test_file3/sub-01/results/hetero_gnn_trained.pt`

**推荐做法**: 将训练好的模型放在 `results/hetero_gnn_trained.pt`，这样启动服务器时无需指定路径。

### 选项3: 使用预训练模型

如果有预训练模型，将其放置在：

```bash
mkdir -p results
cp /path/to/your_model.pt results/hetero_gnn_trained.pt
```

---

## Unity项目设置

### 配置BrainVisualization组件

选择Hierarchy中的**BrainManager**，在Inspector中配置：

#### File Settings（文件设置）
- **Json Path**: `brain_data/model_output`（留空，自动加载）
- **Load Sequence**: ✓（加载序列）

#### Model Settings（模型设置）
- **Use Obj Models**: ✓（使用OBJ模型）
- **Obj Directory**: `StreamingAssets/OBJ`

#### Auto-Reload Settings（自动重载设置）⭐
- **Enable Auto Reload**: ✓（启用，默认）
- **Watch Directory**: `unity_project/brain_data/model_output`
- **Watch Interval**: 2.0秒
- **Auto Load Type**: `both`（监控预测和刺激）

#### Visualization Settings（可视化设置）
- **Region Scale**: 1.0
- **Activity Threshold**: 0.3
- **Show Connections**: ✓（根据需要）
- **Connection Threshold**: 0.5

#### Colors（颜色设置）
- **Low Activity Color**: 蓝色
- **High Activity Color**: 红色

### 虚拟刺激UI组件

左下角面板包含：
- **目标脑区输入框** - 输入脑区ID（逗号分隔，如：1,2,3）
- **振幅滑块** - 0.0到5.0范围
- **振幅显示** - 显示当前值
- **刺激模式下拉菜单** - constant/sine/pulse/ramp
- **应用刺激按钮** - 发送刺激请求
- **状态文本** - 显示操作状态

---

## 后端服务配置

### 启动后端服务器

#### 基础启动（使用默认模型）

```bash
cd /path/to/twinbrain

# 自动查找模型（在results/目录）
python unity_startup.py
```

#### 指定模型路径

```bash
# 使用特定模型文件
python unity_startup.py --model /path/to/your_model.pt

# 或使用相对路径
python unity_startup.py --model results/hetero_gnn_trained.pt
```

#### 演示模式（无模型）

```bash
python unity_startup.py --demo
```

#### 自定义配置

```bash
python unity_startup.py \
    --model results/hetero_gnn_trained.pt \
    --output unity_project \
    --host 0.0.0.0 \
    --port 8765
```

### 服务器启动验证

成功启动后，应看到：

```
============================================================
启动TwinBrain WebSocket服务器
============================================================
✓ PyTorch x.x.x
✓ NumPy x.x.x
✓ websockets x.x.x
模型加载: results/hetero_gnn_trained.pt
模型文件大小: X.XX MB
✓ 模型加载成功
✓ 状态导出器: unity_project/brain_data/model_output
✓ 刺激模拟器: 200个脑区

🚀 服务器启动: ws://0.0.0.0:8765
等待Unity客户端连接...
按 Ctrl+C 停止服务器
```

### 后端服务器功能

服务器自动提供以下功能：

1. **状态查询**: 获取当前大脑状态
2. **预测**: 预测未来N步的大脑活动
   - 自动保存到: `model_output/predictions/pred_YYYYMMDD_HHMMSS/`
3. **刺激模拟**: 计算虚拟刺激响应
   - 自动保存到: `model_output/stimulation/stim_YYYYMMDD_HHMMSS/`
4. **流式传输**: 实时流式传输大脑活动

**自动化特性**:
- ✅ 结果自动导出为JSON（50帧）
- ✅ 自动创建sequence_index.json索引文件
- ✅ 时间戳命名避免覆盖
- ✅ Unity自动检测并加载新结果（2秒轮询）

---

## 虚拟刺激使用（全自动）

### 完整自动化工作流

**旧版本需要9个步骤，新版本只需2步！**

#### 准备工作（一次性，已完成）

- ✅ 后端服务器已启动
- ✅ Unity项目已打开并运行（点击Play）
- ✅ 看到"Connected to server"在Console中

#### 使用虚拟刺激

1. **输入刺激参数**
   - 在左下角面板的"目标脑区"输入框输入：`1,5,10`
   - 调整振幅滑块：设置为 `1.5`
   - 选择刺激模式：`sine`（正弦波）

2. **点击"应用刺激"按钮**

3. **等待2-5秒，结果自动显示！** ✨

**自动化流程（后台）**:
```
Unity UI → StimulationInput → WebSocketClient 
    ↓
后端接收参数 → StimulationSimulator计算响应（50帧）
    ↓
自动保存到: model_output/stimulation/stim_20240215_143022/
    ├── frame_0000.json
    ├── frame_0001.json
    ├── ...
    ├── frame_0049.json
    └── sequence_index.json
    ↓
Unity文件监控检测到新目录（2秒内）
    ↓
自动加载JSON序列 → 自动播放动画
```

### 无需手动操作

**完全消除的手动步骤**:
- ❌ ~~手动保存JSON文件~~
- ❌ ~~手动运行转换脚本~~
- ❌ ~~手动移动文件到Unity目录~~
- ❌ ~~手动点击刷新按钮~~
- ❌ ~~手动点击加载数据按钮~~

### 刺激模式说明

- **constant** - 恒定刺激，持续施加相同强度
- **sine** - 正弦波刺激，模拟tACS（经颅交流电刺激）
- **pulse** - 脉冲刺激，短暂激活
- **ramp** - 渐增刺激，逐渐增加强度

### 查看刺激历史

所有刺激结果保存在：
```
unity_project/brain_data/model_output/stimulation/
├── stim_20240215_143022/    # 第一次刺激
├── stim_20240215_143156/    # 第二次刺激
└── stim_20240215_143301/    # 第三次刺激
```

每个目录包含：
- 50个JSON帧文件（frame_0000.json ~ frame_0049.json）
- 1个索引文件（sequence_index.json）
- 刺激参数记录在索引文件中

---

## 高级功能

### 预测功能

通过WebSocketClient发送预测请求（需要编写自定义脚本或使用其他工具）：

```json
{
  "type": "predict",
  "n_steps": 20
}
```

后端会：
1. 预测未来20步
2. 自动保存到 `model_output/predictions/pred_YYYYMMDD_HHMMSS/`
3. Unity自动检测并加载

### 离线可视化

如果已有JSON文件：

1. 将JSON文件放在 `Assets/StreamingAssets/`
2. 在BrainVisualization组件设置Json Path
3. 点击Play即可查看

### 批量转换缓存文件

如果有PyTorch缓存文件（.pt/.pth）需要转换：

```bash
python -m unity_integration.brain_state_exporter \
    --cache-dir unity_project/brain_data/cache \
    --output-dir unity_project/brain_data/model_output \
    --format json
```

### 键盘快捷键

在Unity Play模式中：
- **空格键** - 播放/暂停动画
- **R键** - 重新加载当前数据

---

## 故障排除

### 问题1: 后端服务器启动失败

**症状**: 运行`unity_startup.py`时出错

**解决方案**:

1. 检查Python版本
   ```bash
   python --version  # 应该是3.8或更高
   ```

2. 安装缺失的依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 检查端口占用
   ```bash
   # Windows
   netstat -ano | findstr :8765
   
   # Linux/Mac
   lsof -i :8765
   ```

4. 使用不同端口
   ```bash
   python unity_startup.py --port 8766
   ```

### 问题2: Unity找不到TwinBrain菜单

**症状**: Unity菜单栏没有"TwinBrain"选项

**解决方案**:

1. 等待Unity完全加载所有脚本
2. 检查Console是否有编译错误（红色）
3. 确认Newtonsoft.Json已安装：
   - Window → Package Manager → 搜索"Newtonsoft.Json"
4. 重新运行安装脚本：
   ```bash
   python unity_one_click_install.py --unity-project /path/to/Unity
   ```

### 问题3: 刺激结果没有自动加载

**症状**: 点击"应用刺激"后没有自动显示结果

**解决方案**:

1. 检查Auto Reload是否启用
   - BrainManager → Inspector → BrainVisualization
   - Enable Auto Reload: ✓

2. 检查Watch Directory路径
   - 应该指向: `unity_project/brain_data/model_output`
   - 确保路径存在且可访问

3. 查看Console日志
   - 应该看到: "Auto-detected new stimulation: stim_YYYYMMDD_HHMMSS"
   - 如果没有，检查后端是否正常保存文件

4. 手动验证文件
   ```bash
   ls -la unity_project/brain_data/model_output/stimulation/
   ```

5. 调整Watch Interval（如果需要）
   - 增加到5.0秒，给文件系统更多时间

### 问题4: Unity连接不到后端

**症状**: Console显示"Not connected to server"

**解决方案**:

1. 确认后端服务器正在运行
   ```bash
   # 应该看到 "服务器启动: ws://0.0.0.0:8765"
   ```

2. 检查WebSocketClient配置
   - BrainManager → Inspector → WebSocketClient
   - Server URL: `http://localhost:8765`
   - Auto Connect: ✓

3. 检查防火墙设置
   - 确保允许端口8765的连接

4. 尝试使用IP地址
   - 改为 `http://127.0.0.1:8765`

### 问题5: 模型文件找不到

**症状**: 服务器报错"模型文件不存在"

**解决方案**:

1. 确认模型文件位置
   ```bash
   ls -la results/hetero_gnn_trained.pt
   ```

2. 使用绝对路径
   ```bash
   python unity_startup.py --model /absolute/path/to/model.pt
   ```

3. 如果没有模型，使用演示模式
   ```bash
   python unity_startup.py --demo
   ```

### 问题6: OBJ文件导入很慢

**症状**: 自动设置工具运行很久

**解决方案**:

1. 耐心等待（200+个OBJ文件需要1-2分钟）
2. 使用SSD硬盘（显著提升速度）
3. 关闭Unity的自动保存功能（Edit → Preferences → Auto Save）
4. 如果不需要OBJ模型，取消勾选"Use Obj Models"

---

## 性能优化建议

### 1. 减少脑区数量

如果性能较差，使用较少的脑区：
- 100个脑区而非200个
- 修改FreeSurfer标注文件使用粗粒度划分

### 2. 禁用视觉效果

在BrainVisualization组件中：
- Show Connections: ✗（禁用连接线）
- Activity Threshold: 提高到0.5（只显示高活动区域）

### 3. 优化Unity质量设置

- Edit → Project Settings → Quality
- 选择"Low"或"Medium"质量级别
- 禁用抗锯齿

### 4. 使用对象池

对于频繁创建/销毁的对象，使用对象池模式提高性能。

---

## 文件格式说明

### JSON状态文件格式

```json
{
  "version": "2.0",
  "timestamp": "2024-02-15T14:30:22",
  "metadata": {
    "subject": "stimulation",
    "atlas": "Schaefer200",
    "model_version": "4.1",
    "time_point": 0,
    "time_second": 0.0
  },
  "brain_state": {
    "time_point": 0,
    "time_second": 0.0,
    "regions": [
      {
        "id": 0,
        "label": "Visual_1",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "activity": {
          "fmri": {"amplitude": 0.75, "raw_value": 1.5},
          "eeg": {"amplitude": 0.62, "raw_value": 0.8}
        }
      }
    ],
    "connections": [
      {
        "source": 0,
        "target": 1,
        "strength": 0.8,
        "type": "structural",
        "bidirectional": true
      }
    ],
    "global_metrics": {
      "mean_activity": 0.65,
      "std_activity": 0.15,
      "max_activity": 0.95,
      "active_regions": 150
    }
  },
  "stimulation": {
    "active": true,
    "target_regions": [1, 5, 10],
    "amplitude": 1.5,
    "pattern": "sine",
    "frequency": 10.0
  }
}
```

### 索引文件格式

```json
{
  "type": "stimulation_sequence",
  "timestamp": "20240215_143022",
  "stimulation_params": {
    "target_regions": [1, 5, 10],
    "amplitude": 1.5,
    "pattern": "sine",
    "frequency": 10.0,
    "duration": 20
  },
  "n_frames": 50,
  "output_dir": "model_output/stimulation/stim_20240215_143022",
  "files": ["frame_0000.json", "frame_0001.json", "..."]
}
```

---

## 技术支持

### 获取帮助

1. **查看文档**
   - [Unity使用指南.md](Unity使用指南.md) - 详细Unity操作
   - [模型说明.md](模型说明.md) - 模型技术细节
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - 问题排查

2. **检查日志**
   - Unity Console: Ctrl+Shift+C
   - Python终端: 查看服务器输出

3. **提交Issue**
   - GitHub: https://github.com/sheinclotho/twinbrain/issues
   - 提供：系统信息、错误信息、重现步骤

### 常见命令速查

```bash
# 安装
pip install -r requirements.txt
python unity_one_click_install.py --unity-project /path/to/Unity

# 训练模型
python main.py

# 启动后端（有模型）
python unity_startup.py --model results/hetero_gnn_trained.pt

# 启动后端（演示模式）
python unity_startup.py --demo

# 转换缓存
python -m unity_integration.brain_state_exporter \
    --cache-dir unity_project/brain_data/cache \
    --output-dir unity_project/brain_data/model_output
```

---

**版本**: v4.1  
**最后更新**: 2024-02-15  
**作者**: TwinBrain团队
