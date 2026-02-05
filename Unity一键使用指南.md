# TwinBrain Unity 一键使用指南

## 🎯 目标

基于真实FreeSurfer数据和实际脑数据，完成Unity可视化设置。

## 📋 前提条件

1. **FreeSurfer文件**（可选，但推荐）
   - lh.pial, rh.pial（表面文件）
   - lh.Schaefer2018_200Parcels_7Networks_order.annot
   - rh.Schaefer2018_200Parcels_7Networks_order.annot

2. **实际脑数据**（可选）
   - fMRI数据（NIfTI格式）
   - 或预处理后的时间序列

## ⚡ 快速开始

### 阶段1: 构建（一次性，使用FreeSurfer）

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 如果有FreeSurfer文件，生成真实OBJ模型
python setup_unity_project.py \
    --freesurfer-dir /path/to/freesurfer/files \
    --atlas Schaefer200
```

如果没有FreeSurfer文件：
```bash
# 创建基本项目结构
python setup_unity_project.py

# 然后将FreeSurfer文件放入生成的freesurfer/文件夹
# 再次运行生成OBJ模型
```

**这会生成：**
- ✅ 200个脑区的真实3D OBJ模型（从FreeSurfer）
- ✅ 正确的文件夹结构（freesurfer/, data/, state/）
- ✅ Unity C#脚本
- ✅ 配置文件和说明文档

### 阶段2: 准备数据

```bash
# 将实际脑数据放入data/文件夹
cp your_fmri_data.nii Unity_TwinBrain/data/raw/

# 或将预处理数据放入cache/
cp preprocessed_data.pkl Unity_TwinBrain/data/cache/

# 处理数据生成JSON状态文件
python -m unity_integration.brain_state_exporter \
    --data-dir Unity_TwinBrain/data \
    --output Unity_TwinBrain/state \
    --atlas Schaefer200
```

### 阶段3: Unity设置

#### 1. 创建Unity项目

1. 打开Unity Hub
2. 创建新的3D项目
3. 命名（例如：TwinBrain_Demo）

#### 2. 导入资源

**C#脚本：**
- 将 `Unity_TwinBrain/Scripts/` 复制到 `Assets/Scripts/`

**3D模型：**
- 将 `Unity_TwinBrain/OBJ/` 复制到 `Assets/Models/`（如果有）

**数据文件：**
- 在 `Assets/` 下创建 `StreamingAssets` 文件夹
- 将 `Unity_TwinBrain/state/` 复制到 `Assets/StreamingAssets/`
- 将 `unity_config.json` 复制到 `Assets/StreamingAssets/`

#### 3. 安装依赖

**Newtonsoft.Json:**
1. Window > Package Manager
2. "+" > "Add package from git URL"
3. 输入：`com.unity.nuget.newtonsoft-json`

#### 4. 设置场景

1. **创建BrainManager**
   - Hierarchy > Create Empty > "BrainManager"

2. **添加组件**
   - Add Component > Brain Visualization
     - Json Path: `StreamingAssets/state`
     - Load Sequence: ✓
   - Add Component > Brain Config Loader
     - Config Path: `StreamingAssets/unity_config.json`

3. **创建脑区预制体**（如果有OBJ模型）
   - 从 `Assets/Models/` 选择一个OBJ文件
   - 拖到Scene中，调整大小
   - 拖到Project创建Prefab
   - 赋值给Brain Visualization的Region Prefab

## 🔄 实时工作流（可选）

### 启动后端服务器

```bash
# 在项目目录
python -m unity_integration.realtime_server
```

### Unity连接

1. 选中BrainManager
2. Add Component > WebSocket Client
3. Server URL: `ws://localhost:8765`
4. Auto Connect: ✓

### 工作流程

1. Unity请求预测
2. 后端处理生成JSON
3. 自动保存到 `state/` 文件夹
4. Unity自动加载新状态

## 📊 数据处理流程

```
原始数据
  ↓ [放入data/raw/]
预处理
  ↓ [自动缓存到data/cache/]
状态提取
  ↓ [生成JSON到state/]
Unity加载
  ↓ [从StreamingAssets/state/读取]
可视化
```

## 🎮 Unity操作

### 快捷键
- **空格**: 播放/暂停动画
- **R**: 重新加载数据

### 可视化参数

在Brain Visualization组件中调整：

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| Region Scale | 脑区大小 | 0.5-2.0 |
| Activity Threshold | 显示阈值 | 0.1-0.5 |
| Show Connections | 显示连接 | 开/关 |
| Connection Threshold | 连接阈值 | 0.3-0.7 |
| FPS | 动画帧率 | 5-30 |

## 📁 文件夹说明

```
Unity_TwinBrain/
├── freesurfer/          # FreeSurfer文件（构建阶段）
│   ├── lh.pial
│   ├── rh.pial
│   ├── lh.Schaefer*.annot
│   └── rh.Schaefer*.annot
├── data/                # 实际脑数据
│   ├── raw/            # 原始fMRI/EEG
│   └── cache/          # 预处理缓存
├── state/              # JSON状态文件（Unity使用）
├── OBJ/                # 3D脑区模型（从FreeSurfer生成）
├── Scripts/            # Unity C#脚本
└── unity_config.json   # Unity配置
```

## ❓ 常见问题

### Q: 没有FreeSurfer文件怎么办？
A: 可以使用基本的Sphere预制体，只需要JSON状态文件即可可视化活动。

### Q: 如何添加更多数据？
A: 将数据放入 `data/` 文件夹，运行数据处理脚本，新的JSON会生成到 `state/`。

### Q: 预测结果在哪里？
A: 后端预测自动保存为JSON到 `state/` 文件夹，Unity可自动加载。

### Q: OBJ模型太多，Unity很慢？
A: 
1. 使用Sphere预制体代替OBJ模型
2. 优化OBJ模型（减少面数）
3. 使用Unity的LOD系统

### Q: 如何使用不同的atlas？
A: 
```bash
# 生成时指定atlas
python setup_unity_project.py \
    --freesurfer-dir /path/to/fs \
    --atlas Schaefer100  # 或Schaefer400
```

### Q: 数据在哪里处理？
A: 使用TwinBrain主项目的处理脚本：
```bash
python -m unity_integration.brain_state_exporter \
    --data-dir Unity_TwinBrain/data \
    --output Unity_TwinBrain/state
```

## 🔧 高级配置

### 自定义颜色

编辑 `unity_config.json`:
```json
{
  "colors": {
    "low_activity": {"r": 0, "g": 255, "b": 0},
    "high_activity": {"r": 255, "g": 0, "b": 0}
  }
}
```

### 批量处理数据

```python
# 批量处理多个被试的数据
for subject in subjects:
    python -m unity_integration.brain_state_exporter \
        --data-dir data/{subject} \
        --output state/{subject} \
        --subject-id {subject}
```

## 📞 获取帮助

- **README**: 查看 `Unity_TwinBrain/README_UNITY.md`
- **GitHub Issues**: https://github.com/sheinclotho/twinbrain/issues
- **文档**: 查看项目中的其他文档

## 🎉 完整示例

```bash
# 1. 生成项目（使用FreeSurfer）
python setup_unity_project.py \
    --freesurfer-dir my_freesurfer_data \
    --atlas Schaefer200

# 2. 处理数据
python -m unity_integration.brain_state_exporter \
    --data-dir Unity_TwinBrain/data \
    --output Unity_TwinBrain/state

# 3. 启动后端（可选）
python -m unity_integration.realtime_server

# 4. 在Unity中导入并运行
```

---

**关键原则：**
- ✅ 使用真实FreeSurfer数据
- ✅ 基于实际脑数据处理
- ✅ 不使用任何编造的示例数据
- ✅ 两阶段工作流：构建 + 使用

最后更新: 2024-02-05
