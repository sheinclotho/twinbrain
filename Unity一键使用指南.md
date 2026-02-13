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
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer/files
```

如果没有FreeSurfer文件：
```bash
# 创建基本项目结构
python setup_unity_project.py --auto-setup

# 然后将FreeSurfer文件放入生成的freesurfer_files/文件夹
# 再次运行生成OBJ模型
```

**这会生成项目文件夹 `unity_project/`，包含：**
- ✅ 200个脑区的真实3D OBJ模型（从FreeSurfer，在 `Unity_Assets/Models/` 中）
- ✅ 正确的文件夹结构（详见下方）
- ✅ Unity C#脚本（在 `Unity_Assets/Scripts/` 中）
- ✅ 配置文件和说明文档

**生成的Unity C#脚本模板：**
- `BrainDataLoader.cs` - 数据加载器
- `AnimationController.cs` - 动画控制
- `StimulationInput.cs` - 刺激输入UI
- `ModelInterface.cs` - 模型接口

**注意**: 脚本是基础模板，根据需要自定义

### 阶段2: 准备数据

**重要**: 以下路径使用的是默认输出目录 `unity_project/`，如果你使用了 `--output-dir` 指定其他目录，请相应修改路径。

```bash
# 将原始fMRI数据放入original文件夹
cp your_fmri_data.nii unity_project/brain_data/original/

# 或将预处理缓存数据放入cache文件夹
cp preprocessed_data.pkl unity_project/brain_data/cache/

# 处理数据生成JSON状态文件到model_output文件夹
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output
```

### 阶段3: Unity设置

#### 1. 创建Unity项目

1. 打开Unity Hub
2. 创建新的3D项目
3. 命名（例如：TwinBrain_Demo）

#### 2. 导入资源

**C#脚本：**
- 将 `unity_project/Unity_Assets/Scripts/` 复制到你的Unity项目的 `Assets/Scripts/`

**3D模型：**（如果使用了FreeSurfer生成）
- 将 `unity_project/Unity_Assets/Models/` 复制到 `Assets/Models/`

**数据文件：**
- 在Unity项目的 `Assets/` 下创建 `StreamingAssets` 文件夹
- 将 `unity_project/brain_data/model_output/` 中的JSON文件复制到 `Assets/StreamingAssets/brain_states/`
- 将 `unity_project/unity_config.json` 复制到 `Assets/StreamingAssets/`

#### 3. 安装依赖

**Newtonsoft.Json:**
1. Window > Package Manager
2. "+" > "Add package from git URL"
3. 输入：`com.unity.nuget.newtonsoft-json`

#### 4. 设置场景

1. **创建BrainManager**
   - Hierarchy > Create Empty > "BrainManager"

2. **添加组件**
   - Add Component > Brain Data Loader（生成的脚本）
     - Json Directory: `StreamingAssets/brain_states`
     - Config Path: `StreamingAssets/unity_config.json`
   - Add Component > Animation Controller（可选，用于时间序列）

3. **创建脑区预制体**（如果有OBJ模型）
   - 从 `Assets/Models/` 选择一个OBJ文件
   - 拖到Scene中，调整大小
   - 拖到Project创建Prefab
   - 赋值给Brain Data Loader的Region Prefab字段

   如果没有OBJ模型，使用简单球体：
   - Hierarchy > 3D Object > Sphere
   - 缩放（Scale: 0.5, 0.5, 0.5）
   - 拖到Project创建Prefab命名为 `BrainRegion`
   - 赋值给Brain Data Loader的Region Prefab字段

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

**setup_unity_project.py 生成的目录结构：**

```
unity_project/                   # 默认输出目录（可用--output-dir修改）
├── freesurfer_files/            # 【放置】FreeSurfer文件
│   ├── lh.pial                  # 左半球表面
│   ├── rh.pial                  # 右半球表面
│   ├── lh.Schaefer*.annot       # 左半球标注
│   └── rh.Schaefer*.annot       # 右半球标注
├── brain_data/                  # 脑数据目录
│   ├── original/                # 【放置】原始fMRI/EEG数据
│   ├── cache/                   # 【放置】预处理缓存文件(.pkl/.npy)
│   └── model_output/            # 【生成】JSON状态文件（Unity读取）
├── Unity_Assets/                # Unity资源
│   ├── Scripts/                 # 【生成】Unity C#脚本模板
│   │   ├── BrainDataLoader.cs
│   │   ├── AnimationController.cs
│   │   ├── StimulationInput.cs
│   │   └── ModelInterface.cs
│   └── Models/                  # 【生成】3D脑区模型（如用FreeSurfer）
│       ├── region_0001.obj
│       ├── region_0002.obj
│       └── ...
└── unity_config.json            # 【生成】Unity配置文件
```

**使用说明：**
- 【放置】= 需要用户放入的文件
- 【生成】= setup_unity_project.py自动生成的文件

## ❓ 常见问题

### Q: FreeSurfer文件要放到哪个文件夹？
A: 
1. 先运行 `python setup_unity_project.py --auto-setup` 创建目录结构
2. 将FreeSurfer文件放入 `unity_project/freesurfer_files/` 文件夹：
   - `lh.pial`, `rh.pial`
   - `lh.Schaefer2018_200Parcels_7Networks_order.annot`
   - `rh.Schaefer2018_200Parcels_7Networks_order.annot`
3. 再次运行生成OBJ模型：`python setup_unity_project.py --freesurfer-dir unity_project/freesurfer_files`

### Q: cache文件要放到哪里？如何自动转换为JSON？
A: 
1. 将预处理的cache文件（.pkl或.npy）放入 `unity_project/brain_data/cache/`
2. 运行以下命令自动生成JSON：
```bash
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output
```
3. JSON文件会自动生成到 `unity_project/brain_data/model_output/` 目录

### Q: 训练好的模型要放到哪里？
A: 
如果要使用实时预测功能（可选）：
1. 将训练好的模型文件（.pt或.pth）放到项目的 `results/` 目录
2. 启动后端服务器：
```bash
python unity_startup.py --model results/your_model.pt --output unity_project
```
3. 服务器会自动将预测结果保存为JSON到 `unity_project/brain_data/model_output/`

**注意**：如果只是可视化已有数据，不需要模型文件，直接使用cache生成的JSON即可。

### Q: Unity中脚本要附加到什么GameObject上？
A:
1. 创建空GameObject命名为 `BrainManager`（Hierarchy > Create Empty）
2. 添加 `BrainDataLoader` 组件（Add Component > Brain Data Loader）
3. 配置参数：
   - Json Directory: `StreamingAssets/brain_states`
   - Config Path: `StreamingAssets/unity_config.json`
   - Region Prefab: 拖入你的脑区预制体（Sphere或OBJ模型）

### Q: 没有FreeSurfer文件怎么办？
A: 可以使用基本的Sphere预制体，只需要JSON状态文件即可可视化活动。

### Q: 如何添加更多数据？
A: 将数据放入 `unity_project/brain_data/cache/` 文件夹，运行 brain_state_exporter，新的JSON会生成到 `model_output/`。

### Q: OBJ模型太多，Unity很慢？
A: 
1. 使用Sphere预制体代替OBJ模型
2. 优化OBJ模型（减少面数）
3. 使用Unity的LOD系统

### Q: 如何使用不同的atlas？
A: FreeSurfer标注文件的名称已经指定了atlas类型（如Schaefer2018_200Parcels），系统会自动识别。只需确保标注文件名称正确即可。
```bash
# 示例：使用不同数量的脑区
# 将对应的标注文件放入unity_project/freesurfer_files/文件夹
# - lh.Schaefer2018_100Parcels_7Networks_order.annot (100个脑区)
# - lh.Schaefer2018_200Parcels_7Networks_order.annot (200个脑区)
# - lh.Schaefer2018_400Parcels_7Networks_order.annot (400个脑区)
python setup_unity_project.py --freesurfer-dir /path/to/fs
```

### Q: 数据在哪里处理？
A: 使用TwinBrain主项目的处理脚本：
```bash
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output
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

```bash
# 批量处理多个被试的数据
for subject in subject_01 subject_02 subject_03; do
    python -m unity_integration.brain_state_exporter \
        --data-dir unity_project/brain_data/cache/$subject \
        --output unity_project/brain_data/model_output/$subject \
        --subject-id $subject
done
```

## 📞 获取帮助

- **架构说明**: 查看 `Unity架构说明.md` 了解技术细节
- **GitHub Issues**: https://github.com/sheinclotho/twinbrain/issues
- **文档**: 查看项目中的其他文档

## 🎉 完整示例

```bash
# 1. 生成项目（使用FreeSurfer）
python setup_unity_project.py --freesurfer-dir my_freesurfer_data

# 2. 处理数据
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output

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
