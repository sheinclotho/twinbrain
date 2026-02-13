# TwinBrain Unity 完整工作流程说明

## 脚本功能区分

### setup_unity_project.py - 项目初始化脚本（运行一次）

**用途**: 生成Unity项目所需的所有静态资源

**功能**:
1. 创建Unity项目文件夹结构
2. 从FreeSurfer .lh/.rh文件和.annot文件生成**每个脑区的独立OBJ模型**
   - 输出: `region_0001.obj`, `region_0002.obj`, ..., `region_0200.obj`
3. 复制Unity C#脚本到Scripts文件夹
4. 生成unity_config.json配置文件
5. 创建README和文档

**运行时机**: 只在项目初始化时运行一次

**示例**:
```bash
# 使用FreeSurfer文件生成多脑区OBJ模型
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer_files

# 生成的文件结构：
Unity_TwinBrain/
├── OBJ/                    # 多个独立脑区OBJ模型
│   ├── region_0001.obj     # 第1个脑区
│   ├── region_0002.obj     # 第2个脑区
│   └── ...
├── Scripts/                # Unity C#脚本
│   ├── BrainVisualization.cs
│   ├── StimulationInput.cs
│   └── ...
├── state/                  # 空文件夹，用于存放运行时数据
└── unity_config.json       # Unity配置
```

---

### unity_startup.py - 运行时后端服务器（每次使用时运行）

**用途**: 提供Unity运行时的后端模型预测和WebSocket通信

**功能**:
1. 加载训练好的PyTorch模型
2. 启动WebSocket服务器，监听Unity客户端
3. 接收预测请求 → 运行模型 → 生成JSON
4. 接收刺激模拟请求 → 计算效果 → 生成JSON
5. **自动将结果保存到state/文件夹**

**运行时机**: 每次使用Unity可视化前启动

**示例**:
```bash
# 启动后端服务器
python unity_startup.py --model results/model.pth --output Unity_TwinBrain

# 服务器运行后：
# 1. Unity连接到 ws://localhost:8765
# 2. Unity发送预测请求
# 3. 后端自动保存到: Unity_TwinBrain/state/brain_state_t0001.json
# 4. Unity自动加载JSON并映射到多个OBJ模型
```

---

## 完整使用流程

### 阶段1: 项目初始化（一次性）

```bash
# 1. 准备FreeSurfer文件
mkdir freesurfer_files
cp lh.pial rh.pial freesurfer_files/
cp lh.Schaefer*.annot rh.Schaefer*.annot freesurfer_files/

# 2. 运行初始化脚本（生成多脑区OBJ）
python setup_unity_project.py --freesurfer-dir freesurfer_files

# 生成内容：
# ✓ 200个独立脑区OBJ文件
# ✓ Unity C#脚本（已修复命名空间）
# ✓ unity_config.json配置
# ✓ 文件夹结构
```

### 阶段2: Unity项目导入

```
1. 打开Unity Hub，创建新3D项目
2. 将 Unity_TwinBrain/Scripts/ 复制到 Assets/Scripts/
3. 将 Unity_TwinBrain/OBJ/ 复制到 Assets/Models/
4. 将 Unity_TwinBrain/state/ 复制到 Assets/StreamingAssets/state/
5. 将 unity_config.json 复制到 Assets/StreamingAssets/
```

### 阶段3: 准备数据（手动或自动）

#### 方式1: 手动放置原始数据

```bash
# 将fMRI原始数据放入data/raw/
cp your_fmri.nii Unity_TwinBrain/data/raw/

# 运行预处理脚本（手动）
python -m preprocess.fmri_preprocessor \
    --input Unity_TwinBrain/data/raw/your_fmri.nii \
    --output Unity_TwinBrain/data/cache/preprocessed.pkl
```

#### 方式2: 使用缓存数据

```bash
# 直接放置预处理好的缓存文件
cp preprocessed_data.pkl Unity_TwinBrain/data/cache/
```

### 阶段4: 启动后端服务（每次使用）

```bash
# 启动后端服务器，加载模型
python unity_startup.py \
    --model results/trained_model.pth \
    --output Unity_TwinBrain
```

**后端服务器做什么**:
1. 加载训练好的PyTorch模型
2. 启动WebSocket服务器 (ws://localhost:8765)
3. 等待Unity连接
4. 接收请求 → 运行模型 → **自动保存JSON到state/文件夹**

### 阶段5: Unity运行

```
1. 在Unity中按Play按钮
2. BrainVisualization组件自动：
   - 连接到WebSocket服务器
   - 发送初始状态请求
3. 用户交互：
   - 点击脑区选择
   - 设置刺激参数
   - 点击"发送刺激"按钮
4. 后端自动：
   - 运行模型预测
   - 保存结果: state/prediction_t0001.json
5. Unity自动：
   - 检测到新JSON文件
   - 加载并解析
   - 将数值映射到200个脑区OBJ
   - 用颜色显示活动（真实=蓝红，预测=绿色）
```

---

## 数据流程图

```
FreeSurfer文件 (.lh, .annot)
    ↓ [setup_unity_project.py - 运行一次]
多个脑区OBJ (region_XXXX.obj)
    ↓ [复制到Unity Assets/]
Unity项目准备就绪
    ↓
原始fMRI数据
    ↓ [手动预处理或使用缓存]
缓存数据 (.pkl, .npy)
    ↓
启动后端服务器 [unity_startup.py]
    ↓
Unity连接 → 发送请求
    ↓
后端模型预测
    ↓ [自动保存]
JSON文件 (state/brain_state_tXXXX.json)
    ↓ [Unity自动加载]
映射到200个脑区OBJ
    ↓
可视化显示（颜色映射）
```

---

## 关键数据格式

### 1. FreeSurfer输入

**文件**:
- `lh.pial` / `rh.pial` - 表面网格（顶点+面）
- `lh.Schaefer2018_200Parcels_7Networks_order.annot` - 脑区标注

**处理**: `FreeSurferLoader` 读取并计算每个脑区的质心坐标

### 2. OBJ模型输出

**文件**: `region_0001.obj`, `region_0002.obj`, ...

**格式**:
```obj
# Region 1: LH_Vis_1
v 10.234 20.456 30.789
v 10.345 20.567 30.890
...
f 1 2 3
```

### 3. 缓存数据格式

**来源**: 预处理后的fMRI/EEG数据

**格式** (numpy/pickle):
```python
{
    'fmri': np.ndarray,      # shape: [n_regions, n_timepoints, n_features]
    'timestamps': np.ndarray, # shape: [n_timepoints]
    'region_ids': np.ndarray  # shape: [n_regions]
}
```

### 4. 模型预测输出

**自动保存位置**: `Unity_TwinBrain/state/prediction_tXXXX.json`

**格式**:
```json
{
  "version": "2.0",
  "timestamp": "2024-02-13T12:00:00",
  "metadata": {
    "subject": "sub-001",
    "atlas": "Schaefer200",
    "time_point": 10
  },
  "brain_state": {
    "regions": [
      {
        "id": 1,
        "label": "LH_Vis_1",
        "position": {"x": 10.2, "y": 20.4, "z": 30.8},
        "activity": {
          "fmri": {"amplitude": 0.75, "raw_value": 1.23},
          "predictionValue": 0.82,
          "isPredicted": true
        }
      },
      ...
    ]
  }
}
```

### 5. JSON映射到OBJ

**Unity中的映射**:
```csharp
// BrainVisualization.cs 自动处理
foreach (RegionData region in currentState.brain_state.regions)
{
    // 加载对应的OBJ模型
    string objPath = $"StreamingAssets/OBJ/region_{region.id:D4}.obj";
    GameObject regionObj = LoadObjModel(objPath);
    
    // 根据预测值设置颜色
    float activity = region.activity.predictionValue;
    Color color = region.activity.isPredicted 
        ? Color.Lerp(lowColor, predictedColor, activity)  // 预测=绿色
        : Color.Lerp(lowColor, highColor, activity);      // 真实=红色
    
    regionObj.GetComponent<Renderer>().material.color = color;
}
```

---

## 常见问题

### Q: 如何确认多脑区OBJ已生成？

**A**: 运行setup_unity_project.py后检查：
```bash
ls -l Unity_TwinBrain/OBJ/
# 应该看到 region_0001.obj, region_0002.obj, ..., region_0200.obj
```

### Q: 后端如何自动保存预测结果？

**A**: `BrainVisualizationServer` 在接收到预测请求后：
1. 调用模型生成预测
2. 使用 `BrainStateExporter.export_brain_state()` 保存JSON
3. JSON自动保存到 `--output` 指定的state/文件夹
4. Unity通过文件监视或定时刷新加载新JSON

### Q: 缓存数据从哪里来？

**A**: 两个来源：
1. **手动预处理**: 使用 `preprocess/fmri_preprocessor.py` 处理原始.nii文件
2. **已有缓存**: 直接使用之前保存的.pkl或.npy文件

### Q: 如何验证完整流程？

**A**: 测试步骤：
```bash
# 1. 初始化项目
python setup_unity_project.py --freesurfer-dir freesurfer_files

# 2. 验证OBJ文件
ls Unity_TwinBrain/OBJ/*.obj | wc -l  # 应该是200

# 3. 启动后端（演示模式）
python unity_startup.py --demo --output Unity_TwinBrain

# 4. 在另一个终端测试WebSocket
python -c "import asyncio, websockets
async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        await ws.send('{\"type\": \"get_state\"}')
        response = await ws.recv()
        print(response)
asyncio.run(test())"

# 5. 在Unity中运行并检查
# - Console应显示"Connected to WebSocket"
# - 200个脑区OBJ应正确加载
# - 点击脑区应有响应
```

---

## 总结

**setup_unity_project.py = 初始化（一次）**
- 输入: FreeSurfer文件
- 输出: 多个OBJ模型 + C#脚本 + 配置

**unity_startup.py = 运行时服务（每次）**
- 输入: 训练模型 + 缓存数据
- 输出: WebSocket服务 + 自动生成JSON

**两者配合使用，缺一不可！**
