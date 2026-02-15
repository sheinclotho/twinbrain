# TwinBrain Unity 完整使用指南

> **最后更新**: 2024-02-15  
> **版本**: 4.0  
> **适用于**: Unity 2019.1+, Python 3.8+

---

## 📋 目录

1. [简介](#简介)
2. [系统要求](#系统要求)
3. [快速开始](#快速开始)
4. [详细安装流程](#详细安装流程)
5. [Unity内操作指南](#unity内操作指南)
6. [脚本功能说明](#脚本功能说明)
7. [进阶使用](#进阶使用)
8. [常见问题](#常见问题)

---

## 简介

TwinBrain Unity可视化模块用于3D展示大脑活动数据。本指南将带您完成从安装到使用的完整流程。

### 主要特点

- ✅ **一键安装**: 使用`unity_one_click_install.py`脚本完成所有安装步骤
- ✅ **自动配置**: Unity Editor工具自动配置所有OBJ文件和组件
- ✅ **无需手动拖拽**: 自动处理200+个脑区OBJ文件
- ✅ **实时可视化**: 支持离线JSON文件和实时WebSocket连接
- ✅ **FreeSurfer支持**: 可选使用真实脑表面数据生成3D模型

---

## 系统要求

### 必需

- **Unity**: 2019.1或更高版本（推荐2020 LTS或2021 LTS）
- **Python**: 3.8或更高版本
- **操作系统**: Windows 10+, macOS 10.14+, 或 Linux (Ubuntu 18.04+)
- **内存**: 8GB RAM（推荐16GB）
- **硬件**: 支持DirectX 11或OpenGL 4.5的GPU

### 可选

- **FreeSurfer 7.0+**: 用于生成真实3D脑区模型
- **训练好的TwinBrain模型**: 用于实时预测功能

---

## 快速开始

### 第一步：创建Unity项目

1. 打开Unity Hub
2. 点击"新建项目"
3. 选择"3D"模板
4. 项目名称：`TwinBrainDemo`（或任意名称）
5. 选择保存位置
6. 点击"创建项目"

### 第二步：运行一键安装脚本

打开终端（命令提示符），进入TwinBrain项目目录：

```bash
cd /path/to/twinbrain
```

**如果您有FreeSurfer数据**（推荐）：

```bash
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo \
    --freesurfer-dir /path/to/freesurfer
```

**如果没有FreeSurfer数据**：

```bash
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo
```

这个脚本会自动完成：
- ✅ 生成Unity资源文件
- ✅ 生成OBJ脑区模型（如果提供FreeSurfer）
- ✅ 复制所有C#脚本到Unity项目
- ✅ 复制所有OBJ文件到Unity项目
- ✅ 安装Unity Editor自动化工具
- ✅ 配置项目依赖

### 第三步：在Unity中完成自动设置

1. **打开Unity项目**
   - 在Unity Hub中打开刚才创建的项目
   - 等待Unity加载（首次打开需要1-2分钟）

2. **等待包下载完成**
   - Unity会自动下载Newtonsoft.Json包
   - 查看Unity编辑器右下角的进度条
   - 等待"Import Package"完成

3. **运行自动设置工具**
   - 在Unity菜单栏中，点击：**TwinBrain → 自动设置场景**
   - 在弹出的窗口中，点击：**"开始自动设置"**
   - 确认对话框中点击：**"继续"**
   - 等待自动设置完成（约30秒-2分钟，取决于OBJ文件数量）

4. **完成！**
   - 查看Hierarchy面板，应该能看到"BrainManager"对象
   - 点击"BrainManager"，在Inspector面板中查看组件
   - 点击Unity编辑器上方的**Play按钮**进行测试

---

## 详细安装流程

### FreeSurfer数据准备（可选）

如果您想使用真实的3D脑区模型，需要准备以下FreeSurfer文件：

```
freesurfer_files/
├── lh.pial                                          # 左半球表面
├── rh.pial                                          # 右半球表面
├── lh.Schaefer2018_200Parcels_7Networks_order.annot # 左半球标注
└── rh.Schaefer2018_200Parcels_7Networks_order.annot # 右半球标注
```

**说明**：
- 这些文件是FreeSurfer软件处理MRI数据后的输出
- `lh`表示左半球(left hemisphere)，`rh`表示右半球(right hemisphere)
- `.pial`文件包含脑表面的顶点和面信息
- `.annot`文件包含脑区分割标注

### 一键安装脚本详解

**脚本功能**：

`unity_one_click_install.py`整合了所有安装步骤：

1. **生成Unity资源** (`setup_unity_project.py`的功能)
   - 创建标准文件夹结构
   - 如果提供FreeSurfer数据，生成200+个独立的OBJ文件（每个脑区一个文件）
   - 生成配置文件

2. **安装到Unity** (`unity_package_installer.py`的功能)
   - 验证Unity项目有效性
   - 复制所有C#脚本到`Assets/TwinBrain/Scripts/`
   - 创建Assembly Definition文件
   - 安装Unity Editor工具

3. **复制OBJ文件**
   - 自动复制所有OBJ文件到`Assets/StreamingAssets/OBJ/`
   - 无需手动拖拽文件

**参数说明**：

```bash
python unity_one_click_install.py \
    --unity-project /path/to/UnityProject \     # Unity项目路径（必需）
    --freesurfer-dir /path/to/freesurfer \      # FreeSurfer数据路径（可选）
    --output-dir ./my_output                    # 中间文件输出目录（可选）
```

**输出说明**：

脚本执行后会创建以下文件结构：

```
unity_project/                      # 中间文件目录
├── freesurfer_files/               # FreeSurfer数据
├── brain_data/
│   ├── cache/                      # 预处理缓存 (.pt, .pth格式)
│   ├── model_output/               # JSON状态文件
│   └── original/                   # 原始数据
└── Unity_Assets/
    ├── Scripts/                    # C#脚本（源文件）
    └── obj/                        # 生成的OBJ文件

UnityProject/                       # 您的Unity项目
└── Assets/
    ├── TwinBrain/
    │   ├── Scripts/                # 已安装的C#脚本
    │   │   ├── BrainVisualization.cs
    │   │   ├── WebSocketClient.cs
    │   │   ├── BrainConfigLoader.cs
    │   │   ├── StimulationInput.cs
    │   │   ├── TimelineController.cs
    │   │   └── ...
    │   ├── Editor/                 # Unity Editor工具
    │   │   └── TwinBrainAutoSetup.cs
    │   └── Prefabs/                # 预制体（运行自动设置后创建）
    └── StreamingAssets/
        └── OBJ/                    # 已复制的OBJ文件
            ├── region_0001.obj
            ├── region_0002.obj
            └── ...
```

---

## Unity内操作指南

### 自动设置工具使用

**TwinBrainAutoSetup**是一个Unity Editor工具，用于自动完成所有配置工作。

#### 打开自动设置工具

在Unity菜单栏中：**TwinBrain → 自动设置场景**

#### 工具界面说明

工具窗口包含以下选项：

1. **OBJ文件夹路径**
   - 默认：`Assets/StreamingAssets/OBJ`
   - 指定OBJ文件所在位置

2. **创建BrainManager**
   - 勾选：自动创建BrainManager GameObject
   - 不勾选：仅导入OBJ文件

3. **创建示例球体预制体**
   - 勾选：创建一个示例预制体用于快速测试
   - 不勾选：仅使用OBJ模型

4. **配置摄像机**
   - 勾选：自动设置主摄像机位置和参数
   - 不勾选：保持摄像机当前设置

#### 完整自动设置流程

点击**"开始自动设置"**按钮后，工具会执行以下操作：

1. **导入并配置OBJ文件**（约1-2分钟）
   - 自动导入所有OBJ文件
   - 设置每个OBJ的缩放比例为0.01
   - 配置材质导入选项
   - 刷新资产数据库

2. **创建示例预制体**（可选）
   - 在`Assets/TwinBrain/Prefabs/`创建预制体文件夹
   - 生成BrainRegion.prefab示例预制体

3. **创建BrainManager对象**
   - 在Hierarchy中创建"BrainManager" GameObject
   - 自动添加BrainVisualization组件
   - 设置默认参数

4. **配置摄像机**（可选）
   - 设置主摄像机位置：(0, 5, -10)
   - 设置摄像机角度：向下30度
   - 设置背景颜色：黑色

#### 部分执行功能

如果只想执行特定操作，可以使用：

- **"仅导入OBJ文件"**：只配置OBJ文件，不创建其他对象
- **"仅创建BrainManager"**：只创建BrainManager对象

### 手动配置BrainManager（可选）

如果您需要手动调整BrainManager的设置：

1. **在Hierarchy中选择BrainManager**
2. **在Inspector面板中查看BrainVisualization组件**
3. **配置参数**：

#### File Settings（文件设置）

- **Json Path**: JSON数据文件路径
  - 例如：`brain_state.json`
  - 或目录路径用于加载序列

- **Load Sequence**: 是否加载目录中的所有JSON文件
  - 勾选：加载整个时间序列
  - 不勾选：只加载单个文件

#### Model Settings（模型设置）

- **Use Obj Models**: 是否使用OBJ模型
  - 勾选：使用导入的OBJ文件作为脑区模型
  - 不勾选：使用预制体（Region Prefab）

- **Obj Directory**: OBJ文件目录
  - 默认：`StreamingAssets/OBJ`

- **Region Prefab**: 脑区预制体
  - 当不使用OBJ模型时，使用此预制体

#### Visualization Settings（可视化设置）

- **Region Scale**: 脑区缩放比例（默认：1.0）
- **Activity Threshold**: 活动阈值（0-1）
- **Show Connections**: 是否显示连接线
- **Connection Threshold**: 连接强度阈值（0-1）
- **Connection Material**: 连接线材质

#### Animation Settings（动画设置）

- **Fps**: 帧率（默认：10）
- **Auto Play**: 是否自动播放序列

#### Colors（颜色设置）

- **Low Activity Color**: 低活动值颜色（默认：蓝色）
- **High Activity Color**: 高活动值颜色（默认：红色）

#### Interaction（交互设置）

- **Enable Interaction**: 是否启用点击交互

---

## 脚本功能说明

### 核心Unity脚本

TwinBrain提供以下C#脚本：

#### 1. BrainVisualization.cs

**功能**：主要的可视化组件

**用途**：
- 加载和显示JSON格式的大脑状态数据
- 支持OBJ模型和预制体两种显示模式
- 颜色映射显示脑区活动强度
- 动画播放时间序列数据
- 点击交互选择脑区

**使用方法**：
```csharp
// 附加到GameObject
gameObject.AddComponent<BrainVisualization>();

// 或在Inspector中添加组件
```

#### 2. WebSocketClient.cs

**功能**：实时通信客户端

**用途**：
- 连接到TwinBrain Python后端服务器
- 实时接收大脑状态更新
- 发送刺激请求
- HTTP轮询模式（兼容性好）

**使用方法**：
```csharp
// 附加到BrainManager或独立GameObject
var wsClient = gameObject.AddComponent<WebSocketClient>();
wsClient.serverUrl = "http://localhost:8765";
wsClient.autoConnect = true;

// 订阅事件
wsClient.OnBrainStateReceived += (state) => {
    Debug.Log($"Received brain state with {state.regions.Count} regions");
};
```

#### 3. BrainConfigLoader.cs

**功能**：配置文件加载器

**用途**：
- 从JSON文件加载配置
- 管理脑区映射和参数

#### 4. StimulationInput.cs

**功能**：虚拟刺激输入组件

**用途**：
- 创建UI界面接收用户输入
- 发送刺激信号到后端
- 模拟脑区刺激

#### 5. TimelineController.cs

**功能**：时间序列控制器

**用途**：
- 控制动画播放
- 提供播放、暂停、快进等功能
- UI界面集成

#### 6. CacheToJsonConverter.cs

**功能**：缓存文件转换工具

**用途**：
- 将PyTorch缓存(.pt, .pth)转换为JSON
- 用于离线可视化

### Python后端模块

位于`unity_integration/`目录：

#### 1. realtime_server.py

**功能**：实时WebSocket服务器

**启动方法**：
```bash
python -m unity_integration.realtime_server --port 8765
```

**提供接口**：
- `/state` - 获取当前大脑状态
- `/predict` - 请求预测
- `/stimulate` - 模拟刺激
- `/stream` - 流式传输数据

#### 2. model_server.py

**功能**：模型加载和推理

**用途**：
- 加载训练好的TwinBrain模型
- 执行预测任务
- 管理模型缓存

#### 3. brain_state_exporter.py

**功能**：状态导出工具

**用途**：
- 导出大脑状态为JSON格式
- 批量转换缓存文件

**使用方法**：
```bash
python -m unity_integration.brain_state_exporter \
    --cache-dir brain_data/cache \
    --output-dir brain_data/model_output
```

#### 4. obj_generator.py

**功能**：OBJ模型生成器

**用途**：
- 从FreeSurfer数据生成OBJ文件
- 每个脑区生成独立的OBJ文件

#### 5. freesurfer_loader.py

**功能**：FreeSurfer数据加载器

**用途**：
- 读取FreeSurfer表面文件
- 解析脑区标注信息

#### 6. stimulation_simulator.py

**功能**：刺激模拟器

**用途**：
- 模拟虚拟刺激
- 计算刺激响应

---

## 进阶使用

### 模式一：离线可视化（JSON文件）

**适用场景**：查看已保存的大脑状态数据

**步骤**：

1. **准备JSON数据文件**
   - 将JSON文件放在`Assets/StreamingAssets/`或其子目录
   - 文件格式示例：
   ```json
   {
       "regions": {
           "0": {"activity": 0.75, "position": [0, 0, 0]},
           "1": {"activity": 0.62, "position": [1, 0, 0]}
       },
       "connections": [
           {"source": 0, "target": 1, "weight": 0.8}
       ],
       "timestamp": 1234567890
   }
   ```

2. **配置BrainVisualization**
   - Json Path: `StreamingAssets/brain_state.json`
   - Load Sequence: 根据需要勾选

3. **点击Play**
   - Unity会自动加载并显示数据

### 模式二：实时可视化（WebSocket连接）

**适用场景**：实时显示模型预测结果或进行交互式刺激

**步骤**：

1. **启动Python后端服务器**
   ```bash
   cd /path/to/twinbrain
   
   # 方式1：使用unity_startup.py（推荐）
   python unity_startup.py --model results/hetero_gnn_trained.pt
   
   # 方式2：直接启动realtime_server
   python -m unity_integration.realtime_server --port 8765
   ```

2. **在Unity中添加WebSocketClient**
   - 选择BrainManager对象
   - 在Inspector中点击"Add Component"
   - 搜索并添加"WebSocketClient"
   - 配置Server URL: `http://localhost:8765`

3. **连接事件处理**
   
   创建一个新脚本`BrainServerConnector.cs`：
   ```csharp
   using UnityEngine;
   using TwinBrain;
   
   public class BrainServerConnector : MonoBehaviour
   {
       private WebSocketClient wsClient;
       private BrainVisualization visualization;
       
       void Start()
       {
           wsClient = GetComponent<WebSocketClient>();
           visualization = GetComponent<BrainVisualization>();
           
           // 订阅数据接收事件
           wsClient.OnBrainStateReceived += OnDataReceived;
       }
       
       void OnDataReceived(BrainStateData state)
       {
           // 更新可视化
           visualization.UpdateState(state);
       }
   }
   ```

4. **点击Play开始实时可视化**

### 模式三：批量转换缓存文件

**适用场景**：将训练过程中生成的PyTorch缓存文件转换为Unity可读的JSON

**步骤**：

1. **找到缓存文件**
   - 通常位于：`brain_data/cache/`
   - 文件格式：`.pt`或`.pth`（PyTorch格式）

2. **运行转换脚本**
   ```bash
   python -m unity_integration.brain_state_exporter \
       --cache-dir brain_data/cache \
       --output-dir brain_data/model_output \
       --format json
   ```

3. **在Unity中加载生成的JSON文件**
   - Json Path: `StreamingAssets/model_output/brain_state_0001.json`
   - Load Sequence: 勾选（如果有多个文件）

### 创建自定义UI

**创建控制面板**：

1. **在Hierarchy中创建Canvas**
   - 右键 → UI → Canvas

2. **添加控制按钮**
   - 在Canvas下创建Button
   - 命名为"PlayButton", "PauseButton"等

3. **创建控制脚本**
   ```csharp
   using UnityEngine;
   using UnityEngine.UI;
   using TwinBrain;
   
   public class BrainControlPanel : MonoBehaviour
   {
       public TimelineController timeline;
       public Button playButton;
       public Button pauseButton;
       
       void Start()
       {
           playButton.onClick.AddListener(OnPlay);
           pauseButton.onClick.AddListener(OnPause);
       }
       
       void OnPlay()
       {
           timeline.Play();
       }
       
       void OnPause()
       {
           timeline.Pause();
       }
   }
   ```

4. **附加脚本到Canvas**
   - 将脚本拖到Canvas对象上
   - 在Inspector中连接引用

### 性能优化

**优化大量OBJ文件的性能**：

1. **使用LOD（Level of Detail）**
   ```csharp
   // 为每个脑区添加LOD组件
   var lodGroup = regionObject.AddComponent<LODGroup>();
   ```

2. **合批处理**
   - 启用GPU Instancing
   - 使用相同材质

3. **异步加载**
   ```csharp
   // 异步加载OBJ文件
   IEnumerator LoadOBJAsync(string path)
   {
       var request = Resources.LoadAsync<GameObject>(path);
       yield return request;
       var obj = Instantiate(request.asset as GameObject);
   }
   ```

4. **视锥体剔除**
   - 确保摄像机的Culling Mask正确设置

---

## 常见问题

### Q1: 运行一键安装脚本时出错

**A**: 检查以下几点：

1. **Python环境**
   ```bash
   python --version  # 确保是3.8+
   pip install -r requirements.txt  # 安装依赖
   ```

2. **Unity项目路径**
   - 确保路径正确且包含Assets目录
   - 路径中不要有中文或特殊字符

3. **权限问题**
   - Windows: 以管理员身份运行命令提示符
   - macOS/Linux: 使用`sudo`或检查文件权限

### Q2: Unity中找不到"TwinBrain"菜单

**A**: 可能的原因和解决方案：

1. **包未完全加载**
   - 等待Unity完全加载项目
   - 查看右下角进度条

2. **Newtonsoft.Json包未安装**
   - 打开Package Manager (Window → Package Manager)
   - 搜索"Newtonsoft.Json"
   - 点击"Install"

3. **编译错误**
   - 查看Console面板（Ctrl+Shift+C）
   - 修复所有红色错误
   - 等待脚本重新编译

4. **Editor脚本未安装**
   - 检查`Assets/TwinBrain/Editor/TwinBrainAutoSetup.cs`是否存在
   - 重新运行安装脚本

### Q3: 自动设置工具报错

**A**: 常见错误处理：

1. **"OBJ文件夹不存在"**
   - 确保OBJ文件已复制到`Assets/StreamingAssets/OBJ/`
   - 或取消勾选"Use Obj Models"

2. **"BrainManager已存在"**
   - 选择"重新创建"或手动删除现有对象

3. **"Assets导入失败"**
   - 等待Unity完成当前导入任务
   - 保存场景后重试

### Q4: OBJ文件太多，导入很慢

**A**: 优化建议：

1. **分批导入**
   - 先导入部分OBJ文件测试
   - 确认无误后再导入全部

2. **使用SSD**
   - 将Unity项目放在SSD上
   - 显著提升导入速度

3. **减少脑区数量**
   - 使用更粗粒度的脑区划分
   - 例如：使用100个脑区而不是200个

### Q5: 点击Play后没有显示任何内容

**A**: 检查清单：

1. **BrainManager对象**
   - 确保Hierarchy中有BrainManager
   - 确保对象是激活状态（勾选对象名称旁的复选框）

2. **数据文件**
   - 确保Json Path指向有效的JSON文件
   - 检查JSON文件格式是否正确

3. **摄像机位置**
   - 调整摄像机位置，确保能看到脑区
   - 或使用自动设置工具配置摄像机

4. **OBJ模型**
   - 检查"Use Obj Models"设置
   - 如果没有OBJ文件，取消勾选并设置Region Prefab

5. **Console错误**
   - 查看Console面板的错误信息
   - 根据错误提示修复问题

### Q6: 如何连接到Python后端？

**A**: 实时连接步骤：

1. **启动Python服务器**
   ```bash
   python unity_startup.py --model results/hetero_gnn_trained.pt
   ```
   
   看到以下输出表示成功：
   ```
   Server started on http://localhost:8765
   Waiting for Unity connection...
   ```

2. **在Unity中添加WebSocketClient**
   - 详见"进阶使用"部分的说明

3. **测试连接**
   - 点击Play
   - 查看Console输出
   - 应该看到"Connected to server"消息

### Q7: FreeSurfer文件在哪里获取？

**A**: FreeSurfer文件获取方法：

1. **使用FreeSurfer软件处理MRI数据**
   - 下载安装FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
   - 运行recon-all命令处理MRI数据
   - 输出文件位于`subjects/[subject_id]/surf/`

2. **使用公开数据集**
   - Human Connectome Project
   - OpenNeuro
   - 等神经影像数据库

3. **没有FreeSurfer数据**
   - 使用默认球体模型
   - 功能完全相同，只是视觉效果不同

### Q8: 性能很慢，帧率低

**A**: 性能优化建议：

1. **减少脑区数量**
   - 使用较少的脑区划分（如100个而不是200个）

2. **禁用不必要的视觉效果**
   - 取消勾选"Show Connections"
   - 降低Activity Threshold

3. **优化材质**
   - 使用简单的Shader
   - 避免透明材质

4. **关闭抗锯齿**
   - Edit → Project Settings → Quality
   - 降低质量等级

5. **使用对象池**
   - 预创建脑区对象
   - 重用而不是每帧创建

### Q9: 能否在WebGL中使用？

**A**: 可以，但有限制：

1. **WebSocket支持**
   - WebGL平台使用浏览器的WebSocket API
   - 需要额外的JavaScript插件

2. **文件访问**
   - 只能访问StreamingAssets中的文件
   - 不能访问本地文件系统

3. **性能限制**
   - WebGL性能通常低于桌面平台
   - 建议减少脑区数量

### Q10: 如何添加新的可视化功能？

**A**: 扩展步骤：

1. **创建新脚本**
   - 继承MonoBehaviour
   - 添加到BrainManager对象

2. **访问BrainVisualization**
   ```csharp
   var visualization = GetComponent<BrainVisualization>();
   var regions = visualization.GetRegions();
   ```

3. **实现自定义逻辑**
   ```csharp
   // 例如：高亮特定脑区
   void HighlightRegion(int regionId)
   {
       var region = visualization.GetRegion(regionId);
       if (region != null)
       {
           region.GetComponent<Renderer>().material.color = Color.yellow;
       }
   }
   ```

---

## 附录

### 文件格式说明

#### JSON状态文件格式

```json
{
    "regions": {
        "0": {
            "id": 0,
            "activity": 0.75,
            "position": [0.0, 0.0, 0.0],
            "label": "Visual_1"
        },
        "1": {
            "id": 1,
            "activity": 0.62,
            "position": [1.5, 0.5, 0.0],
            "label": "Visual_2"
        }
    },
    "connections": [
        {
            "source": 0,
            "target": 1,
            "weight": 0.8,
            "delay": 0.05
        }
    ],
    "metadata": {
        "timestamp": 1234567890,
        "model_version": "2.5",
        "num_regions": 200
    }
}
```

#### OBJ文件命名规则

- 格式：`region_XXXX.obj`
- XXXX：四位数字，零填充
- 例如：`region_0001.obj`, `region_0002.obj`, ..., `region_0200.obj`

### 相关命令速查

```bash
# 完整安装（带FreeSurfer）
python unity_one_click_install.py \
    --unity-project /path/to/UnityProject \
    --freesurfer-dir /path/to/freesurfer

# 基础安装（无FreeSurfer）
python unity_one_click_install.py \
    --unity-project /path/to/UnityProject

# 启动实时服务器
python unity_startup.py --model results/hetero_gnn_trained.pt

# 转换缓存文件
python -m unity_integration.brain_state_exporter \
    --cache-dir brain_data/cache \
    --output-dir brain_data/model_output

# 生成OBJ文件（单独运行）
python setup_unity_project.py \
    --freesurfer-dir /path/to/freesurfer \
    --output-dir ./output

# 仅安装到Unity（单独运行）
python unity_package_installer.py \
    --unity-project /path/to/UnityProject
```

### Unity快捷键

- **Play/Stop**: Ctrl+P (Windows) / Cmd+P (Mac)
- **Pause**: Ctrl+Shift+P
- **Console**: Ctrl+Shift+C
- **Inspector**: Ctrl+3
- **Scene View**: Ctrl+1
- **Game View**: Ctrl+2

---

## 技术支持

如遇到问题，请：

1. 查看本指南的"常见问题"部分
2. 查看Unity Console的错误信息
3. 查看Python终端的日志输出
4. 在GitHub上创建Issue：https://github.com/sheinclotho/twinbrain/issues

**提供以下信息有助于快速解决问题**：
- Unity版本
- Python版本
- 操作系统
- 完整的错误信息
- 重现步骤

---

**最后更新**: 2024-02-15  
**文档版本**: 4.0  
**作者**: TwinBrain团队
