# Unity 集成示例

本目录包含用于在 Unity 3D 中可视化 TwinBrain 脑状态的 Unity C# 脚本。

## 📁 文件说明

### 核心脚本

1. **BrainVisualization.cs**
   - 主要的大脑可视化控制器
   - 加载和渲染 200 个脑区
   - 处理时间序列动画
   - 根据活动值更新颜色和大小

2. **BrainConfigLoader.cs**
   - 加载 JSON 配置文件
   - 解析脑区位置和网络信息
   - 管理材质和颜色映射

3. **WebSocketClient.cs**
   - 与 Python 后端实时通信
   - 接收实时脑状态更新
   - 处理虚拟刺激命令

## 🚀 快速开始

### 步骤 1：设置 Unity 项目

1. 创建新的 Unity 3D 项目（推荐 Unity 2021.3 LTS 或更高版本）
2. 将此目录中的所有 `.cs` 文件复制到 `Assets/Scripts/` 文件夹

### 步骤 2：导入脑数据

运行 Python 导出脚本：

```bash
python unity_automation.py --mode export --output unity_data
```

这会生成：
- `unity_data/json/` - 时间序列脑状态数据
- `unity_data/obj/` - 3D 脑模型
- `unity_data/unity_config.json` - Unity 配置

### 步骤 3：在 Unity 中设置场景

1. **创建空的 GameObject**：
   - 命名为 "BrainVisualizationManager"
   - 添加 `BrainVisualization.cs` 组件

2. **配置组件**：
   - **JSON Directory**: 指向 `unity_data/json/`
   - **Config File**: 指向 `unity_data/unity_config.json`
   - **Region Prefab**: 创建球体预制件
   - **Material**: 创建标准材质

3. **导入 OBJ 模型**（可选）：
   - 将 `brain_regions.obj` 拖入 Assets
   - 在场景中实例化

### 步骤 4：运行可视化

点击 Unity 播放按钮。您应该看到：
- 200 个脑区球体
- 根据活动值变化的颜色
- 时间序列动画（如果启用）

## 📖 使用说明

### BrainVisualization 组件

**公共属性**：

```csharp
public class BrainVisualization : MonoBehaviour
{
    // 数据路径
    public string jsonDirectory = "unity_data/json";
    public string configFilePath = "unity_data/unity_config.json";
    
    // 预制件和材质
    public GameObject regionPrefab;  // 球体预制件
    public Material regionMaterial;  // 脑区材质
    
    // 可视化设置
    public float regionScale = 1.0f;  // 脑区大小缩放
    public float activityThreshold = 0.3f;  // 活动阈值
    
    // 动画设置
    public bool autoPlay = true;  // 自动播放动画
    public float fps = 10f;  // 每秒帧数
}
```

**公共方法**：

```csharp
// 加载特定时间点的脑状态
public void LoadBrainState(int timePoint);

// 播放/暂停动画
public void PlayAnimation();
public void PauseAnimation();

// 设置活动阈值
public void SetActivityThreshold(float threshold);

// 更新单个脑区的活动
public void UpdateRegionActivity(int regionId, float activity);
```

### BrainConfigLoader 组件

**用途**：加载和解析配置文件

```csharp
public class BrainConfigLoader
{
    // 加载配置
    public BrainConfig LoadConfig(string configPath);
    
    // 获取脑区信息
    public RegionInfo GetRegionInfo(int regionId);
    
    // 获取网络信息
    public NetworkInfo GetNetworkInfo(string networkName);
}
```

**配置文件格式**：

```json
{
  "atlas": "Schaefer200",
  "regions": [
    {
      "id": 1,
      "label": "7Networks_LH_Vis_1",
      "position": [-10.5, -85.2, 5.1],
      "network": "Visual",
      "hemisphere": "left"
    }
  ],
  "networks": {
    "Visual": {"color": [120, 18, 134]},
    "Somatomotor": {"color": [70, 130, 180]}
  }
}
```

### WebSocketClient 组件

**用途**：实时通信

```csharp
public class WebSocketClient : MonoBehaviour
{
    public string serverUrl = "ws://localhost:8765";
    
    // 连接到服务器
    public async Task Connect();
    
    // 发送消息
    public async Task SendMessage(string message);
    
    // 接收消息的回调
    public event Action<string> OnMessageReceived;
}
```

**使用示例**：

```csharp
// 连接到 Python 后端
WebSocketClient client = GetComponent<WebSocketClient>();
await client.Connect();

// 监听消息
client.OnMessageReceived += (message) => {
    BrainState state = JsonUtility.FromJson<BrainState>(message);
    UpdateVisualization(state);
};

// 发送刺激请求
var stimRequest = new {
    type = "stimulation",
    target_regions = new int[] {10, 15, 20},
    amplitude = 0.5
};
await client.SendMessage(JsonUtility.ToJson(stimRequest));
```

## 🎨 自定义可视化

### 颜色映射

修改颜色梯度：

```csharp
public class CustomColorMapper
{
    // 活动值到颜色
    public Color ActivityToColor(float activity)
    {
        // 低活动：蓝色
        // 高活动：红色
        if (activity < 0.3f)
            return Color.blue;
        else if (activity < 0.7f)
            return Color.yellow;
        else
            return Color.red;
    }
}
```

### 大小映射

根据活动调整脑区大小：

```csharp
public void UpdateRegionSize(GameObject region, float activity)
{
    // 活动越高，球体越大
    float scale = 1.0f + activity * 2.0f;
    region.transform.localScale = Vector3.one * scale;
}
```

### 网络着色

按网络给脑区着色：

```csharp
// 网络颜色
Dictionary<string, Color> networkColors = new Dictionary<string, Color>()
{
    {"Visual", new Color(120/255f, 18/255f, 134/255f)},
    {"Somatomotor", new Color(70/255f, 130/255f, 180/255f)},
    {"Dorsal Attention", new Color(0/255f, 118/255f, 14/255f)},
    {"Ventral Attention", new Color(196/255f, 58/255f, 250/255f)},
    {"Limbic", new Color(220/255f, 248/255f, 164/255f)},
    {"Frontoparietal", new Color(230/255f, 148/255f, 34/255f)},
    {"Default Mode", new Color(205/255f, 62/255f, 78/255f)}
};

// 应用网络颜色
public void ColorByNetwork(GameObject region, string network)
{
    Renderer renderer = region.GetComponent<Renderer>();
    renderer.material.color = networkColors[network];
}
```

## 🔧 高级功能

### 连接可视化

显示脑区之间的连接：

```csharp
public class ConnectionVisualizer
{
    public void DrawConnection(Vector3 start, Vector3 end, float strength)
    {
        LineRenderer line = CreateLine();
        line.SetPositions(new Vector3[] {start, end});
        line.startWidth = strength * 0.1f;
        line.endWidth = strength * 0.1f;
        
        // 颜色基于强度
        Color color = Color.Lerp(Color.white, Color.yellow, strength);
        line.material.color = color;
    }
}
```

### 交互控制

添加鼠标交互：

```csharp
public class BrainInteraction : MonoBehaviour
{
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit))
            {
                // 点击了脑区
                BrainRegion region = hit.collider.GetComponent<BrainRegion>();
                if (region != null)
                {
                    ShowRegionInfo(region);
                }
            }
        }
    }
    
    void ShowRegionInfo(BrainRegion region)
    {
        Debug.Log($"Region {region.id}: {region.label}");
        Debug.Log($"Activity: {region.currentActivity}");
        Debug.Log($"Network: {region.network}");
    }
}
```

### 时间控制

添加时间轴控制：

```csharp
public class TimelineController : MonoBehaviour
{
    public Slider timelineSlider;
    private BrainVisualization brain;
    
    void Start()
    {
        brain = GetComponent<BrainVisualization>();
        timelineSlider.onValueChanged.AddListener(OnTimelineChanged);
    }
    
    void OnTimelineChanged(float value)
    {
        // 值范围 0-1，映射到时间点
        int maxTime = brain.GetMaxTimePoint();
        int timePoint = Mathf.RoundToInt(value * maxTime);
        brain.LoadBrainState(timePoint);
    }
}
```

## 📊 性能优化

### 对象池

重用游戏对象：

```csharp
public class RegionPool
{
    private Queue<GameObject> pool = new Queue<GameObject>();
    
    public GameObject Get()
    {
        if (pool.Count > 0)
            return pool.Dequeue();
        else
            return Instantiate(regionPrefab);
    }
    
    public void Return(GameObject obj)
    {
        obj.SetActive(false);
        pool.Enqueue(obj);
    }
}
```

### LOD（细节层次）

距离相机较远时简化：

```csharp
public void UpdateLOD(GameObject region, float distance)
{
    if (distance > 50f)
    {
        // 远距离：低细节
        region.GetComponent<MeshFilter>().mesh = lowDetailMesh;
    }
    else
    {
        // 近距离：高细节
        region.GetComponent<MeshFilter>().mesh = highDetailMesh;
    }
}
```

## 🐛 故障排除

### 问题：脑区不显示

**检查**：
1. JSON 文件路径是否正确？
2. 配置文件是否存在？
3. 预制件和材质是否分配？

### 问题：颜色不变化

**检查**：
1. 活动数据是否正确加载？
2. 材质是否支持颜色变化？
3. 着色器是否正确？

### 问题：WebSocket 连接失败

**检查**：
1. Python 后端是否运行？
   ```bash
   python unity_automation.py --mode server
   ```
2. 端口是否正确（默认 8765）？
3. 防火墙是否阻止？

## 📚 示例场景

### 示例 1：静态脑可视化

```csharp
// 加载单个时间点
brain.LoadBrainState(0);
brain.autoPlay = false;
```

### 示例 2：动画播放

```csharp
// 循环播放动画
brain.autoPlay = true;
brain.fps = 10f;
brain.PlayAnimation();
```

### 示例 3：实时更新

```csharp
// 连接 WebSocket
WebSocketClient client = GetComponent<WebSocketClient>();
await client.Connect();

client.OnMessageReceived += (message) => {
    var state = JsonUtility.FromJson<BrainState>(message);
    foreach (var region in state.regions)
    {
        brain.UpdateRegionActivity(region.id, region.activity);
    }
};
```

## 🔗 相关资源

- [Unity 工作流说明](../docs/Unity工作流说明.md)
- [Unity 快速开始指南](../docs/Unity快速开始指南.md)
- [TwinBrain 系统使用指南](../docs/TwinBrain系统使用指南.md)

## 📞 支持

如有问题，请参考：
- [GitHub Issues](https://github.com/sheinclotho/twinbrain/issues)
- [文档](../docs/)

---

**维护者**：TwinBrain Development Team  
**最后更新**：2026-02-04  
**Unity 版本要求**：Unity 2021.3 LTS 或更高
