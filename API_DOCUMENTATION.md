# TwinBrain Unity Integration API 文档 (v2.5)

## 概述

本文档描述TwinBrain Unity集成的完整API规范，包括服务器端点、请求/响应格式、错误处理和使用示例。

**版本**: 2.5  
**协议**: HTTP/REST  
**数据格式**: JSON  
**字符编码**: UTF-8

## 服务器配置

### 默认设置

```python
HOST = "0.0.0.0"  # 监听所有接口
PORT = 8765       # 默认端口
TIMEOUT = 30      # 请求超时（秒）
```

### 安全考虑

**端口配置**:
- 默认端口8765可能需要在防火墙中开放
- 生产环境建议修改为标准HTTP端口（80/443）或非标准端口

**绑定地址**:
- `0.0.0.0`: 监听所有网络接口（包括外网）
  - ⚠️ 仅在受信任网络中使用
  - 适用于多机器测试
- `127.0.0.1`/`localhost`: 仅本地连接
  - ✅ 推荐用于开发和单机使用
  - 更安全，防止外部访问

**推荐配置**:
```bash
# 开发环境（安全）
python unity_startup.py --host 127.0.0.1 --port 8765

# 局域网测试（注意安全）
python unity_startup.py --host 0.0.0.0 --port 8765

# 生产环境（使用反向代理）
python unity_startup.py --host 127.0.0.1 --port 8765
# 然后配置Nginx/Apache作为反向代理，添加HTTPS和认证
```

**防火墙规则**:
```bash
# Linux (iptables)
sudo iptables -A INPUT -p tcp --dport 8765 -j ACCEPT

# Windows (PowerShell)
New-NetFirewallRule -DisplayName "TwinBrain" -Direction Inbound -LocalPort 8765 -Protocol TCP -Action Allow

# macOS (pfctl)
# 编辑 /etc/pf.conf，添加:
# pass in proto tcp from any to any port 8765
```

**其他安全建议**:
- 不要在公网直接暴露此服务
- 使用VPN或SSH隧道进行远程访问
- 考虑添加认证机制（Token, API Key）
- 定期更新依赖包以修复安全漏洞
- 监控异常请求模式

### 启动服务器

```bash
python unity_startup.py --model results/model.pt --port 8765
```

## API端点

### 1. 获取大脑状态 (get_state)

获取当前大脑活动状态。

**请求**:
```json
{
  "type": "get_state",
  "request_id": "req_123_456789"  // 可选
}
```

**成功响应**:
```json
{
  "type": "brain_state",
  "success": true,
  "request_id": "req_123_456789",
  "data": {
    "version": "2.0",
    "timestamp": "2024-02-14T12:00:00",
    "brain_state": {
      "regions": [
        {
          "id": 1,
          "label": "LH_Vis_1",
          "position": {"x": 10.2, "y": 20.4, "z": 30.8},
          "activity": {
            "fmri": {"amplitude": 0.75, "confidence": 0.9}
          }
        }
      ]
    }
  }
}
```

**错误响应**:
```json
{
  "type": "error",
  "success": false,
  "message": "Failed to get state: Model not loaded",
  "request_id": "req_123_456789"
}
```

**Unity C#使用**:
```csharp
wsClient.GetBrainState((response) => {
    if (response["success"].Value<bool>()) {
        var data = response["data"];
        Debug.Log($"收到 {data["brain_state"]["regions"].Count()} 个脑区");
    }
});
```

---

### 2. 预测未来状态 (predict)

请求未来N步的大脑状态预测。

**请求**:
```json
{
  "type": "predict",
  "request_id": "req_124_456790",
  "n_steps": 10  // 预测步数 (1-1000)
}
```

**参数验证**:
- `n_steps`: 整数，范围 1-1000
- 无效值将被限制到有效范围，并记录警告

**成功响应**:
```json
{
  "type": "prediction",
  "success": true,
  "request_id": "req_124_456790",
  "n_steps": 10,
  "predictions": [
    {
      "time_point": 0,
      "time_second": 0.0,
      "brain_state": { /* ... */ }
    },
    // ... 更多时间点
  ],
  "saved_to": "unity_project/brain_data/model_output"
}
```

**错误响应**:
```json
{
  "type": "error",
  "success": false,
  "message": "Invalid n_steps: must be 1-1000",
  "request_id": "req_124_456790"
}
```

**Unity C#使用**:
```csharp
wsClient.RequestPrediction(10, (response) => {
    if (response["success"].Value<bool>()) {
        int nSteps = response["n_steps"].Value<int>();
        var predictions = response["predictions"];
        Debug.Log($"收到 {nSteps} 步预测");
        
        // 处理预测数据
        foreach (var pred in predictions) {
            // ...
        }
    } else {
        Debug.LogError($"预测失败: {response["message"]}");
    }
});
```

---

### 3. 虚拟刺激模拟 (simulate)

模拟对特定脑区的虚拟刺激效果。

**请求**:
```json
{
  "type": "simulate",
  "request_id": "req_125_456791",
  "stimulation": {
    "target_regions": [10, 20, 30],  // 目标脑区ID列表 (0-199)
    "amplitude": 0.5,                // 刺激幅度 (0.01-10.0)
    "pattern": "sine",               // 刺激模式
    "frequency": 10.0,               // 频率 (0.1-100.0 Hz)
    "duration": 20                   // 持续时间 (1-1000 steps)
  }
}
```

**参数验证**:
- `target_regions`: 整数数组，每个值 0-199，至少1个
- `amplitude`: 浮点数，范围 0.01-10.0
- `frequency`: 浮点数，范围 0.1-100.0
- `duration`: 整数，范围 1-1000
- `pattern`: 字符串，支持 "sine", "square", "ramp", "pulse"
- 无效的脑区ID会被自动过滤

**成功响应**:
```json
{
  "type": "simulation",
  "success": true,
  "request_id": "req_125_456791",
  "n_steps": 50,
  "stimulation": {
    "target_regions": [10, 20, 30],
    "amplitude": 0.5,
    "pattern": "sine"
  },
  "responses": [
    {
      "time_point": 0,
      "brain_state": { /* ... */ },
      "stimulation": {
        "active": true,
        "target_regions": [10, 20, 30],
        "amplitude": 0.5
      }
    }
    // ... 更多时间点
  ],
  "saved_to": "unity_project/brain_data/model_output"
}
```

**错误响应**:
```json
{
  "type": "error",
  "success": false,
  "message": "No valid target regions (must be 0-199)",
  "request_id": "req_125_456791"
}
```

**Unity C#使用**:
```csharp
int[] targetRegions = {10, 20, 30};
float amplitude = 0.5f;
string pattern = "sine";

wsClient.SimulateStimulation(targetRegions, amplitude, pattern, (response) => {
    if (response["success"].Value<bool>()) {
        var responses = response["responses"];
        Debug.Log($"模拟完成，{responses.Count()} 个时间点");
        
        // 可视化刺激效果
        foreach (var resp in responses) {
            // ...
        }
    } else {
        Debug.LogError($"模拟失败: {response["message"]}");
    }
});
```

---

### 4. 开始流式传输 (stream_start)

开始连续的大脑活动流式传输。

**请求**:
```json
{
  "type": "stream_start",
  "request_id": "req_126_456792",
  "fps": 10,        // 帧率 (1-60)
  "duration": 60    // 持续时间（秒） (1-3600)
}
```

**参数验证**:
- `fps`: 整数，范围 1-60
- `duration`: 整数，范围 1-3600（1小时）

**成功响应**:
```json
{
  "type": "stream_started",
  "success": true,
  "request_id": "req_126_456792",
  "fps": 10,
  "duration": 60
}
```

**流帧消息** (服务器推送，HTTP模式下不支持):
```json
{
  "type": "stream_frame",
  "frame": 5,
  "time": 0.5,
  "data": {
    "brain_state": { /* ... */ }
  }
}
```

**流结束消息**:
```json
{
  "type": "stream_ended",
  "n_frames": 600
}
```

**Unity C#使用**:
```csharp
wsClient.StartStream(10, 60, (response) => {
    if (response["success"].Value<bool>()) {
        Debug.Log($"流开始: {response["fps"]} FPS, {response["duration"]}秒");
    }
});

// 订阅流帧事件
wsClient.OnMessageReceived += (message) => {
    if (message["type"].ToString() == "stream_frame") {
        int frame = message["frame"].Value<int>();
        var data = message["data"];
        // 更新可视化
    }
};
```

---

### 5. 停止流式传输 (stream_stop)

停止当前的流式传输。

**请求**:
```json
{
  "type": "stream_stop",
  "request_id": "req_127_456793"
}
```

**成功响应**:
```json
{
  "type": "stream_stopped",
  "success": true,
  "request_id": "req_127_456793"
}
```

**Unity C#使用**:
```csharp
wsClient.StopStream((response) => {
    if (response["success"].Value<bool>()) {
        Debug.Log("流已停止");
    }
});
```

---

### 6. Cache转JSON (convert_cache)

将cache文件批量转换为JSON格式。

**请求**:
```json
{
  "type": "convert_cache",
  "request_id": "req_128_456794",
  "cache_dir": "/path/to/cache",
  "output_dir": "/path/to/output"
}
```

**参数验证**:
- `cache_dir`: 字符串，必须存在的目录路径
- `output_dir`: 字符串，目标目录（会自动创建）

**支持的文件格式**:
- `.pt`, `.pth` (PyTorch - 这是实际的缓存格式)

**注意**: 缓存文件由训练过程自动生成，通常命名为：
- `eeg_data.pt` - EEG数据缓存
- `hetero_graphs.pt` - 异构图数据缓存

**成功响应**:
```json
{
  "type": "convert_cache_response",
  "success": true,
  "request_id": "req_128_456794",
  "message": "Successfully converted 5 cache files",
  "converted_count": 5,
  "errors": null,
  "output_dir": "/path/to/output"
}
```

**部分成功响应**:
```json
{
  "type": "convert_cache_response",
  "success": true,
  "converted_count": 3,
  "errors": [
    "Error processing unknown_file.pt: Unsupported cache format"
  ],
  "output_dir": "/path/to/output"
}
```

**错误响应**:
```json
{
  "type": "error",
  "success": false,
  "message": "Cache directory does not exist: /path/to/cache",
  "request_id": "req_128_456794"
}
```

**Unity C#使用**:
```csharp
string cacheDir = Application.streamingAssetsPath + "/brain_data/cache";
string outputDir = Application.streamingAssetsPath + "/brain_data/model_output";

wsClient.ConvertCacheToJson(cacheDir, outputDir, (response) => {
    if (response["success"].Value<bool>()) {
        int count = response["converted_count"].Value<int>();
        Debug.Log($"转换成功: {count} 个文件");
        
        var errors = response["errors"];
        if (errors != null && errors.HasValues) {
            foreach (var error in errors) {
                Debug.LogWarning($"转换警告: {error}");
            }
        }
    } else {
        Debug.LogError($"转换失败: {response["message"]}");
    }
});
```

---

## 错误处理

### 错误响应格式

所有错误响应遵循统一格式：

```json
{
  "type": "error",
  "success": false,
  "message": "详细错误描述",
  "request_id": "req_xxx_xxxxxx",  // 如果请求包含
  "error_code": "ERROR_CODE"        // 可选
}
```

### 错误类型

| 错误代码 | 描述 | HTTP状态码等价 |
|---------|------|---------------|
| `INVALID_REQUEST` | 请求格式无效 | 400 |
| `INVALID_PARAMETERS` | 参数验证失败 | 400 |
| `SERVER_ERROR` | 服务器内部错误 | 500 |
| `MODEL_ERROR` | 模型加载或推理错误 | 500 |
| `FILE_NOT_FOUND` | 文件不存在 | 404 |
| `TIMEOUT` | 请求超时 | 408 |

### Unity端错误处理示例

```csharp
void HandleResponse(JObject response)
{
    if (!response.ContainsKey("success") || 
        !response["success"].Value<bool>())
    {
        string errorMsg = response["message"].ToString();
        string errorCode = response.ContainsKey("error_code") 
            ? response["error_code"].ToString() 
            : "UNKNOWN";
        
        Debug.LogError($"API错误 [{errorCode}]: {errorMsg}");
        
        // 根据错误类型采取行动
        switch (errorCode)
        {
            case "INVALID_PARAMETERS":
                // 提示用户输入错误
                break;
            case "SERVER_ERROR":
                // 尝试重连
                wsClient.Connect();
                break;
            case "MODEL_ERROR":
                // 切换到演示模式
                break;
        }
    }
}
```

---

## 连接管理

### 连接状态

WebSocketClientImproved维护以下连接状态：

```csharp
public enum ConnectionState
{
    Disconnected,   // 未连接
    Connecting,     // 正在连接
    Connected,      // 已连接
    Reconnecting,   // 正在重连
    Failed          // 连接失败
}
```

### 重连策略

- **初始延迟**: 2秒
- **最大延迟**: 60秒
- **倍增因子**: 2x
- **最大尝试**: 10次（可配置）

**指数退避算法**:
```
delay = min(initial_delay * (multiplier ^ attempt), max_delay)
```

示例：
```
尝试1: 2秒
尝试2: 4秒
尝试3: 8秒
尝试4: 16秒
尝试5: 32秒
尝试6+: 60秒
```

### 健康检查

- **间隔**: 10秒
- **超时**: 5秒
- **失败处理**: 自动触发重连

---

## 数据模型

### 配置常量

```csharp
// 在WebSocketClientImproved.cs中定义
private const int MAX_REGION_ID = 199;  // 最大脑区ID (0-199 = 200个脑区)
```

```python
# 在realtime_server.py中定义
MAX_REGIONS = 200  # 脑区总数
```

**说明**: 默认支持200个脑区（Schaefer200图谱）。如需支持其他图谱（如AAL 116, Destrieux 148等），需要修改这些常量。

### BrainStateData

```csharp
[Serializable]
public class BrainStateData
{
    public string version;
    public string timestamp;
    public string subject_id;
    public BrainState brain_state;
    public StimulationInfo stimulation;
}

[Serializable]
public class BrainState
{
    public List<RegionData> regions;
    public List<ConnectionData> connections;
}

[Serializable]
public class RegionData
{
    public int id;
    public string label;
    public PositionData position;
    public ActivityData activity;
}

[Serializable]
public class PositionData
{
    public float x;
    public float y;
    public float z;
}

[Serializable]
public class ActivityData
{
    public ModalityData fmri;
    public ModalityData eeg;
}

[Serializable]
public class ModalityData
{
    public float amplitude;
    public float confidence;
}

[Serializable]
public class ConnectionData
{
    public int source;
    public int target;
    public float strength;
    public string type;
}

[Serializable]
public class StimulationInfo
{
    public bool active;
    public int[] target_regions;
    public float amplitude;
    public string pattern;
}
```

---

## 性能优化

### 客户端优化

```csharp
// 批量请求避免
private float lastRequestTime = 0f;
private const float REQUEST_THROTTLE = 0.1f;  // 100ms

public void RequestWithThrottle()
{
    if (Time.time - lastRequestTime < REQUEST_THROTTLE)
    {
        return;  // 忽略过快的请求
    }
    
    lastRequestTime = Time.time;
    // 发送请求
}

// 缓存响应
private Dictionary<string, JObject> responseCache = new Dictionary<string, JObject>();

public void RequestWithCache(string key)
{
    if (responseCache.ContainsKey(key))
    {
        // 使用缓存
        HandleResponse(responseCache[key]);
        return;
    }
    
    // 发送请求并缓存
}
```

### 服务器端优化

Python后端自动实现：
- ✅ 输入验证（避免无效计算）
- ✅ 错误捕获（避免崩溃）
- ✅ 日志记录（便于调试）
- ✅ 异步处理（提高吞吐量）

---

## 版本兼容性

### API版本

- **当前版本**: 2.5
- **JSON格式版本**: 2.0
- **最低Unity版本**: 2019.1

### 向后兼容性

v2.5保持与v2.x的API兼容性，但有以下变更：

**破坏性变更**:
- 无

**新增**:
- `request_id` 字段（可选）
- `success` 字段（所有响应）
- `convert_cache` 端点

**废弃**:
- 无

---

## 测试和调试

### 测试服务器连接

```bash
# 使用curl测试
curl -X POST http://localhost:8765 \
  -H "Content-Type: application/json" \
  -d '{"type": "get_state"}'
```

### Unity测试脚本

```csharp
using UnityEngine;
using TwinBrain;

public class APITester : MonoBehaviour
{
    private WebSocketClientImproved wsClient;
    
    void Start()
    {
        wsClient = GetComponent<WebSocketClientImproved>();
        
        // 测试所有API端点
        StartCoroutine(RunTests());
    }
    
    IEnumerator RunTests()
    {
        yield return new WaitForSeconds(2);
        
        Debug.Log("=== 开始API测试 ===");
        
        // 测试1: get_state
        wsClient.GetBrainState((resp) => {
            Debug.Log($"Test 1 - get_state: {resp["success"]}");
        });
        yield return new WaitForSeconds(2);
        
        // 测试2: predict
        wsClient.RequestPrediction(5, (resp) => {
            Debug.Log($"Test 2 - predict: {resp["success"]}");
        });
        yield return new WaitForSeconds(2);
        
        // 测试3: simulate
        int[] regions = {10, 20};
        wsClient.SimulateStimulation(regions, 0.5f, "sine", (resp) => {
            Debug.Log($"Test 3 - simulate: {resp["success"]}");
        });
        yield return new WaitForSeconds(2);
        
        Debug.Log("=== API测试完成 ===");
    }
}
```

### 日志级别

Python端：
```python
import logging
logging.getLogger("unity_integration").setLevel(logging.DEBUG)
```

Unity端：
```csharp
// 在WebSocketClientImproved中
public bool debugMode = true;
```

---

## 常见问题

### Q: 为什么不是真正的WebSocket？

A: Unity原生不支持WebSocket。使用HTTP/REST可以：
- 兼容所有Unity平台（包括WebGL）
- 无需第三方插件
- 简化部署

### Q: 如何实现服务器推送？

A: 当前使用请求-响应模式。对于实时更新：
1. 客户端轮询（简单）
2. 集成WebSocketSharp（高级）
3. 使用 `stream_start` 并轮询读取

### Q: 超时时间能修改吗？

A: 可以，在WebSocketClientImproved中：
```csharp
public int requestTimeout = 30;  // 秒
```

---

**文档版本**: 2.5.0  
**最后更新**: 2024-02-14  
**维护者**: TwinBrain Team
