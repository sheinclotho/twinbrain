using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using TwinBrain;

/// <summary>
/// TwinBrain WebSocket客户端
/// 
/// 连接到TwinBrain WebSocket服务器以获取实时大脑状态更新。
/// 
/// 功能:
/// - 获取当前大脑状态
/// - 请求预测
/// - 模拟刺激
/// - 流式传输大脑活动
/// 
/// 使用方法:
/// 1. 启动TwinBrain WebSocket服务器: python -m unity_integration.realtime_server
/// 2. 将此脚本附加到GameObject
/// 3. 配置服务器URL（默认: ws://localhost:8765）
/// 4. 脚本将在启动时自动连接
/// 
/// 注意: 此版本提供WebSocket接口定义。实际的WebSocket连接需要：
/// - WebGL平台: 使用浏览器WebSocket API（需要实现JavaScript插件）
/// - 独立平台: 需要安装WebSocketSharp库 (https://github.com/sta/websocket-sharp)
/// </summary>
public class WebSocketClient : MonoBehaviour
{
    [Header("连接设置")]
    [Tooltip("WebSocket服务器URL")]
    public string serverUrl = "ws://localhost:8765";
    
    [Tooltip("启动时自动连接")]
    public bool autoConnect = true;
    
    [Tooltip("断开连接时自动重连")]
    public bool autoReconnect = true;
    
    [Tooltip("重连延迟（秒）")]
    public float reconnectDelay = 5f;
    
    [Header("状态")]
    public bool isConnected = false;
    public string lastError = "";
    
    // 事件
    public event Action OnConnected;
    public event Action OnDisconnected;
    public event Action<string> OnError;
    public event Action<BrainStateData> OnBrainStateReceived;
    public event Action<JObject> OnMessageReceived;
    
    private Queue<string> messageQueue = new Queue<string>();
    private bool isReconnecting = false;
    
    void Start()
    {
        if (autoConnect)
        {
            Connect();
        }
    }
    
    void OnDestroy()
    {
        Disconnect();
    }
    
    void Update()
    {
        // 处理消息队列
        ProcessMessages();
    }
    
    /// <summary>
    /// 连接到WebSocket服务器
    /// </summary>
    public void Connect()
    {
        if (isConnected)
        {
            Debug.LogWarning("已经连接");
            return;
        }
        
        Debug.Log($"连接到 {serverUrl}...");
        
        // 注意: 实际的WebSocket实现需要平台特定的库
        // 这里提供接口定义，实际连接需要实现或使用第三方库
        Debug.LogWarning("WebSocket连接需要平台特定的实现。" +
                        "请参考文档安装WebSocketSharp（独立平台）或实现WebGL插件。");
    }
    
    /// <summary>
    /// 从服务器断开连接
    /// </summary>
    public void Disconnect()
    {
        if (!isConnected)
            return;
        
        Debug.Log("断开连接...");
        
        isConnected = false;
        OnDisconnected?.Invoke();
    }
    
    /// <summary>
    /// 向服务器发送请求
    /// </summary>
    public void SendRequest(string type, JObject parameters = null)
    {
        if (!isConnected)
        {
            Debug.LogWarning("未连接到服务器");
            return;
        }
        
        JObject request = new JObject();
        request["type"] = type;
        
        if (parameters != null)
        {
            foreach (var kvp in parameters)
            {
                request[kvp.Key] = kvp.Value;
            }
        }
        
        string json = request.ToString();
        SendMessage(json);
    }
    
    /// <summary>
    /// 请求当前大脑状态
    /// </summary>
    public void GetBrainState()
    {
        SendRequest("get_state");
    }
    
    /// <summary>
    /// 请求未来预测
    /// </summary>
    public void RequestPrediction(int nSteps = 10)
    {
        JObject parameters = new JObject();
        parameters["n_steps"] = nSteps;
        SendRequest("predict", parameters);
    }
    
    /// <summary>
    /// 请求刺激模拟
    /// </summary>
    public void SimulateStimulation(int[] targetRegions, float amplitude, string pattern = "sine")
    {
        JObject stimulation = new JObject();
        stimulation["target_regions"] = new JArray(targetRegions);
        stimulation["amplitude"] = amplitude;
        stimulation["pattern"] = pattern;
        
        JObject parameters = new JObject();
        parameters["stimulation"] = stimulation;
        
        SendRequest("simulate", parameters);
    }
    
    /// <summary>
    /// 开始流式传输大脑活动
    /// </summary>
    public void StartStream(int fps = 10, int duration = 60)
    {
        JObject parameters = new JObject();
        parameters["fps"] = fps;
        parameters["duration"] = duration;
        
        SendRequest("stream_start", parameters);
    }
    
    /// <summary>
    /// 停止流式传输
    /// </summary>
    public void StopStream()
    {
        SendRequest("stream_stop");
    }
    
    /// <summary>
    /// 处理接收到的消息
    /// </summary>
    private void ProcessMessages()
    {
        while (messageQueue.Count > 0)
        {
            string message = messageQueue.Dequeue();
            HandleMessage(message);
        }
    }
    
    /// <summary>
    /// 处理接收到的消息
    /// </summary>
    private void HandleMessage(string message)
    {
        try
        {
            JObject data = JObject.Parse(message);
            string type = data["type"]?.ToString();
            
            // 调用通用消息事件
            OnMessageReceived?.Invoke(data);
            
            // 处理特定消息类型
            switch (type)
            {
                case "welcome":
                    Debug.Log($"已连接: {data["message"]}");
                    break;
                
                case "brain_state":
                    HandleBrainState(data);
                    break;
                
                case "prediction":
                    Debug.Log($"收到预测 {data["n_steps"]} 步");
                    break;
                
                case "simulation":
                    Debug.Log("收到模拟结果");
                    break;
                
                case "stream_frame":
                    HandleStreamFrame(data);
                    break;
                
                case "stream_started":
                    Debug.Log($"流开始: {data["fps"]} fps, {data["duration"]}s");
                    break;
                
                case "stream_ended":
                    Debug.Log($"流结束: {data["n_frames"]} 帧");
                    break;
                
                case "error":
                    string error = data["message"]?.ToString();
                    Debug.LogError($"服务器错误: {error}");
                    lastError = error;
                    OnError?.Invoke(error);
                    break;
                
                default:
                    Debug.Log($"未知消息类型: {type}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"处理消息时出错: {e.Message}");
        }
    }
    
    /// <summary>
    /// 处理大脑状态消息
    /// </summary>
    private void HandleBrainState(JObject data)
    {
        try
        {
            // 尝试解析为完整的BrainStateData
            string json = data.ToString();
            BrainStateData brainState = JsonConvert.DeserializeObject<BrainStateData>(json);
            OnBrainStateReceived?.Invoke(brainState);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"无法解析大脑状态: {e.Message}");
        }
    }
    
    /// <summary>
    /// 处理流帧
    /// </summary>
    private void HandleStreamFrame(JObject data)
    {
        int frame = data["frame"]?.Value<int>() ?? 0;
        float time = data["time"]?.Value<float>() ?? 0f;
        // 处理帧数据...
    }
    
    /// <summary>
    /// 发送消息（需要平台特定的实现）
    /// </summary>
    private void SendMessage(string message)
    {
        // 注意: 实际发送需要WebSocket库支持
        // 这里只是接口定义
        Debug.Log($"[WebSocket] 发送: {message}");
    }
    
    /// <summary>
    /// 模拟接收消息（用于测试）
    /// </summary>
    public void SimulateReceiveMessage(string message)
    {
        messageQueue.Enqueue(message);
    }
    
    private IEnumerator ReconnectCoroutine()
    {
        isReconnecting = true;
        Debug.Log($"{reconnectDelay}秒后重新连接...");
        yield return new WaitForSeconds(reconnectDelay);
        Connect();
        isReconnecting = false;
    }
}
