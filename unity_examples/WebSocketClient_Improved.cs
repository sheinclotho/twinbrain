using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace TwinBrain
{
    /// <summary>
    /// TwinBrain WebSocket客户端 - 改进版
    /// Unity 2019+ 兼容版本
    /// 
    /// 连接到TwinBrain WebSocket服务器以获取实时大脑状态更新。
    /// 
    /// **重要说明**:
    /// 虽然命名为"WebSocket"客户端，但此实现使用HTTP/REST通信而非真正的WebSocket协议。
    /// 这是因为Unity原生不支持WebSocket，使用HTTP可以：
    /// - 兼容所有Unity平台（包括WebGL）
    /// - 无需第三方插件
    /// - 简化部署和配置
    /// 
    /// 如需真正的WebSocket支持（双向实时通信），请考虑：
    /// - WebSocketSharp插件（独立平台）
    /// - NativeWebSocket插件（多平台）
    /// - 浏览器原生WebSocket API（仅WebGL）
    /// 
    /// 改进内容:
    /// - 使用UnityWebRequest实现完整的HTTP通信
    /// - 添加指数退避重连机制
    /// - 完善的错误处理和日志记录
    /// - 请求ID追踪，匹配异步响应
    /// - 输入验证和边界检查
    /// - 连接状态管理
    /// 
    /// 功能:
    /// - 获取当前大脑状态
    /// - 请求预测
    /// - 模拟刺激
    /// - 流式传输大脑活动
    /// - Cache文件转换
    /// 
    /// 使用方法:
    /// 1. 启动TwinBrain服务器: python unity_startup.py
    /// 2. 将此脚本附加到GameObject
    /// 3. 配置服务器URL（默认: http://localhost:8765）
    /// 4. 脚本将在启动时自动连接
    /// </summary>
    public class WebSocketClientImproved : MonoBehaviour
    {
        [Header("连接设置")]
        [Tooltip("后端服务器URL (使用HTTP协议，非真正的WebSocket)")]
        public string serverUrl = "http://localhost:8765";
        
        [Tooltip("启动时自动连接")]
        public bool autoConnect = true;
        
        [Tooltip("断开连接时自动重连")]
        public bool autoReconnect = true;
        
        [Header("重连设置")]
        [Tooltip("初始重连延迟（秒）")]
        public float initialReconnectDelay = 2f;
        
        [Tooltip("最大重连延迟（秒）")]
        public float maxReconnectDelay = 60f;
        
        [Tooltip("重连延迟倍增因子")]
        public float reconnectDelayMultiplier = 2f;
        
        [Tooltip("最大重连尝试次数（0=无限）")]
        public int maxReconnectAttempts = 10;
        
        [Header("超时设置")]
        [Tooltip("请求超时时间（秒）")]
        public int requestTimeout = 30;
        
        [Tooltip("连接健康检查间隔（秒）")]
        public float healthCheckInterval = 10f;
        
        [Header("状态")]
        [SerializeField] private ConnectionState _connectionState = ConnectionState.Disconnected;
        public string lastError = "";
        public int reconnectAttempt = 0;
        
        // 连接状态枚举
        public enum ConnectionState
        {
            Disconnected,
            Connecting,
            Connected,
            Reconnecting,
            Failed
        }
        
        // 事件
        public delegate void ConnectedHandler();
        public delegate void DisconnectedHandler();
        public delegate void ErrorHandler(string error);
        public delegate void BrainStateReceivedHandler(BrainStateData state);
        public delegate void MessageReceivedHandler(JObject message);
        
        public event ConnectedHandler OnConnected;
        public event DisconnectedHandler OnDisconnected;
        public event ErrorHandler OnError;
        public event BrainStateReceivedHandler OnBrainStateReceived;
        public event MessageReceivedHandler OnMessageReceived;
        
        // 配置常量
        private const int MAX_REGION_ID = 199;  // 最大脑区ID (0-199 = 200个脑区)
        private Queue<string> messageQueue = new Queue<string>();
        private Dictionary<string, Action<JObject>> pendingRequests = new Dictionary<string, Action<JObject>>();
        private Coroutine healthCheckCoroutine = null;
        private float currentReconnectDelay;
        private int requestIdCounter = 0;
        
        // 属性
        public ConnectionState State
        {
            get { return _connectionState; }
            private set
            {
                if (_connectionState != value)
                {
                    _connectionState = value;
                    Debug.Log($"[WebSocket] 状态变更: {value}");
                }
            }
        }
        
        public bool IsConnected()
        {
            return State == ConnectionState.Connected;
        }
        
        void Start()
        {
            currentReconnectDelay = initialReconnectDelay;
            
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
            ProcessMessages();
        }
        
        /// <summary>
        /// 连接到服务器
        /// </summary>
        public void Connect()
        {
            if (State == ConnectionState.Connected || State == ConnectionState.Connecting)
            {
                Debug.LogWarning("[WebSocket] 已经连接或正在连接");
                return;
            }
            
            State = ConnectionState.Connecting;
            Debug.Log($"[WebSocket] 连接到 {serverUrl}...");
            
            // 测试连接
            StartCoroutine(TestConnection());
        }
        
        /// <summary>
        /// 测试连接
        /// </summary>
        private IEnumerator TestConnection()
        {
            // 发送一个简单的健康检查请求
            using (UnityWebRequest www = new UnityWebRequest(serverUrl, "GET"))
            {
                www.downloadHandler = new DownloadHandlerBuffer();
                www.timeout = requestTimeout;
                
                yield return www.SendWebRequest();
                
                #if UNITY_2020_1_OR_NEWER
                if (www.result == UnityWebRequest.Result.Success)
                #else
                if (!www.isNetworkError && !www.isHttpError)
                #endif
                {
                    // 连接成功
                    State = ConnectionState.Connected;
                    reconnectAttempt = 0;
                    currentReconnectDelay = initialReconnectDelay;
                    
                    Debug.Log("[WebSocket] 连接成功");
                    
                    if (OnConnected != null)
                    {
                        OnConnected();
                    }
                    
                    // 启动健康检查
                    if (healthCheckCoroutine != null)
                    {
                        StopCoroutine(healthCheckCoroutine);
                    }
                    healthCheckCoroutine = StartCoroutine(HealthCheckRoutine());
                }
                else
                {
                    // 连接失败
                    HandleConnectionError($"连接失败: {www.error}");
                }
            }
        }
        
        /// <summary>
        /// 健康检查协程
        /// </summary>
        private IEnumerator HealthCheckRoutine()
        {
            while (State == ConnectionState.Connected)
            {
                yield return new WaitForSeconds(healthCheckInterval);
                
                // 发送简单的ping请求
                using (UnityWebRequest www = new UnityWebRequest(serverUrl, "GET"))
                {
                    www.downloadHandler = new DownloadHandlerBuffer();
                    www.timeout = 5;
                    
                    yield return www.SendWebRequest();
                    
                    #if UNITY_2020_1_OR_NEWER
                    if (www.result != UnityWebRequest.Result.Success)
                    #else
                    if (www.isNetworkError || www.isHttpError)
                    #endif
                    {
                        Debug.LogWarning("[WebSocket] 健康检查失败，连接可能断开");
                        HandleDisconnection();
                        yield break;
                    }
                }
            }
        }
        
        /// <summary>
        /// 处理连接错误
        /// </summary>
        private void HandleConnectionError(string error)
        {
            lastError = error;
            Debug.LogError($"[WebSocket] {error}");
            
            if (OnError != null)
            {
                OnError(error);
            }
            
            // 尝试重连
            if (autoReconnect)
            {
                reconnectAttempt++;
                
                if (maxReconnectAttempts == 0 || reconnectAttempt <= maxReconnectAttempts)
                {
                    State = ConnectionState.Reconnecting;
                    StartCoroutine(ReconnectCoroutine());
                }
                else
                {
                    State = ConnectionState.Failed;
                    Debug.LogError($"[WebSocket] 达到最大重连次数 ({maxReconnectAttempts})，放弃重连");
                }
            }
            else
            {
                State = ConnectionState.Disconnected;
            }
        }
        
        /// <summary>
        /// 处理断开连接
        /// </summary>
        private void HandleDisconnection()
        {
            State = ConnectionState.Disconnected;
            
            if (healthCheckCoroutine != null)
            {
                StopCoroutine(healthCheckCoroutine);
                healthCheckCoroutine = null;
            }
            
            if (OnDisconnected != null)
            {
                OnDisconnected();
            }
            
            if (autoReconnect)
            {
                StartCoroutine(ReconnectCoroutine());
            }
        }
        
        /// <summary>
        /// 重连协程（指数退避）
        /// </summary>
        private IEnumerator ReconnectCoroutine()
        {
            float delay = Mathf.Min(currentReconnectDelay, maxReconnectDelay);
            Debug.Log($"[WebSocket] {delay}秒后重新连接... (尝试 {reconnectAttempt}/{maxReconnectAttempts})");
            
            yield return new WaitForSeconds(delay);
            
            // 指数退避
            currentReconnectDelay *= reconnectDelayMultiplier;
            
            Connect();
        }
        
        /// <summary>
        /// 从服务器断开连接
        /// </summary>
        public void Disconnect()
        {
            if (State == ConnectionState.Disconnected)
                return;
            
            Debug.Log("[WebSocket] 断开连接...");
            
            if (healthCheckCoroutine != null)
            {
                StopCoroutine(healthCheckCoroutine);
                healthCheckCoroutine = null;
            }
            
            State = ConnectionState.Disconnected;
            
            if (OnDisconnected != null)
            {
                OnDisconnected();
            }
        }
        
        /// <summary>
        /// 向服务器发送请求
        /// </summary>
        public void SendRequest(string requestType, JObject parameters = null, Action<JObject> callback = null)
        {
            if (!IsConnected())
            {
                Debug.LogWarning("[WebSocket] 未连接到服务器");
                if (callback != null)
                {
                    callback(CreateErrorResponse("未连接到服务器"));
                }
                return;
            }
            
            // 验证请求类型
            if (string.IsNullOrEmpty(requestType))
            {
                Debug.LogError("[WebSocket] 请求类型不能为空");
                if (callback != null)
                {
                    callback(CreateErrorResponse("请求类型不能为空"));
                }
                return;
            }
            
            JObject request = new JObject();
            request["type"] = requestType;
            
            // 生成唯一请求ID
            string requestId = $"req_{requestIdCounter++}_{DateTime.Now.Ticks}";
            request["request_id"] = requestId;
            
            if (parameters != null)
            {
                foreach (KeyValuePair<string, JToken> kvp in parameters)
                {
                    request[kvp.Key] = kvp.Value;
                }
            }
            
            // 注册回调
            if (callback != null)
            {
                pendingRequests[requestId] = callback;
            }
            
            string json = request.ToString();
            StartCoroutine(SendRequestCoroutine(json, requestId));
        }
        
        /// <summary>
        /// 发送请求协程
        /// </summary>
        private IEnumerator SendRequestCoroutine(string jsonData, string requestId)
        {
            Debug.Log($"[WebSocket] 发送请求: {jsonData.Substring(0, Math.Min(100, jsonData.Length))}...");
            
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
            
            using (UnityWebRequest www = new UnityWebRequest(serverUrl, "POST"))
            {
                www.uploadHandler = new UploadHandlerRaw(bodyRaw);
                www.downloadHandler = new DownloadHandlerBuffer();
                www.SetRequestHeader("Content-Type", "application/json");
                www.timeout = requestTimeout;
                
                yield return www.SendWebRequest();
                
                #if UNITY_2020_1_OR_NEWER
                if (www.result == UnityWebRequest.Result.Success)
                #else
                if (!www.isNetworkError && !www.isHttpError)
                #endif
                {
                    try
                    {
                        string response = www.downloadHandler.text;
                        JObject responseObj = JObject.Parse(response);
                        
                        // 调用回调
                        if (pendingRequests.ContainsKey(requestId))
                        {
                            pendingRequests[requestId](responseObj);
                            pendingRequests.Remove(requestId);
                        }
                        
                        // 加入消息队列
                        messageQueue.Enqueue(response);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"[WebSocket] 解析响应失败: {e.Message}");
                        
                        if (pendingRequests.ContainsKey(requestId))
                        {
                            pendingRequests[requestId](CreateErrorResponse($"解析响应失败: {e.Message}"));
                            pendingRequests.Remove(requestId);
                        }
                    }
                }
                else
                {
                    string error = $"请求失败: {www.error}";
                    Debug.LogError($"[WebSocket] {error}");
                    
                    if (pendingRequests.ContainsKey(requestId))
                    {
                        pendingRequests[requestId](CreateErrorResponse(error));
                        pendingRequests.Remove(requestId);
                    }
                    
                    // 检查是否需要重连
                    #if UNITY_2020_1_OR_NEWER
                    if (www.result == UnityWebRequest.Result.ConnectionError)
                    #else
                    if (www.isNetworkError)
                    #endif
                    {
                        HandleDisconnection();
                    }
                }
            }
        }
        
        /// <summary>
        /// 创建错误响应
        /// </summary>
        private JObject CreateErrorResponse(string message)
        {
            JObject error = new JObject();
            error["type"] = "error";
            error["success"] = false;
            error["message"] = message;
            return error;
        }
        
        /// <summary>
        /// 请求当前大脑状态
        /// </summary>
        public void GetBrainState(Action<JObject> callback = null)
        {
            SendRequest("get_state", null, callback);
        }
        
        /// <summary>
        /// 请求未来预测
        /// </summary>
        public void RequestPrediction(int nSteps = 10, Action<JObject> callback = null)
        {
            // 输入验证
            if (nSteps <= 0 || nSteps > 1000)
            {
                Debug.LogWarning($"[WebSocket] 无效的预测步数: {nSteps}，使用默认值10");
                nSteps = 10;
            }
            
            JObject parameters = new JObject();
            parameters["n_steps"] = nSteps;
            SendRequest("predict", parameters, callback);
        }
        
        /// <summary>
        /// 请求刺激模拟
        /// </summary>
        public void SimulateStimulation(int[] targetRegions, float amplitude, string pattern = "sine", Action<JObject> callback = null)
        {
            // 输入验证
            if (targetRegions == null || targetRegions.Length == 0)
            {
                Debug.LogWarning("[WebSocket] 目标脑区为空");
                if (callback != null)
                {
                    callback(CreateErrorResponse("目标脑区为空"));
                }
                return;
            }
            
            if (amplitude <= 0 || amplitude > 10)
            {
                Debug.LogWarning($"[WebSocket] 无效的刺激幅度: {amplitude}，限制在0-10范围内");
                amplitude = Mathf.Clamp(amplitude, 0.01f, 10f);
            }
            
            // 验证脑区ID
            List<int> validRegions = new List<int>();
            foreach (int region in targetRegions)
            {
                if (region >= 0 && region <= MAX_REGION_ID)
                {
                    validRegions.Add(region);
                }
                else
                {
                    Debug.LogWarning($"[WebSocket] 忽略无效脑区ID: {region}（有效范围: 0-{MAX_REGION_ID}）");
                }
            }
            
            if (validRegions.Count == 0)
            {
                Debug.LogWarning("[WebSocket] 没有有效的目标脑区");
                if (callback != null)
                {
                    callback(CreateErrorResponse("没有有效的目标脑区"));
                }
                return;
            }
            
            JObject stimulation = new JObject();
            stimulation["target_regions"] = new JArray(validRegions);
            stimulation["amplitude"] = amplitude;
            stimulation["pattern"] = pattern;
            
            JObject parameters = new JObject();
            parameters["stimulation"] = stimulation;
            
            SendRequest("simulate", parameters, callback);
        }
        
        /// <summary>
        /// 开始流式传输大脑活动
        /// </summary>
        public void StartStream(int fps = 10, int duration = 60, Action<JObject> callback = null)
        {
            // 输入验证
            if (fps <= 0 || fps > 60)
            {
                Debug.LogWarning($"[WebSocket] 无效的FPS: {fps}，限制在1-60范围内");
                fps = Mathf.Clamp(fps, 1, 60);
            }
            
            if (duration <= 0 || duration > 3600)
            {
                Debug.LogWarning($"[WebSocket] 无效的持续时间: {duration}，限制在1-3600秒范围内");
                duration = Mathf.Clamp(duration, 1, 3600);
            }
            
            JObject parameters = new JObject();
            parameters["fps"] = fps;
            parameters["duration"] = duration;
            
            SendRequest("stream_start", parameters, callback);
        }
        
        /// <summary>
        /// 停止流式传输
        /// </summary>
        public void StopStream(Action<JObject> callback = null)
        {
            SendRequest("stream_stop", null, callback);
        }
        
        /// <summary>
        /// 转换Cache文件到JSON
        /// </summary>
        public void ConvertCacheToJson(string cacheDir, string outputDir, Action<JObject> callback = null)
        {
            // 输入验证
            if (string.IsNullOrEmpty(cacheDir) || string.IsNullOrEmpty(outputDir))
            {
                Debug.LogError("[WebSocket] cache_dir和output_dir不能为空");
                if (callback != null)
                {
                    callback(CreateErrorResponse("cache_dir和output_dir不能为空"));
                }
                return;
            }
            
            JObject parameters = new JObject();
            parameters["cache_dir"] = cacheDir;
            parameters["output_dir"] = outputDir;
            
            SendRequest("convert_cache", parameters, callback);
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
                
                JToken typeToken = data["type"];
                string msgType = (typeToken != null) ? typeToken.ToString() : null;
                
                if (OnMessageReceived != null)
                {
                    OnMessageReceived(data);
                }
                
                switch (msgType)
                {
                    case "welcome":
                        Debug.Log($"[WebSocket] 已连接: {data["message"]}");
                        break;
                    
                    case "brain_state":
                        HandleBrainState(data);
                        break;
                    
                    case "prediction":
                        Debug.Log($"[WebSocket] 收到预测 {data["n_steps"]} 步");
                        break;
                    
                    case "simulation":
                        Debug.Log("[WebSocket] 收到模拟结果");
                        HandleBrainState(data);
                        break;
                    
                    case "stream_frame":
                        HandleStreamFrame(data);
                        break;
                    
                    case "stream_started":
                        Debug.Log($"[WebSocket] 流开始: {data["fps"]} fps, {data["duration"]}s");
                        break;
                    
                    case "stream_ended":
                        Debug.Log($"[WebSocket] 流结束: {data["n_frames"]} 帧");
                        break;
                    
                    case "convert_cache_response":
                        Debug.Log($"[WebSocket] Cache转换完成: {data["message"]}");
                        break;
                    
                    case "error":
                        JToken errorToken = data["message"];
                        string error = (errorToken != null) ? errorToken.ToString() : "Unknown error";
                        Debug.LogError($"[WebSocket] 服务器错误: {error}");
                        lastError = error;
                        if (OnError != null)
                        {
                            OnError(error);
                        }
                        break;
                    
                    default:
                        Debug.Log($"[WebSocket] 未知消息类型: {msgType}");
                        break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[WebSocket] 处理消息时出错: {e.Message}\n{e.StackTrace}");
            }
        }
        
        /// <summary>
        /// 处理大脑状态消息
        /// </summary>
        private void HandleBrainState(JObject data)
        {
            try
            {
                string json = data.ToString();
                BrainStateData brainState = JsonConvert.DeserializeObject<BrainStateData>(json);
                
                if (OnBrainStateReceived != null)
                {
                    OnBrainStateReceived(brainState);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[WebSocket] 无法解析大脑状态: {e.Message}");
            }
        }
        
        /// <summary>
        /// 处理流帧
        /// </summary>
        private void HandleStreamFrame(JObject data)
        {
            JToken frameToken = data["frame"];
            JToken timeToken = data["time"];
            
            int frame = (frameToken != null) ? frameToken.Value<int>() : 0;
            float time = (timeToken != null) ? timeToken.Value<float>() : 0f;
            
            HandleBrainState(data);
        }
        
        /// <summary>
        /// 模拟接收消息（用于测试）
        /// </summary>
        public void SimulateReceiveMessage(string message)
        {
            messageQueue.Enqueue(message);
        }
    }
}
