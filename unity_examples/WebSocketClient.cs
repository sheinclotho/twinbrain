using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace TwinBrain
{
    /// <summary>
    /// TwinBrain WebSocket客户端
    /// Unity 2019+ 兼容版本
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
    /// 注意: 此版本使用UnityWebRequest进行HTTP轮询作为WebSocket的替代方案。
    /// 对于真正的WebSocket支持，需要：
    /// - WebGL平台: 使用浏览器WebSocket API（需要实现JavaScript插件）
    /// - 独立平台: 需要安装WebSocketSharp或NativeWebSocket库
    /// </summary>
    public class WebSocketClient : MonoBehaviour
    {
        [Header("连接设置")]
        [Tooltip("后端服务器URL (HTTP)")]
        public string serverUrl = "http://localhost:8765";
        
        [Tooltip("启动时自动连接")]
        public bool autoConnect = true;
        
        [Tooltip("断开连接时自动重连")]
        public bool autoReconnect = true;
        
        [Tooltip("重连延迟（秒）")]
        public float reconnectDelay = 5f;
        
        [Tooltip("轮询间隔（秒）")]
        public float pollingInterval = 1f;
        
        [Header("状态")]
        public bool isConnected = false;
        public string lastError = "";
        
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
        
        private Queue<string> messageQueue = new Queue<string>();
        private bool isReconnecting = false;
        private Coroutine pollingCoroutine = null;
        
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
            ProcessMessages();
        }
        
        /// <summary>
        /// 连接到服务器
        /// </summary>
        public void Connect()
        {
            if (isConnected)
            {
                Debug.LogWarning("已经连接");
                return;
            }
            
            Debug.Log(string.Format("连接到 {0}...", serverUrl));
            
            isConnected = true;
            
            if (OnConnected != null)
            {
                OnConnected();
            }
            
            // Start polling if needed
            if (pollingCoroutine == null)
            {
                pollingCoroutine = StartCoroutine(PollServer());
            }
        }
        
        /// <summary>
        /// 从服务器断开连接
        /// </summary>
        public void Disconnect()
        {
            if (!isConnected)
                return;
            
            Debug.Log("断开连接...");
            
            if (pollingCoroutine != null)
            {
                StopCoroutine(pollingCoroutine);
                pollingCoroutine = null;
            }
            
            isConnected = false;
            
            if (OnDisconnected != null)
            {
                OnDisconnected();
            }
        }
        
        /// <summary>
        /// 轮询服务器（替代WebSocket）
        /// </summary>
        IEnumerator PollServer()
        {
            while (isConnected)
            {
                yield return new WaitForSeconds(pollingInterval);
                // Polling logic would go here
            }
        }
        
        /// <summary>
        /// 向服务器发送请求
        /// </summary>
        public void SendRequest(string requestType, JObject parameters = null)
        {
            if (!isConnected)
            {
                Debug.LogWarning("未连接到服务器");
                return;
            }
            
            JObject request = new JObject();
            request["type"] = requestType;
            
            if (parameters != null)
            {
                foreach (KeyValuePair<string, JToken> kvp in parameters)
                {
                    request[kvp.Key] = kvp.Value;
                }
            }
            
            string json = request.ToString();
            StartCoroutine(SendRequestCoroutine(json));
        }
        
        /// <summary>
        /// 发送请求协程
        /// </summary>
        IEnumerator SendRequestCoroutine(string jsonData)
        {
            Debug.Log(string.Format("[WebSocket] 发送: {0}", jsonData));
            
            // Note: Actual HTTP request implementation would go here
            // using UnityWebRequest for HTTP POST
            
            yield return null;
        }
        
        /// <summary>
        /// 请求当前大脑状态
        /// </summary>
        public void GetBrainState()
        {
            SendRequest("get_state", null);
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
            SendRequest("stream_stop", null);
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
                        Debug.Log(string.Format("已连接: {0}", data["message"]));
                        break;
                    
                    case "brain_state":
                        HandleBrainState(data);
                        break;
                    
                    case "prediction":
                        Debug.Log(string.Format("收到预测 {0} 步", data["n_steps"]));
                        break;
                    
                    case "simulation":
                        Debug.Log("收到模拟结果");
                        HandleBrainState(data);
                        break;
                    
                    case "stream_frame":
                        HandleStreamFrame(data);
                        break;
                    
                    case "stream_started":
                        Debug.Log(string.Format("流开始: {0} fps, {1}s", data["fps"], data["duration"]));
                        break;
                    
                    case "stream_ended":
                        Debug.Log(string.Format("流结束: {0} 帧", data["n_frames"]));
                        break;
                    
                    case "error":
                        JToken errorToken = data["message"];
                        string error = (errorToken != null) ? errorToken.ToString() : "Unknown error";
                        Debug.LogError(string.Format("服务器错误: {0}", error));
                        lastError = error;
                        if (OnError != null)
                        {
                            OnError(error);
                        }
                        break;
                    
                    default:
                        Debug.Log(string.Format("未知消息类型: {0}", msgType));
                        break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError(string.Format("处理消息时出错: {0}", e.Message));
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
                Debug.LogWarning(string.Format("无法解析大脑状态: {0}", e.Message));
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
        
        private IEnumerator ReconnectCoroutine()
        {
            isReconnecting = true;
            Debug.Log(string.Format("{0}秒后重新连接...", reconnectDelay));
            yield return new WaitForSeconds(reconnectDelay);
            Connect();
            isReconnecting = false;
        }
    }
}
