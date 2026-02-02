using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
#if UNITY_WEBGL && !UNITY_EDITOR
using System.Runtime.InteropServices;
#else
// Note: For non-WebGL platforms, you'll need to install a WebSocket library
// such as: WebSocketSharp (https://github.com/sta/websocket-sharp)
// or use Unity's built-in networking
#endif
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

/// <summary>
/// TwinBrain WebSocket Client
/// 
/// Connects to the TwinBrain WebSocket server for real-time brain state updates.
/// 
/// Features:
/// - Get current brain state
/// - Request predictions
/// - Simulate stimulation
/// - Stream brain activity
/// 
/// Requirements:
/// - For desktop/standalone: Install WebSocketSharp or similar
/// - For WebGL: Uses browser WebSocket API
/// 
/// Usage:
/// 1. Start TwinBrain WebSocket server: python -m unity_integration.realtime_server
/// 2. Attach this script to a GameObject
/// 3. Configure server URL (default: ws://localhost:8765)
/// 4. Script will auto-connect on Start
/// </summary>
public class WebSocketClient : MonoBehaviour
{
    [Header("Connection Settings")]
    [Tooltip("WebSocket server URL")]
    public string serverUrl = "ws://localhost:8765";
    
    [Tooltip("Auto-connect on start")]
    public bool autoConnect = true;
    
    [Tooltip("Auto-reconnect if disconnected")]
    public bool autoReconnect = true;
    
    [Tooltip("Reconnect delay (seconds)")]
    public float reconnectDelay = 5f;
    
    [Header("Status")]
    public bool isConnected = false;
    public string lastError = "";
    
    // Events
    public event Action OnConnected;
    public event Action OnDisconnected;
    public event Action<string> OnError;
    public event Action<BrainStateData> OnBrainStateReceived;
    public event Action<JObject> OnMessageReceived;
    
    private Queue<string> messageQueue = new Queue<string>();
    private bool isReconnecting = false;
    
#if !UNITY_WEBGL || UNITY_EDITOR
    // For standalone builds - you'll need to implement WebSocket client
    // using a library like WebSocketSharp
    // private WebSocket ws;
#endif
    
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
        // Process message queue
        ProcessMessages();
    }
    
    /// <summary>
    /// Connect to WebSocket server
    /// </summary>
    public void Connect()
    {
        if (isConnected)
        {
            Debug.LogWarning("Already connected");
            return;
        }
        
        Debug.Log($"Connecting to {serverUrl}...");
        
#if UNITY_WEBGL && !UNITY_EDITOR
        // WebGL implementation
        ConnectWebGL();
#else
        // Standalone implementation
        ConnectStandalone();
#endif
    }
    
    /// <summary>
    /// Disconnect from server
    /// </summary>
    public void Disconnect()
    {
        if (!isConnected)
            return;
        
        Debug.Log("Disconnecting...");
        
#if UNITY_WEBGL && !UNITY_EDITOR
        DisconnectWebGL();
#else
        DisconnectStandalone();
#endif
        
        isConnected = false;
        OnDisconnected?.Invoke();
    }
    
    /// <summary>
    /// Send a request to the server
    /// </summary>
    public void SendRequest(string type, JObject parameters = null)
    {
        if (!isConnected)
        {
            Debug.LogWarning("Not connected to server");
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
    /// Request current brain state
    /// </summary>
    public void GetBrainState()
    {
        SendRequest("get_state");
    }
    
    /// <summary>
    /// Request future prediction
    /// </summary>
    public void RequestPrediction(int nSteps = 10)
    {
        JObject params = new JObject();
        params["n_steps"] = nSteps;
        SendRequest("predict", params);
    }
    
    /// <summary>
    /// Request stimulation simulation
    /// </summary>
    public void SimulateStimulation(int[] targetRegions, float amplitude, string pattern = "sine")
    {
        JObject stimulation = new JObject();
        stimulation["target_regions"] = new JArray(targetRegions);
        stimulation["amplitude"] = amplitude;
        stimulation["pattern"] = pattern;
        
        JObject params = new JObject();
        params["stimulation"] = stimulation;
        
        SendRequest("simulate", params);
    }
    
    /// <summary>
    /// Start streaming brain activity
    /// </summary>
    public void StartStream(int fps = 10, int duration = 60)
    {
        JObject params = new JObject();
        params["fps"] = fps;
        params["duration"] = duration;
        
        SendRequest("stream_start", params);
    }
    
    /// <summary>
    /// Stop streaming
    /// </summary>
    public void StopStream()
    {
        SendRequest("stream_stop");
    }
    
    /// <summary>
    /// Process received messages
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
    /// Handle received message
    /// </summary>
    private void HandleMessage(string message)
    {
        try
        {
            JObject data = JObject.Parse(message);
            string type = data["type"]?.ToString();
            
            // Invoke general message event
            OnMessageReceived?.Invoke(data);
            
            // Handle specific message types
            switch (type)
            {
                case "welcome":
                    Debug.Log($"Connected: {data["message"]}");
                    break;
                
                case "brain_state":
                    HandleBrainState(data);
                    break;
                
                case "prediction":
                    Debug.Log($"Received prediction for {data["n_steps"]} steps");
                    break;
                
                case "simulation":
                    Debug.Log("Received simulation result");
                    break;
                
                case "stream_frame":
                    HandleStreamFrame(data);
                    break;
                
                case "stream_started":
                    Debug.Log($"Stream started: {data["fps"]} fps, {data["duration"]}s");
                    break;
                
                case "stream_ended":
                    Debug.Log($"Stream ended: {data["n_frames"]} frames");
                    break;
                
                case "error":
                    string error = data["message"]?.ToString();
                    Debug.LogError($"Server error: {error}");
                    lastError = error;
                    OnError?.Invoke(error);
                    break;
                
                default:
                    Debug.Log($"Unknown message type: {type}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error handling message: {e.Message}");
        }
    }
    
    /// <summary>
    /// Handle brain state message
    /// </summary>
    private void HandleBrainState(JObject data)
    {
        try
        {
            // Try to parse as full BrainStateData
            string json = data.ToString();
            BrainStateData brainState = JsonConvert.DeserializeObject<BrainStateData>(json);
            OnBrainStateReceived?.Invoke(brainState);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Could not parse brain state: {e.Message}");
        }
    }
    
    /// <summary>
    /// Handle stream frame
    /// </summary>
    private void HandleStreamFrame(JObject data)
    {
        int frame = data["frame"]?.Value<int>() ?? 0;
        float time = data["time"]?.Value<float>() ?? 0f;
        // Process frame data...
    }
    
    // Platform-specific implementations
    
#if UNITY_WEBGL && !UNITY_EDITOR
    
    [DllImport("__Internal")]
    private static extern void WebSocketConnect(string url);
    
    [DllImport("__Internal")]
    private static extern void WebSocketClose();
    
    [DllImport("__Internal")]
    private static extern void WebSocketSend(string message);
    
    private void ConnectWebGL()
    {
        WebSocketConnect(serverUrl);
        // Note: You'll need to implement the JavaScript plugin
        // See Unity WebGL documentation
    }
    
    private void DisconnectWebGL()
    {
        WebSocketClose();
    }
    
    private void SendMessage(string message)
    {
        WebSocketSend(message);
    }
    
#else
    
    private void ConnectStandalone()
    {
        // TODO: Implement using WebSocketSharp or similar library
        /*
        ws = new WebSocket(serverUrl);
        
        ws.OnOpen += (sender, e) =>
        {
            isConnected = true;
            OnConnected?.Invoke();
        };
        
        ws.OnMessage += (sender, e) =>
        {
            messageQueue.Enqueue(e.Data);
        };
        
        ws.OnError += (sender, e) =>
        {
            lastError = e.Message;
            OnError?.Invoke(e.Message);
        };
        
        ws.OnClose += (sender, e) =>
        {
            isConnected = false;
            OnDisconnected?.Invoke();
            
            if (autoReconnect && !isReconnecting)
            {
                StartCoroutine(ReconnectCoroutine());
            }
        };
        
        ws.Connect();
        */
        
        Debug.LogWarning("WebSocket implementation not available. Install WebSocketSharp or similar.");
    }
    
    private void DisconnectStandalone()
    {
        /*
        if (ws != null)
        {
            ws.Close();
            ws = null;
        }
        */
    }
    
    private void SendMessage(string message)
    {
        /*
        if (ws != null && ws.ReadyState == WebSocketState.Open)
        {
            ws.Send(message);
        }
        */
    }
    
#endif
    
    private IEnumerator ReconnectCoroutine()
    {
        isReconnecting = true;
        Debug.Log($"Reconnecting in {reconnectDelay} seconds...");
        yield return new WaitForSeconds(reconnectDelay);
        Connect();
        isReconnecting = false;
    }
}
