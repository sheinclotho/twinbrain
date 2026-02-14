using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace TwinBrain
{
    /// <summary>
    /// Cache文件到JSON转换器
    /// 
    /// 提供UI按钮，点击后自动调用后端处理cache文件夹中的数据，
    /// 转换为JSON格式供Unity可视化使用。
    /// 
    /// 使用方法:
    /// 1. 确保后端服务器已启动: python unity_startup.py --model results/model.pt
    /// 2. 将此脚本附加到UI Canvas或GameObject
    /// 3. 配置UI按钮和文本组件
    /// 4. 点击"转换Cache到JSON"按钮即可自动处理
    /// 
    /// 工作流程:
    /// - 用户将cache文件(.pt PyTorch格式)放入 unity_project/brain_data/cache/
    /// - 点击按钮
    /// - 脚本调用后端 brain_state_exporter 接口
    /// - 后端自动读取cache文件，生成JSON到 model_output/
    /// - Unity可以立即加载新生成的JSON进行可视化
    /// </summary>
    public class CacheToJsonConverter : MonoBehaviour
    {
        [Header("UI组件")]
        [Tooltip("转换按钮")]
        public Button convertButton;
        
        [Tooltip("状态文本显示")]
        public Text statusText;
        
        [Tooltip("进度条（可选）")]
        public Slider progressSlider;
        
        [Header("路径设置")]
        [Tooltip("Cache文件目录（相对于StreamingAssets）")]
        public string cacheDirectory = "brain_data/cache";
        
        [Tooltip("JSON输出目录（相对于StreamingAssets）")]
        public string outputDirectory = "brain_data/model_output";
        
        [Header("后端连接")]
        [Tooltip("后端服务器URL")]
        public string backendUrl = "http://localhost:8765";
        
        [Tooltip("转换超时时间（秒）")]
        public float timeout = 300f;
        
        // 私有变量
        private bool isConverting = false;
        private WebSocketClient wsClient;
        
        void Start()
        {
            // 查找WebSocketClient组件（尝试两种类型）
            wsClient = FindObjectOfType<WebSocketClient>();
            
            // 如果没找到标准版，尝试查找改进版
            if (wsClient == null)
            {
                var wsClientImproved = FindObjectOfType<WebSocketClientImproved>();
                if (wsClientImproved != null)
                {
                    // 创建适配器以兼容接口
                    Debug.Log("CacheToJsonConverter: Using WebSocketClientImproved");
                }
                else
                {
                    Debug.LogWarning("CacheToJsonConverter: WebSocketClient not found. Will use HTTP API instead.");
                }
            }
            
            // 设置按钮点击事件
            if (convertButton != null)
            {
                convertButton.onClick.AddListener(OnConvertButtonClicked);
            }
            else
            {
                Debug.LogError("CacheToJsonConverter: Convert button not assigned!");
            }
            
            UpdateStatus("就绪。点击按钮转换cache文件。", Color.white);
        }
        
        /// <summary>
        /// 按钮点击处理
        /// </summary>
        public void OnConvertButtonClicked()
        {
            if (isConverting)
            {
                UpdateStatus("转换正在进行中，请稍候...", Color.yellow);
                return;
            }
            
            StartCoroutine(ConvertCacheToJson());
        }
        
        /// <summary>
        /// 执行Cache到JSON转换
        /// </summary>
        private IEnumerator ConvertCacheToJson()
        {
            isConverting = true;
            
            if (convertButton != null)
            {
                convertButton.interactable = false;
            }
            
            UpdateStatus("正在扫描cache文件...", Color.cyan);
            UpdateProgress(0.1f);
            
            // 检查cache目录
            string cachePath = GetFullPath(cacheDirectory);
            if (!Directory.Exists(cachePath))
            {
                UpdateStatus("错误: Cache目录不存在: " + cachePath, Color.red);
                yield return new WaitForSeconds(2f);
                FinishConversion();
                yield break;
            }
            
            // 查找cache文件（仅.pt/.pth格式，这是实际的PyTorch缓存格式）
            string[] cacheFiles = Directory.GetFiles(cachePath, "*.*");
            List<string> validCacheFiles = new List<string>();
            
            foreach (string file in cacheFiles)
            {
                string ext = Path.GetExtension(file).ToLower();
                if (ext == ".pt" || ext == ".pth")
                {
                    validCacheFiles.Add(file);
                }
            }
            
            if (validCacheFiles.Count == 0)
            {
                UpdateStatus("警告: 未找到cache文件 (.pt, .pth)", Color.yellow);
                yield return new WaitForSeconds(2f);
                FinishConversion();
                yield break;
            }
            
            UpdateStatus(string.Format("找到 {0} 个cache文件，正在转换...", validCacheFiles.Count), Color.cyan);
            UpdateProgress(0.3f);
            
            // 调用后端API进行转换
            bool success = false;
            
            // 目前统一使用HTTP API进行转换
            // WebSocket版本适用于真正的双向通信场景
            yield return StartCoroutine(ConvertViaHttp(cachePath, result => success = result));
            
            if (success)
            {
                UpdateStatus("转换完成！JSON文件已生成。", Color.green);
                UpdateProgress(1.0f);
                
                // 可以触发事件通知其他组件重新加载数据
                BroadcastMessage("OnJsonFilesUpdated", SendMessageOptions.DontRequireReceiver);
            }
            else
            {
                UpdateStatus("转换失败。请检查后端服务器。", Color.red);
                UpdateProgress(0f);
            }
            
            yield return new WaitForSeconds(2f);
            FinishConversion();
        }
        
        /// <summary>
        /// 通过WebSocket调用后端转换
        /// </summary>
        private IEnumerator ConvertViaWebSocket(string cachePath, System.Action<bool> callback)
        {
            UpdateStatus("通过WebSocket请求转换...", Color.cyan);
            
            // 构建请求
            var request = new Dictionary<string, object>
            {
                {"type", "convert_cache"},
                {"cache_dir", cachePath},
                {"output_dir", GetFullPath(outputDirectory)}
            };
            
            string requestJson = JsonConvert.SerializeObject(request);
            
            // 发送请求（需要WebSocketClient支持异步响应）
            bool responseReceived = false;
            string responseData = null;
            
            // 注册回调（如果WebSocketClient支持）
            // 这里简化处理，实际需要WebSocketClient提供回调机制
            
            yield return new WaitForSeconds(1f);
            
            // 轮询检查输出目录是否有新文件生成
            float elapsed = 0f;
            int initialFileCount = GetJsonFileCount();
            
            while (elapsed < timeout)
            {
                yield return new WaitForSeconds(1f);
                elapsed += 1f;
                
                int currentFileCount = GetJsonFileCount();
                if (currentFileCount > initialFileCount)
                {
                    UpdateProgress(0.7f + (elapsed / timeout) * 0.3f);
                    yield return new WaitForSeconds(2f); // 等待所有文件写入完成
                    callback(true);
                    yield break;
                }
                
                UpdateProgress(0.3f + (elapsed / timeout) * 0.4f);
            }
            
            // 超时
            UpdateStatus("转换超时。请检查后端日志。", Color.red);
            callback(false);
            yield break;
        }
        
        /// <summary>
        /// 通过HTTP API调用后端转换
        /// </summary>
        private IEnumerator ConvertViaHttp(string cachePath, System.Action<bool> callback)
        {
            UpdateStatus("通过HTTP请求转换...", Color.cyan);
            
            // 构建HTTP请求
            string url = backendUrl + "/api/convert_cache";
            
            var requestData = new Dictionary<string, object>
            {
                {"cache_dir", cachePath},
                {"output_dir", GetFullPath(outputDirectory)}
            };
            
            string jsonData = JsonConvert.SerializeObject(requestData);
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
            
            // Unity 2019+ 使用UnityWebRequest
            using (UnityEngine.Networking.UnityWebRequest www = new UnityEngine.Networking.UnityWebRequest(url, "POST"))
            {
                www.uploadHandler = new UnityEngine.Networking.UploadHandlerRaw(bodyRaw);
                www.downloadHandler = new UnityEngine.Networking.DownloadHandlerBuffer();
                www.SetRequestHeader("Content-Type", "application/json");
                www.timeout = (int)timeout;
                
                yield return www.SendWebRequest();
                
                #if UNITY_2020_1_OR_NEWER
                if (www.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
                #else
                if (!www.isNetworkError && !www.isHttpError)
                #endif
                {
                    UpdateProgress(0.9f);
                    
                    try
                    {
                        string response = www.downloadHandler.text;
                        var responseObj = JsonConvert.DeserializeObject<Dictionary<string, object>>(response);
                        
                        if (responseObj.ContainsKey("success") && (bool)responseObj["success"])
                        {
                            callback(true);
                            yield break;
                        }
                        else
                        {
                            string error = responseObj.ContainsKey("error") ? responseObj["error"].ToString() : "Unknown error";
                            UpdateStatus("后端错误: " + error, Color.red);
                            callback(false);
                            yield break;
                        }
                    }
                    catch (System.Exception e)
                    {
                        UpdateStatus("解析响应失败: " + e.Message, Color.red);
                        Debug.LogError("Response parsing error: " + e.ToString());
                        callback(false);
                        yield break;
                    }
                }
                else
                {
                    UpdateStatus("HTTP请求失败: " + www.error, Color.red);
                    Debug.LogError("HTTP error: " + www.error);
                    callback(false);
                    yield break;
                }
            }
        }
        
        /// <summary>
        /// 获取完整路径
        /// </summary>
        private string GetFullPath(string relativePath)
        {
            // 如果是绝对路径，直接返回
            if (Path.IsPathRooted(relativePath))
            {
                return relativePath;
            }
            
            // 相对于StreamingAssets
            string streamingPath = Application.streamingAssetsPath;
            return Path.Combine(streamingPath, relativePath);
        }
        
        /// <summary>
        /// 获取输出目录中的JSON文件数量
        /// </summary>
        private int GetJsonFileCount()
        {
            string outputPath = GetFullPath(outputDirectory);
            if (!Directory.Exists(outputPath))
            {
                return 0;
            }
            
            string[] jsonFiles = Directory.GetFiles(outputPath, "*.json");
            return jsonFiles.Length;
        }
        
        /// <summary>
        /// 更新状态文本
        /// </summary>
        private void UpdateStatus(string message, Color color)
        {
            Debug.Log("CacheToJsonConverter: " + message);
            
            if (statusText != null)
            {
                statusText.text = message;
                statusText.color = color;
            }
        }
        
        /// <summary>
        /// 更新进度条
        /// </summary>
        private void UpdateProgress(float progress)
        {
            if (progressSlider != null)
            {
                progressSlider.value = progress;
            }
        }
        
        /// <summary>
        /// 完成转换，恢复UI状态
        /// </summary>
        private void FinishConversion()
        {
            isConverting = false;
            
            if (convertButton != null)
            {
                convertButton.interactable = true;
            }
        }
        
        /// <summary>
        /// 清理
        /// </summary>
        void OnDestroy()
        {
            if (convertButton != null)
            {
                convertButton.onClick.RemoveListener(OnConvertButtonClicked);
            }
        }
    }
}
