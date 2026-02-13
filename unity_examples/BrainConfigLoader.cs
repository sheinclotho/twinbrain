using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace TwinBrain
{
    /// <summary>
    /// TwinBrain Unity配置加载器
    /// Unity 2019+ 兼容版本
    /// 
    /// 自动加载并应用unity_config.json配置。
    /// 增强BrainVisualization脚本的自动配置功能。
    /// 
    /// 使用方法:
    /// 1. 将此脚本附加到BrainVisualization GameObject
    /// 2. 指向unity_config.json文件
    /// 3. 它将自动配置BrainVisualization组件
    /// </summary>
    [RequireComponent(typeof(BrainVisualization))]
    public class BrainConfigLoader : MonoBehaviour
    {
        [Header("Configuration File")]
        [Tooltip("unity_config.json的路径")]
        public string configPath = "StreamingAssets/unity_config.json";
        
        [Tooltip("启动时自动加载配置")]
        public bool autoLoad = true;
        
        private BrainVisualization visualization;
        
        void Start()
        {
            visualization = GetComponent<BrainVisualization>();
            
            if (autoLoad)
            {
                string fullPath = GetFullPath(configPath);
                if (File.Exists(fullPath))
                {
                    LoadConfiguration();
                }
                else
                {
                    Debug.LogWarning(string.Format("Configuration file not found: {0}. Using default settings.", fullPath));
                }
            }
        }
        
        /// <summary>
        /// 获取完整路径
        /// </summary>
        string GetFullPath(string path)
        {
            if (Path.IsPathRooted(path))
            {
                return path;
            }
            
            if (path.StartsWith("StreamingAssets/") || path.StartsWith("StreamingAssets\\"))
            {
                return Path.Combine(Application.streamingAssetsPath, path.Substring(16));
            }
            
            return Path.Combine(Application.dataPath, path);
        }
        
        /// <summary>
        /// 从JSON文件加载并应用配置
        /// </summary>
        public void LoadConfiguration()
        {
            try
            {
                string fullPath = GetFullPath(configPath);
                string jsonContent = File.ReadAllText(fullPath);
                JObject config = JObject.Parse(jsonContent);
                
                Debug.Log(string.Format("Loading configuration from: {0}", fullPath));
                
                // Apply data paths
                JToken dataPathsToken = config["data_paths"];
                if (dataPathsToken != null)
                {
                    JToken jsonDirToken = dataPathsToken["json_dir"];
                    if (jsonDirToken != null)
                    {
                        string jsonDir = jsonDirToken.ToString();
                        if (!string.IsNullOrEmpty(jsonDir))
                        {
                            string baseDir = Path.GetDirectoryName(fullPath);
                            visualization.jsonPath = Path.Combine(baseDir, jsonDir);
                            Debug.Log(string.Format("JSON path set to: {0}", visualization.jsonPath));
                        }
                    }
                    
                    JToken objDirToken = dataPathsToken["obj_dir"];
                    if (objDirToken != null)
                    {
                        string objDir = objDirToken.ToString();
                        if (!string.IsNullOrEmpty(objDir))
                        {
                            string baseDir = Path.GetDirectoryName(fullPath);
                            visualization.objDirectory = Path.Combine(baseDir, objDir);
                            Debug.Log(string.Format("OBJ directory set to: {0}", visualization.objDirectory));
                        }
                    }
                }
                
                // Apply visualization settings
                JToken visToken = config["visualization"];
                if (visToken != null)
                {
                    JToken regionScaleToken = visToken["region_scale"];
                    if (regionScaleToken != null)
                    {
                        visualization.regionScale = (float)regionScaleToken;
                    }
                    
                    JToken activityThresholdToken = visToken["activity_threshold"];
                    if (activityThresholdToken != null)
                    {
                        visualization.activityThreshold = (float)activityThresholdToken;
                    }
                    
                    JToken connectionThresholdToken = visToken["connection_threshold"];
                    if (connectionThresholdToken != null)
                    {
                        visualization.connectionThreshold = (float)connectionThresholdToken;
                    }
                    
                    JToken showConnectionsToken = visToken["show_connections"];
                    if (showConnectionsToken != null)
                    {
                        visualization.showConnections = (bool)showConnectionsToken;
                    }
                    
                    JToken fpsToken = visToken["fps"];
                    if (fpsToken != null)
                    {
                        visualization.fps = (float)fpsToken;
                    }
                    
                    JToken autoPlayToken = visToken["auto_play"];
                    if (autoPlayToken != null)
                    {
                        visualization.autoPlay = (bool)autoPlayToken;
                    }
                    
                    JToken useObjModelsToken = visToken["use_obj_models"];
                    if (useObjModelsToken != null)
                    {
                        visualization.useObjModels = (bool)useObjModelsToken;
                    }
                    
                    Debug.Log("Visualization settings applied");
                }
                
                // Apply color settings
                JToken colorsToken = config["colors"];
                if (colorsToken != null)
                {
                    JToken lowActivityToken = colorsToken["low_activity"];
                    if (lowActivityToken != null)
                    {
                        visualization.lowActivityColor = ParseColor(lowActivityToken);
                    }
                    
                    JToken highActivityToken = colorsToken["high_activity"];
                    if (highActivityToken != null)
                    {
                        visualization.highActivityColor = ParseColor(highActivityToken);
                    }
                    
                    JToken predictedSignalToken = colorsToken["predicted_signal"];
                    if (predictedSignalToken != null)
                    {
                        visualization.predictedColor = ParseColor(predictedSignalToken);
                    }
                    
                    Debug.Log("Color settings applied");
                }
                
                // Apply animation settings
                JToken animToken = config["animation"];
                if (animToken != null)
                {
                    JToken startFrameToken = animToken["start_frame"];
                    if (startFrameToken != null)
                    {
                        Debug.Log(string.Format("Animation start frame: {0}", startFrameToken));
                    }
                    
                    JToken endFrameToken = animToken["end_frame"];
                    if (endFrameToken != null)
                    {
                        Debug.Log(string.Format("Animation end frame: {0}", endFrameToken));
                    }
                }
                
                Debug.Log("✓ Configuration loaded successfully!");
                
            }
            catch (System.Exception e)
            {
                Debug.LogError(string.Format("Failed to load configuration: {0}", e.Message));
            }
        }
        
        /// <summary>
        /// 从JSON解析颜色
        /// </summary>
        private Color ParseColor(JToken colorData)
        {
            JToken rToken = colorData["r"];
            JToken gToken = colorData["g"];
            JToken bToken = colorData["b"];
            JToken aToken = colorData["a"];
            
            float r = (rToken != null) ? (float)rToken / 255f : 0f;
            float g = (gToken != null) ? (float)gToken / 255f : 0f;
            float b = (bToken != null) ? (float)bToken / 255f : 0f;
            float a = (aToken != null) ? (float)aToken / 255f : 1f;
            
            return new Color(r, g, b, a);
        }
    }
}
