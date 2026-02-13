using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using TwinBrain;

/// <summary>
/// TwinBrain Unity配置加载器
/// 
/// 自动加载并应用unity_config.json配置。
/// 增强BrainVisualization脚本的自动配置功能。
/// 
/// Unity版本兼容性:
/// - Unity 2017.1+ (C# 6.0支持)
/// - 对于更早版本的Unity, 请将 ?. 操作符替换为传统null检查
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
    public string configPath = "output/unity/unity_config.json";
    
    [Tooltip("启动时自动加载配置")]
    public bool autoLoad = true;
    
    private BrainVisualization visualization;
    
    void Start()
    {
        visualization = GetComponent<BrainVisualization>();
        
        if (autoLoad)
        {
            if (File.Exists(configPath))
            {
                LoadConfiguration();
            }
            else
            {
                Debug.LogWarning($"Configuration file not found: {configPath}. Using default settings.");
            }
        }
    }
    
    /// <summary>
    /// 从JSON文件加载并应用配置
    /// </summary>
    public void LoadConfiguration()
    {
        try
        {
            string jsonContent = File.ReadAllText(configPath);
            JObject config = JObject.Parse(jsonContent);
            
            Debug.Log($"Loading configuration from: {configPath}");
            
            // Apply data paths
            if (config["data_paths"] != null)
            {
                // For Unity pre-2017: JToken jsonDir = config["data_paths"]["json_dir"]; 
                //                     if (jsonDir != null) { string jsonDirStr = jsonDir.ToString(); ... }
                string jsonDir = config["data_paths"]["json_dir"]?.ToString();
                if (!string.IsNullOrEmpty(jsonDir))
                {
                    string baseDir = Path.GetDirectoryName(configPath);
                    visualization.jsonPath = Path.Combine(baseDir, jsonDir);
                    Debug.Log($"JSON path set to: {visualization.jsonPath}");
                }
            }
            
            // Apply visualization settings
            if (config["visualization"] != null)
            {
                var vis = config["visualization"];
                
                if (vis["region_scale"] != null)
                    visualization.regionScale = (float)vis["region_scale"];
                
                if (vis["activity_threshold"] != null)
                    visualization.activityThreshold = (float)vis["activity_threshold"];
                
                if (vis["connection_threshold"] != null)
                    visualization.connectionThreshold = (float)vis["connection_threshold"];
                
                if (vis["show_connections"] != null)
                    visualization.showConnections = (bool)vis["show_connections"];
                
                if (vis["fps"] != null)
                    visualization.fps = (float)vis["fps"];
                
                if (vis["auto_play"] != null)
                    visualization.autoPlay = (bool)vis["auto_play"];
                
                Debug.Log("Visualization settings applied");
            }
            
            // Apply color settings
            if (config["colors"] != null)
            {
                var colors = config["colors"];
                
                if (colors["low_activity"] != null)
                {
                    visualization.lowActivityColor = ParseColor(colors["low_activity"]);
                }
                
                if (colors["high_activity"] != null)
                {
                    visualization.highActivityColor = ParseColor(colors["high_activity"]);
                }
                
                Debug.Log("Color settings applied");
            }
            
            // Apply animation settings
            if (config["animation"] != null)
            {
                var anim = config["animation"];
                
                // These are informational but could be used to validate loaded data
                if (anim["start_frame"] != null)
                    Debug.Log($"Animation start frame: {anim["start_frame"]}");
                
                if (anim["end_frame"] != null)
                    Debug.Log($"Animation end frame: {anim["end_frame"]}");
            }
            
            Debug.Log("✓ Configuration loaded successfully!");
            
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to load configuration: {e.Message}");
        }
    }
    
    /// <summary>
    /// 从JSON解析颜色
    /// </summary>
    private Color ParseColor(JToken colorData)
    {
        float r = colorData["r"] != null ? (float)colorData["r"] / 255f : 0;
        float g = colorData["g"] != null ? (float)colorData["g"] / 255f : 0;
        float b = colorData["b"] != null ? (float)colorData["b"] / 255f : 0;
        float a = colorData["a"] != null ? (float)colorData["a"] / 255f : 1;
        
        return new Color(r, g, b, a);
    }
}
