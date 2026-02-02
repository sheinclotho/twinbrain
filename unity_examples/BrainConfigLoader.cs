using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

/// <summary>
/// TwinBrain Unity Configuration Loader
/// 
/// Automatically loads and applies configuration from unity_config.json.
/// This enhances the BrainVisualization script with auto-configuration.
/// 
/// Usage:
/// 1. Attach this script to your BrainVisualization GameObject
/// 2. Point it to the unity_config.json file
/// 3. It will automatically configure the BrainVisualization component
/// </summary>
[RequireComponent(typeof(BrainVisualization))]
public class BrainConfigLoader : MonoBehaviour
{
    [Header("Configuration File")]
    [Tooltip("Path to unity_config.json")]
    public string configPath = "output/unity/unity_config.json";
    
    [Tooltip("Auto-load configuration on start")]
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
    /// Load and apply configuration from JSON file
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
    /// Parse color from JSON
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

/// <summary>
/// Data class for Unity configuration
/// </summary>
[System.Serializable]
public class UnityConfig
{
    public string project_name;
    public string atlas;
    public DataPaths data_paths;
    public VisualizationSettings visualization;
    public ColorSettings colors;
    public AnimationSettings animation;
}

[System.Serializable]
public class DataPaths
{
    public string json_dir;
    public string obj_dir;
    public string materials_dir;
}

[System.Serializable]
public class VisualizationSettings
{
    public float region_scale;
    public float activity_threshold;
    public float connection_threshold;
    public bool show_connections;
    public int fps;
    public bool auto_play;
}

[System.Serializable]
public class ColorSettings
{
    public ColorRGB low_activity;
    public ColorRGB high_activity;
    public ColorRGBA connection_structural;
    public ColorRGBA connection_functional;
}

[System.Serializable]
public class ColorRGB
{
    public int r;
    public int g;
    public int b;
    
    public Color ToUnityColor()
    {
        return new Color(r / 255f, g / 255f, b / 255f);
    }
}

[System.Serializable]
public class ColorRGBA
{
    public int r;
    public int g;
    public int b;
    public int a;
    
    public Color ToUnityColor()
    {
        return new Color(r / 255f, g / 255f, b / 255f, a / 255f);
    }
}

[System.Serializable]
public class AnimationSettings
{
    public int start_frame;
    public int end_frame;
    public int frame_step;
}
