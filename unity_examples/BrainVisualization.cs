using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using TwinBrain;

/// <summary>
/// TwinBrain Unity可视化组件
/// 
/// 此脚本从TwinBrain加载并可视化大脑状态JSON文件。
/// 将此脚本附加到Unity场景中的GameObject上。
/// 
/// 依赖项:
/// - Newtonsoft.Json（通过Package Manager安装）
/// - 带有Renderer组件的脑区预制体
/// - 连接线预制体
/// - BrainDataStructures.cs（数据结构定义）
/// </summary>
public class BrainVisualization : MonoBehaviour
{
    [Header("File Settings")]
    [Tooltip("Path to the JSON file or directory")]
    public string jsonPath = "brain_state.json";
    
    [Tooltip("For sequences: load all JSON files in directory")]
    public bool loadSequence = false;
    
    [Header("Visualization Settings")]
    [Tooltip("Prefab for brain regions (e.g., sphere)")]
    public GameObject regionPrefab;
    
    [Tooltip("Material for connections")]
    public Material connectionMaterial;
    
    [Tooltip("Scale factor for region size")]
    public float regionScale = 1.0f;
    
    [Tooltip("Minimum activity threshold to display")]
    [Range(0f, 1f)]
    public float activityThreshold = 0.3f;
    
    [Tooltip("Show connections")]
    public bool showConnections = true;
    
    [Tooltip("Connection strength threshold")]
    [Range(0f, 1f)]
    public float connectionThreshold = 0.5f;
    
    [Header("Animation Settings")]
    [Tooltip("For sequences: frame rate")]
    public float fps = 10f;
    
    [Tooltip("Auto-play sequence")]
    public bool autoPlay = true;
    
    [Header("Colors")]
    [Tooltip("Color for low activity")]
    public Color lowActivityColor = Color.blue;
    
    [Tooltip("Color for high activity")]
    public Color highActivityColor = Color.red;
    
    // Private variables
    private BrainStateData currentState;
    private Dictionary<int, GameObject> regionObjects = new Dictionary<int, GameObject>();
    private List<LineRenderer> connectionLines = new List<LineRenderer>();
    private List<string> sequenceFiles;
    private int currentFrame = 0;
    private bool isPlaying = false;
    
    void Start()
    {
        if (loadSequence)
        {
            LoadSequence();
            if (autoPlay)
            {
                Play();
            }
        }
        else
        {
            LoadSingleState(jsonPath);
        }
    }
    
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (isPlaying)
                Pause();
            else
                Play();
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            Reload();
        }
    }
    
    /// <summary>
    /// 从JSON文件加载单个大脑状态
    /// </summary>
    public void LoadSingleState(string path)
    {
        try
        {
            string jsonContent = File.ReadAllText(path);
            currentState = JsonConvert.DeserializeObject<BrainStateData>(jsonContent);
            
            Debug.Log($"Loaded brain state: {currentState.metadata.subject} at time {currentState.brain_state.time_second}s");
            
            UpdateVisualization();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to load brain state: {e.Message}");
        }
    }
    
    /// <summary>
    /// 加载大脑状态序列
    /// </summary>
    public void LoadSequence()
    {
        try
        {
            // Load index file
            string indexPath = Path.Combine(jsonPath, "sequence_index.json");
            if (File.Exists(indexPath))
            {
                string indexContent = File.ReadAllText(indexPath);
                var index = JsonConvert.DeserializeObject<SequenceIndex>(indexContent);
                
                sequenceFiles = new List<string>();
                foreach (string file in index.files)
                {
                    sequenceFiles.Add(Path.Combine(jsonPath, file));
                }
                
                Debug.Log($"Loaded sequence with {sequenceFiles.Count} frames");
                
                // Load first frame
                if (sequenceFiles.Count > 0)
                {
                    LoadSingleState(sequenceFiles[0]);
                }
            }
            else
            {
                Debug.LogError($"Sequence index not found: {indexPath}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to load sequence: {e.Message}");
        }
    }
    
    /// <summary>
    /// 使用当前状态更新可视化
    /// </summary>
    void UpdateVisualization()
    {
        if (currentState == null) return;
        
        // Clear existing visualization
        ClearVisualization();
        
        // Create regions
        foreach (var region in currentState.brain_state.regions)
        {
            CreateRegion(region);
        }
        
        // Create connections
        if (showConnections)
        {
            foreach (var conn in currentState.brain_state.connections)
            {
                if (conn.strength >= connectionThreshold)
                {
                    CreateConnection(conn);
                }
            }
        }
    }
    
    /// <summary>
    /// 创建脑区可视化
    /// </summary>
    void CreateRegion(RegionData region)
    {
        // Check activity threshold
        float activity = region.activity.fmri?.amplitude ?? 0f;
        if (activity < activityThreshold) return;
        
        // Instantiate region object
        GameObject regionObj;
        if (regionPrefab != null)
        {
            regionObj = Instantiate(regionPrefab, transform);
        }
        else
        {
            regionObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            regionObj.transform.SetParent(transform);
        }
        
        // Set position (convert from brain coordinates)
        Vector3 position = new Vector3(
            region.position.x / 100f,  // Scale down
            region.position.z / 100f,  // Z becomes Y in Unity
            region.position.y / 100f   // Y becomes Z in Unity
        );
        regionObj.transform.localPosition = position;
        
        // Set scale based on activity
        float scale = regionScale * (0.5f + activity * 0.5f);
        regionObj.transform.localScale = Vector3.one * scale;
        
        // Set color based on activity
        Color color = Color.Lerp(lowActivityColor, highActivityColor, activity);
        Renderer renderer = regionObj.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.color = color;
        }
        
        // Set name
        regionObj.name = $"Region_{region.id}_{region.label}";
        
        // Store reference
        regionObjects[region.id] = regionObj;
    }
    
    /// <summary>
    /// 创建连接可视化
    /// </summary>
    void CreateConnection(ConnectionData conn)
    {
        if (!regionObjects.ContainsKey(conn.source) || 
            !regionObjects.ContainsKey(conn.target))
        {
            return;
        }
        
        GameObject lineObj = new GameObject($"Connection_{conn.source}_{conn.target}");
        lineObj.transform.SetParent(transform);
        
        LineRenderer line = lineObj.AddComponent<LineRenderer>();
        line.material = connectionMaterial ?? new Material(Shader.Find("Sprites/Default"));
        
        // Set positions
        Vector3 startPos = regionObjects[conn.source].transform.position;
        Vector3 endPos = regionObjects[conn.target].transform.position;
        
        line.SetPosition(0, startPos);
        line.SetPosition(1, endPos);
        
        // Set width based on strength
        float width = 0.01f * conn.strength;
        line.startWidth = width;
        line.endWidth = width;
        
        // Set color (different for structural vs functional)
        Color lineColor = conn.type == "structural" ? Color.white : Color.yellow;
        lineColor.a = conn.strength;
        line.startColor = lineColor;
        line.endColor = lineColor;
        
        connectionLines.Add(line);
    }
    
    /// <summary>
    /// 清除所有可视化
    /// </summary>
    void ClearVisualization()
    {
        // Destroy region objects
        foreach (var obj in regionObjects.Values)
        {
            Destroy(obj);
        }
        regionObjects.Clear();
        
        // Destroy connection lines
        foreach (var line in connectionLines)
        {
            Destroy(line.gameObject);
        }
        connectionLines.Clear();
    }
    
    /// <summary>
    /// 播放序列动画
    /// </summary>
    public void Play()
    {
        if (sequenceFiles == null || sequenceFiles.Count == 0) return;
        
        isPlaying = true;
        StartCoroutine(PlaySequence());
    }
    
    /// <summary>
    /// 暂停序列动画
    /// </summary>
    public void Pause()
    {
        isPlaying = false;
    }
    
    /// <summary>
    /// 重新加载当前状态
    /// </summary>
    public void Reload()
    {
        if (loadSequence)
        {
            LoadSequence();
        }
        else
        {
            LoadSingleState(jsonPath);
        }
    }
    
    /// <summary>
    /// 播放序列的协程
    /// </summary>
    IEnumerator PlaySequence()
    {
        while (isPlaying && currentFrame < sequenceFiles.Count)
        {
            LoadSingleState(sequenceFiles[currentFrame]);
            currentFrame++;
            
            if (currentFrame >= sequenceFiles.Count)
            {
                currentFrame = 0; // Loop
            }
            
            yield return new WaitForSeconds(1f / fps);
        }
        
        isPlaying = false;
    }
}
