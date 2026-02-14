using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace TwinBrain
{
    /// <summary>
    /// TwinBrain Unity可视化组件
    /// Unity 2019+ 兼容版本
    /// 
    /// 此脚本从TwinBrain加载并可视化大脑状态JSON文件。
    /// 支持单个OBJ模型和多个独立脑区OBJ模型两种模式。
    /// 
    /// 功能特性:
    /// - 支持从.lh文件加载的多脑区OBJ模型
    /// - 每个脑区对应后端模型的预测数值
    /// - 颜色映射显示真实/预测信号
    /// - 点击交互选择脑区
    /// - 虚拟刺激输入
    /// 
    /// OBJ模型加载说明:
    /// - 运行时OBJ加载需要第三方插件（如TriLib, Runtime OBJ Importer）
    /// - 或在Unity Editor中预先导入OBJ文件作为资产
    /// - OBJ文件命名格式: region_XXXX.obj（XXXX为零填充的区域ID）
    /// 
    /// 依赖项:
    /// - Newtonsoft.Json (通过Package Manager安装)
    /// - BrainDataStructures.cs
    /// </summary>
    public class BrainVisualization : MonoBehaviour
    {
        [Header("File Settings")]
        [Tooltip("Path to the JSON file or directory")]
        public string jsonPath = "brain_state.json";
        
        [Tooltip("For sequences: load all JSON files in directory")]
        public bool loadSequence = false;
        
        [Header("Model Settings")]
        [Tooltip("Use individual OBJ models for each brain region")]
        public bool useObjModels = true;
        
        [Tooltip("Directory containing region OBJ files (region_0001.obj, etc)")]
        public string objDirectory = "StreamingAssets/OBJ";
        
        [Tooltip("Prefab for brain regions when not using OBJ models")]
        public GameObject regionPrefab;
        
        [Header("Visualization Settings")]
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
        [Tooltip("Color for low activity values")]
        public Color lowActivityColor = Color.blue;
        
        [Tooltip("Color for high activity values")]
        public Color highActivityColor = Color.red;
        
        [Header("Interaction")]
        [Tooltip("Enable click interaction")]
        public bool enableInteraction = true;
        
        // Private variables
        private BrainStateData currentState;
        private Dictionary<int, GameObject> regionObjects = new Dictionary<int, GameObject>();
        private Dictionary<int, Renderer> regionRenderers = new Dictionary<int, Renderer>();
        private List<LineRenderer> connectionLines = new List<LineRenderer>();
        private List<string> sequenceFiles;
        private List<BrainStateData> loadedSequence = new List<BrainStateData>();
        private int currentFrame = 0;
        private bool isPlaying = false;
        private int selectedRegionId = -1;
        
        // Normalization values for color mapping across entire sequence
        private float globalMinActivity = 0f;
        private float globalMaxActivity = 1f;
        
        // Events
        public delegate void RegionClickedHandler(int regionId, RegionData regionData);
        public event RegionClickedHandler OnRegionClicked;
        
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
            // Keyboard controls
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
            
            // Mouse click interaction
            if (enableInteraction && Input.GetMouseButtonDown(0))
            {
                HandleMouseClick();
            }
        }
        
        /// <summary>
        /// 处理鼠标点击事件
        /// </summary>
        void HandleMouseClick()
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit))
            {
                GameObject hitObject = hit.collider.gameObject;
                
                // Find region ID from object
                foreach (var kvp in regionObjects)
                {
                    if (kvp.Value == hitObject)
                    {
                        OnRegionClick(kvp.Key);
                        break;
                    }
                }
            }
        }
        
        /// <summary>
        /// 处理脑区点击
        /// </summary>
        void OnRegionClick(int regionId)
        {
            selectedRegionId = regionId;
            
            if (currentState != null)
            {
                RegionData regionData = currentState.brain_state.regions.Find(r => r.id == regionId);
                if (regionData != null)
                {
                    Debug.Log(string.Format("Clicked Region {0}: {1}, Activity: {2:F3}", 
                        regionId, regionData.label, GetRegionActivity(regionData)));
                    
                    if (OnRegionClicked != null)
                    {
                        OnRegionClicked(regionId, regionData);
                    }
                    
                    // Highlight selected region
                    HighlightRegion(regionId);
                }
            }
        }
        
        /// <summary>
        /// 高亮选中的脑区
        /// </summary>
        void HighlightRegion(int regionId)
        {
            // Reset all regions
            foreach (var kvp in regionRenderers)
            {
                if (kvp.Value != null)
                {
                    kvp.Value.material.SetFloat("_Emission", 0f);
                }
            }
            
            // Highlight selected region
            if (regionRenderers.ContainsKey(regionId) && regionRenderers[regionId] != null)
            {
                regionRenderers[regionId].material.SetFloat("_Emission", 0.5f);
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
                
                Debug.Log(string.Format("Loaded brain state: {0} at time {1:F2}s", 
                    currentState.metadata.subject, currentState.brain_state.time_second));
                
                UpdateVisualization();
            }
            catch (System.Exception e)
            {
                Debug.LogError(string.Format("Failed to load brain state: {0}", e.Message));
            }
        }
        
        /// <summary>
        /// 加载大脑状态序列
        /// </summary>
        public void LoadSequence()
        {
            try
            {
                string indexPath = Path.Combine(jsonPath, "sequence_index.json");
                if (File.Exists(indexPath))
                {
                    string indexContent = File.ReadAllText(indexPath);
                    SequenceIndex index = JsonConvert.DeserializeObject<SequenceIndex>(indexContent);
                    
                    sequenceFiles = new List<string>();
                    loadedSequence = new List<BrainStateData>();
                    
                    foreach (string file in index.files)
                    {
                        string filePath = Path.Combine(jsonPath, file);
                        
                        // Preload state for normalization
                        try
                        {
                            string jsonContent = File.ReadAllText(filePath);
                            BrainStateData state = JsonConvert.DeserializeObject<BrainStateData>(jsonContent);
                            loadedSequence.Add(state);
                            sequenceFiles.Add(filePath);  // Only add if successfully loaded
                        }
                        catch (System.Exception e)
                        {
                            Debug.LogWarning(string.Format("Failed to preload {0}: {1}", file, e.Message));
                            // Skip this file - don't add to either list
                        }
                    }
                    
                    Debug.Log(string.Format("Loaded sequence with {0} frames", loadedSequence.Count));
                    
                    // Compute global min/max for normalization
                    ComputeGlobalActivityRange();
                    
                    // Load first frame
                    if (loadedSequence.Count > 0)
                    {
                        currentState = loadedSequence[0];
                        UpdateVisualization();
                    }
                }
                else
                {
                    Debug.LogError(string.Format("Sequence index not found: {0}", indexPath));
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError(string.Format("Failed to load sequence: {0}", e.Message));
            }
        }
        
        /// <summary>
        /// 使用当前状态更新可视化
        /// </summary>
        void UpdateVisualization()
        {
            if (currentState == null) return;
            
            ClearVisualization();
            
            // Create regions
            foreach (RegionData region in currentState.brain_state.regions)
            {
                CreateRegion(region);
            }
            
            // Create connections
            if (showConnections && currentState.brain_state.connections != null)
            {
                foreach (ConnectionData conn in currentState.brain_state.connections)
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
            float activity = GetRegionActivity(region);
            if (activity < activityThreshold) return;
            
            GameObject regionObj = null;
            
            if (useObjModels)
            {
                // Load OBJ model for this region
                string objPath = Path.Combine(objDirectory, string.Format("region_{0:D4}.obj", region.id));
                regionObj = LoadObjModel(objPath);
            }
            
            // Fallback to prefab or primitive
            if (regionObj == null)
            {
                if (regionPrefab != null)
                {
                    regionObj = Instantiate(regionPrefab, transform);
                }
                else
                {
                    regionObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    regionObj.transform.SetParent(transform);
                    
                    // Add collider for interaction
                    if (enableInteraction)
                    {
                        Collider col = regionObj.GetComponent<Collider>();
                        if (col == null)
                        {
                            regionObj.AddComponent<SphereCollider>();
                        }
                    }
                }
            }
            
            // Set position (convert from brain coordinates)
            Vector3 position = new Vector3(
                region.position.x / 100f,
                region.position.z / 100f,
                region.position.y / 100f
            );
            regionObj.transform.localPosition = position;
            
            // Set scale based on activity
            float scale = regionScale * (0.5f + activity * 0.5f);
            regionObj.transform.localScale = Vector3.one * scale;
            
            // Set color based on activity and prediction status
            Color color = GetActivityColor(region);
            Renderer renderer = regionObj.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = color;
                regionRenderers[region.id] = renderer;
            }
            
            // Set name
            regionObj.name = string.Format("Region_{0}_{1}", region.id, region.label);
            
            // Store reference
            regionObjects[region.id] = regionObj;
        }
        
        /// <summary>
        /// 加载OBJ模型
        /// </summary>
        GameObject LoadObjModel(string objPath)
        {
            if (!File.Exists(objPath))
            {
                return null;
            }
            
            try
            {
                // Note: Unity doesn't have built-in OBJ loader at runtime
                // This requires either:
                // 1. Pre-import OBJ files as assets in Editor
                // 2. Use a runtime OBJ loader plugin (e.g., TriLib, Runtime OBJ Importer)
                // 3. Convert OBJ to Unity-compatible format
                
                Debug.LogWarning(string.Format("OBJ runtime loading not implemented. Use Editor import or runtime OBJ loader plugin for: {0}", objPath));
                return null;
            }
            catch (System.Exception e)
            {
                Debug.LogError(string.Format("Failed to load OBJ model: {0}", e.Message));
                return null;
            }
        }
        
        /// <summary>
        /// 获取脑区活动值
        /// </summary>
        float GetRegionActivity(RegionData region)
        {
            if (region.activity == null) return 0f;
            
            if (region.activity.isPredicted && region.activity.predictionValue > 0)
            {
                return region.activity.predictionValue;
            }
            
            if (region.activity.fmri != null)
            {
                return region.activity.fmri.amplitude;
            }
            
            if (region.activity.eeg != null)
            {
                return region.activity.eeg.amplitude;
            }
            
            return 0f;
        }
        
        /// <summary>
        /// 根据活动获取颜色（使用全局归一化）
        /// </summary>
        Color GetActivityColor(RegionData region)
        {
            float activity = GetRegionActivity(region);
            
            // Normalize activity using global min/max across entire sequence
            float normalizedActivity = 0.5f;
            if (globalMaxActivity > globalMinActivity)
            {
                normalizedActivity = (activity - globalMinActivity) / (globalMaxActivity - globalMinActivity);
                normalizedActivity = Mathf.Clamp01(normalizedActivity);
            }
            
            // Use single color scale for all data (real or predicted)
            return Color.Lerp(lowActivityColor, highActivityColor, normalizedActivity);
        }
        
        /// <summary>
        /// 计算整个序列的活动值范围（用于归一化）
        /// </summary>
        void ComputeGlobalActivityRange()
        {
            if (loadedSequence == null || loadedSequence.Count == 0)
            {
                globalMinActivity = 0f;
                globalMaxActivity = 1f;
                Debug.LogWarning("No sequence loaded, using default activity range [0, 1]");
                return;
            }
            
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            int validValueCount = 0;
            
            foreach (BrainStateData state in loadedSequence)
            {
                if (state == null || state.brain_state == null || state.brain_state.regions == null)
                {
                    continue;
                }
                
                foreach (RegionData region in state.brain_state.regions)
                {
                    float activity = GetRegionActivityRaw(region);
                    if (activity < minVal) minVal = activity;
                    if (activity > maxVal) maxVal = activity;
                    validValueCount++;
                }
            }
            
            // Handle edge case: no valid data or all values are the same
            if (validValueCount == 0 || minVal >= maxVal)
            {
                globalMinActivity = 0f;
                globalMaxActivity = 1f;
                Debug.LogWarning("No valid activity data or uniform values, using default range [0, 1]");
            }
            else
            {
                globalMinActivity = minVal;
                globalMaxActivity = maxVal;
                Debug.Log(string.Format("Global activity range: [{0:F3}, {1:F3}] from {2} values", 
                    globalMinActivity, globalMaxActivity, validValueCount));
            }
        }
        
        /// <summary>
        /// 获取脑区原始活动值（不归一化）
        /// </summary>
        float GetRegionActivityRaw(RegionData region)
        {
            if (region.activity == null) return 0f;
            
            // Priority: prediction > fmri > eeg
            // Note: Include all prediction values, including zero and negative
            if (region.activity.isPredicted)
            {
                return region.activity.predictionValue;
            }
            
            if (region.activity.fmri != null)
            {
                return region.activity.fmri.amplitude;
            }
            
            if (region.activity.eeg != null)
            {
                return region.activity.eeg.amplitude;
            }
            
            return 0f;
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
            
            GameObject lineObj = new GameObject(string.Format("Connection_{0}_{1}", conn.source, conn.target));
            lineObj.transform.SetParent(transform);
            
            LineRenderer line = lineObj.AddComponent<LineRenderer>();
            
            if (connectionMaterial != null)
            {
                line.material = connectionMaterial;
            }
            else
            {
                line.material = new Material(Shader.Find("Sprites/Default"));
            }
            
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
            foreach (GameObject obj in regionObjects.Values)
            {
                if (obj != null)
                {
                    Destroy(obj);
                }
            }
            regionObjects.Clear();
            regionRenderers.Clear();
            
            foreach (LineRenderer line in connectionLines)
            {
                if (line != null)
                {
                    Destroy(line.gameObject);
                }
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
            while (isPlaying && loadedSequence != null && currentFrame < loadedSequence.Count)
            {
                currentState = loadedSequence[currentFrame];
                UpdateVisualization();
                currentFrame++;
                
                if (currentFrame >= loadedSequence.Count)
                {
                    currentFrame = 0;
                }
                
                yield return new WaitForSeconds(1f / fps);
            }
            
            isPlaying = false;
        }
        
        /// <summary>
        /// 跳转到指定帧（用于进度条控制）
        /// </summary>
        public void SetFrame(int frameIndex)
        {
            if (loadedSequence == null || loadedSequence.Count == 0)
            {
                return;
            }
            
            currentFrame = Mathf.Clamp(frameIndex, 0, loadedSequence.Count - 1);
            currentState = loadedSequence[currentFrame];
            UpdateVisualization();
        }
        
        /// <summary>
        /// 获取当前帧索引
        /// </summary>
        public int GetCurrentFrame()
        {
            return currentFrame;
        }
        
        /// <summary>
        /// 获取总帧数
        /// </summary>
        public int GetTotalFrames()
        {
            return loadedSequence != null ? loadedSequence.Count : 0;
        }
        
        /// <summary>
        /// 获取当前选中的脑区ID
        /// </summary>
        public int GetSelectedRegionId()
        {
            return selectedRegionId;
        }
        
        /// <summary>
        /// 获取当前状态
        /// </summary>
        public BrainStateData GetCurrentState()
        {
            return currentState;
        }
    }
}
