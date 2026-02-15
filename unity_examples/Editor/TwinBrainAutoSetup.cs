using UnityEngine;
using UnityEditor;
using System.IO;

namespace TwinBrain.Editor
{
    /// <summary>
    /// 自动化Unity设置工具 - 解决200+个OBJ文件需要手动配置的问题
    /// 
    /// 功能：
    /// 1. 自动导入OBJ文件并设置导入选项（缩放、材质等）
    /// 2. 创建BrainManager GameObject和组件
    /// 3. 创建示例预制体
    /// 4. 配置场景基本设置
    /// 
    /// 使用方法：
    /// Unity菜单 -> TwinBrain -> 自动设置场景
    /// </summary>
    public class TwinBrainAutoSetup : EditorWindow
    {
        private string objFolderPath = "Assets/StreamingAssets/OBJ";
        private bool createBrainManager = true;
        private bool createExampleSphere = true;
        private bool setupCamera = true;
        private bool setupStimulationUI = true;  // New option
        private Vector2 scrollPosition;
        
        [MenuItem("TwinBrain/自动设置场景", false, 1)]
        public static void ShowWindow()
        {
            var window = GetWindow<TwinBrainAutoSetup>("TwinBrain自动设置");
            window.minSize = new Vector2(400, 500);
            window.Show();
        }
        
        void OnGUI()
        {
            scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);
            
            GUILayout.Label("TwinBrain 自动场景设置", EditorStyles.boldLabel);
            EditorGUILayout.Space();
            
            EditorGUILayout.HelpBox(
                "此工具将自动完成以下设置：\n" +
                "1. 导入并配置OBJ文件（如果存在）\n" +
                "2. 创建BrainManager GameObject\n" +
                "3. 添加必要的组件\n" +
                "4. 创建示例预制体\n" +
                "5. 配置摄像机\n" +
                "6. 创建虚拟刺激UI（可选）",
                MessageType.Info
            );
            
            EditorGUILayout.Space();
            
            GUILayout.Label("设置选项", EditorStyles.boldLabel);
            
            objFolderPath = EditorGUILayout.TextField("OBJ文件夹路径", objFolderPath);
            createBrainManager = EditorGUILayout.Toggle("创建BrainManager", createBrainManager);
            createExampleSphere = EditorGUILayout.Toggle("创建示例球体预制体", createExampleSphere);
            setupCamera = EditorGUILayout.Toggle("配置摄像机", setupCamera);
            setupStimulationUI = EditorGUILayout.Toggle("创建虚拟刺激UI", setupStimulationUI);
            
            EditorGUILayout.Space();
            
            if (GUILayout.Button("开始自动设置", GUILayout.Height(40)))
            {
                RunAutoSetup();
            }
            
            EditorGUILayout.Space();
            
            if (GUILayout.Button("仅导入OBJ文件", GUILayout.Height(30)))
            {
                ImportOBJFiles();
            }
            
            if (GUILayout.Button("仅创建BrainManager", GUILayout.Height(30)))
            {
                CreateBrainManagerOnly();
            }
            
            EditorGUILayout.EndScrollView();
        }
        
        void RunAutoSetup()
        {
            if (!EditorUtility.DisplayDialog(
                "确认自动设置",
                "这将在当前场景中创建和配置对象。\n\n建议先保存当前场景。\n\n是否继续？",
                "继续", "取消"))
            {
                return;
            }
            
            EditorUtility.DisplayProgressBar("TwinBrain自动设置", "开始设置...", 0f);
            
            try
            {
                EditorUtility.DisplayProgressBar("TwinBrain自动设置", "导入OBJ文件...", 0.2f);
                ImportOBJFiles();
                
                if (createExampleSphere)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain自动设置", "创建示例预制体...", 0.4f);
                    CreateExamplePrefab();
                }
                
                if (createBrainManager)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain自动设置", "创建BrainManager...", 0.6f);
                    CreateBrainManagerOnly();
                }
                
                if (setupCamera)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain自动设置", "配置摄像机...", 0.8f);
                    SetupCamera();
                }
                
                if (setupStimulationUI)
                {
                    EditorUtility.DisplayProgressBar("TwinBrain自动设置", "创建虚拟刺激UI...", 0.9f);
                    SetupStimulationUI();
                }
                
                EditorUtility.DisplayProgressBar("TwinBrain自动设置", "完成！", 1f);
                EditorUtility.DisplayDialog(
                    "设置完成",
                    "TwinBrain场景设置完成！\n\n请检查Hierarchy中的BrainManager对象。\n\n" +
                    "注意：OBJ文件已导入并自动设置，无需手动配置每个文件。\n" +
                    "在BrainVisualization组件中勾选'Use Obj Models'即可使用所有OBJ模型。\n\n" +
                    "虚拟刺激UI已创建（如果勾选），可通过Canvas/StimulationPanel访问。",
                    "确定"
                );
            }
            catch (System.Exception e)
            {
                EditorUtility.DisplayDialog("错误", $"设置过程中出现错误：\n{e.Message}", "确定");
                Debug.LogError($"TwinBrain自动设置失败: {e}");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }
        
        void ImportOBJFiles()
        {
            if (!Directory.Exists(objFolderPath))
            {
                Debug.LogWarning($"OBJ文件夹不存在: {objFolderPath}");
                return;
            }
            
            string[] objFiles = Directory.GetFiles(objFolderPath, "*.obj", SearchOption.TopDirectoryOnly);
            
            if (objFiles.Length == 0)
            {
                Debug.LogWarning("未找到OBJ文件");
                return;
            }
            
            Debug.Log($"找到 {objFiles.Length} 个OBJ文件，开始自动配置...");
            
            int configured = 0;
            foreach (string objPath in objFiles)
            {
                ModelImporter importer = AssetImporter.GetAtPath(objPath) as ModelImporter;
                if (importer != null)
                {
                    importer.globalScale = 0.01f;
                    importer.importMaterials = true;
                    importer.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
                    importer.SaveAndReimport();
                    configured++;
                }
                
                if (configured % 50 == 0)
                {
                    EditorUtility.DisplayProgressBar(
                        "配置OBJ文件",
                        $"已配置 {configured}/{objFiles.Length} 个文件...",
                        (float)configured / objFiles.Length
                    );
                }
            }
            
            AssetDatabase.Refresh();
            Debug.Log($"成功配置 {configured} 个OBJ文件");
        }
        
        void CreateExamplePrefab()
        {
            string prefabPath = "Assets/TwinBrain/Prefabs";
            if (!Directory.Exists(prefabPath))
            {
                Directory.CreateDirectory(prefabPath);
                AssetDatabase.Refresh();
            }
            
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "BrainRegion";
            sphere.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
            
            Material mat = new Material(Shader.Find("Standard"));
            mat.name = "RegionMaterial";
            sphere.GetComponent<Renderer>().material = mat;
            
            string prefabFullPath = Path.Combine(prefabPath, "BrainRegion.prefab");
            PrefabUtility.SaveAsPrefabAsset(sphere, prefabFullPath);
            
            DestroyImmediate(sphere);
            
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            
            Debug.Log($"创建示例预制体: {prefabFullPath}");
        }
        
        void CreateBrainManagerOnly()
        {
            GameObject existing = GameObject.Find("BrainManager");
            if (existing != null)
            {
                if (!EditorUtility.DisplayDialog(
                    "BrainManager已存在",
                    "场景中已存在BrainManager对象。\n\n是否删除并重新创建？",
                    "重新创建", "取消"))
                {
                    return;
                }
                DestroyImmediate(existing);
            }
            
            GameObject brainManager = new GameObject("BrainManager");
            brainManager.transform.position = Vector3.zero;
            
            // Add BrainVisualization component
            var brainVis = brainManager.AddComponent(System.Type.GetType("TwinBrain.BrainVisualization"));
            
            // Add WebSocketClient component for real-time communication
            var wsClient = brainManager.AddComponent(System.Type.GetType("TwinBrain.WebSocketClient"));
            
            // Add StimulationInput component if creating UI
            if (setupStimulationUI)
            {
                var stimInput = brainManager.AddComponent(System.Type.GetType("TwinBrain.StimulationInput"));
                Debug.Log("✓ 已添加 StimulationInput 组件");
            }
            
            Selection.activeGameObject = brainManager;
            
            Debug.Log("BrainManager创建完成（含WebSocketClient" + 
                      (setupStimulationUI ? "和StimulationInput" : "") + "）");
        }
        
        void SetupStimulationUI()
        {
            // Create Canvas if it doesn't exist
            Canvas canvas = FindObjectOfType<Canvas>();
            if (canvas == null)
            {
                GameObject canvasObj = new GameObject("Canvas");
                canvas = canvasObj.AddComponent<Canvas>();
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                canvasObj.AddComponent<UnityEngine.UI.CanvasScaler>();
                canvasObj.AddComponent<UnityEngine.UI.GraphicRaycaster>();
                Debug.Log("✓ 创建 Canvas");
            }
            
            // Create StimulationPanel
            GameObject panel = new GameObject("StimulationPanel");
            panel.transform.SetParent(canvas.transform, false);
            
            var panelImage = panel.AddComponent<UnityEngine.UI.Image>();
            panelImage.color = new Color(0.2f, 0.2f, 0.2f, 0.8f);
            
            var rectTransform = panel.GetComponent<RectTransform>();
            rectTransform.anchorMin = new Vector2(0.02f, 0.02f);
            rectTransform.anchorMax = new Vector2(0.35f, 0.4f);
            rectTransform.offsetMin = Vector2.zero;
            rectTransform.offsetMax = Vector2.zero;
            
            // Add title
            GameObject title = new GameObject("Title");
            title.transform.SetParent(panel.transform, false);
            var titleText = title.AddComponent<UnityEngine.UI.Text>();
            titleText.text = "虚拟刺激控制";
            titleText.font = GetPreferredFont();
            titleText.fontSize = 18;
            titleText.color = Color.white;
            titleText.alignment = TextAnchor.MiddleCenter;
            
            var titleRect = title.GetComponent<RectTransform>();
            titleRect.anchorMin = new Vector2(0.1f, 0.85f);
            titleRect.anchorMax = new Vector2(0.9f, 0.95f);
            titleRect.offsetMin = Vector2.zero;
            titleRect.offsetMax = Vector2.zero;
            
            // Add target regions input field
            GameObject inputFieldObj = new GameObject("TargetRegionsInput");
            inputFieldObj.transform.SetParent(panel.transform, false);
            
            var inputFieldImage = inputFieldObj.AddComponent<UnityEngine.UI.Image>();
            inputFieldImage.color = Color.white;
            
            var inputField = inputFieldObj.AddComponent<UnityEngine.UI.InputField>();
            
            GameObject placeholder = new GameObject("Placeholder");
            placeholder.transform.SetParent(inputFieldObj.transform, false);
            var placeholderText = placeholder.AddComponent<UnityEngine.UI.Text>();
            placeholderText.text = "输入目标脑区ID (逗号分隔)";
            placeholderText.font = GetPreferredFont();
            placeholderText.fontSize = 12;
            placeholderText.color = new Color(0.5f, 0.5f, 0.5f);
            placeholderText.fontStyle = FontStyle.Italic;
            
            GameObject textObj = new GameObject("Text");
            textObj.transform.SetParent(inputFieldObj.transform, false);
            var textComponent = textObj.AddComponent<UnityEngine.UI.Text>();
            textComponent.text = "";
            textComponent.font = GetPreferredFont();
            textComponent.fontSize = 12;
            textComponent.color = Color.black;
            textComponent.supportRichText = false;
            
            inputField.textComponent = textComponent;
            inputField.placeholder = placeholderText;
            
            var inputRect = inputFieldObj.GetComponent<RectTransform>();
            inputRect.anchorMin = new Vector2(0.1f, 0.7f);
            inputRect.anchorMax = new Vector2(0.9f, 0.8f);
            inputRect.offsetMin = Vector2.zero;
            inputRect.offsetMax = Vector2.zero;
            
            // Add amplitude slider
            GameObject sliderObj = new GameObject("AmplitudeSlider");
            sliderObj.transform.SetParent(panel.transform, false);
            
            var slider = sliderObj.AddComponent<UnityEngine.UI.Slider>();
            slider.minValue = 0f;
            slider.maxValue = 5f;
            slider.value = 1f;
            
            var sliderRect = sliderObj.GetComponent<RectTransform>();
            sliderRect.anchorMin = new Vector2(0.1f, 0.5f);
            sliderRect.anchorMax = new Vector2(0.9f, 0.6f);
            sliderRect.offsetMin = Vector2.zero;
            sliderRect.offsetMax = Vector2.zero;
            
            // Slider components
            GameObject background = new GameObject("Background");
            background.transform.SetParent(sliderObj.transform, false);
            var bgImage = background.AddComponent<UnityEngine.UI.Image>();
            bgImage.color = new Color(0.3f, 0.3f, 0.3f);
            
            GameObject fillArea = new GameObject("Fill Area");
            fillArea.transform.SetParent(sliderObj.transform, false);
            
            GameObject fill = new GameObject("Fill");
            fill.transform.SetParent(fillArea.transform, false);
            var fillImage = fill.AddComponent<UnityEngine.UI.Image>();
            fillImage.color = Color.green;
            
            GameObject handleSlideArea = new GameObject("Handle Slide Area");
            handleSlideArea.transform.SetParent(sliderObj.transform, false);
            
            GameObject handle = new GameObject("Handle");
            handle.transform.SetParent(handleSlideArea.transform, false);
            var handleImage = handle.AddComponent<UnityEngine.UI.Image>();
            handleImage.color = Color.white;
            
            slider.fillRect = fill.GetComponent<RectTransform>();
            slider.handleRect = handle.GetComponent<RectTransform>();
            
            // Add amplitude text
            GameObject amplitudeTextObj = new GameObject("AmplitudeText");
            amplitudeTextObj.transform.SetParent(panel.transform, false);
            var amplitudeText = amplitudeTextObj.AddComponent<UnityEngine.UI.Text>();
            amplitudeText.text = "Amplitude: 1.00";
            amplitudeText.font = GetPreferredFont();
            amplitudeText.fontSize = 12;
            amplitudeText.color = Color.white;
            
            var ampTextRect = amplitudeTextObj.GetComponent<RectTransform>();
            ampTextRect.anchorMin = new Vector2(0.1f, 0.4f);
            ampTextRect.anchorMax = new Vector2(0.9f, 0.5f);
            ampTextRect.offsetMin = Vector2.zero;
            ampTextRect.offsetMax = Vector2.zero;
            
            // Add pattern dropdown
            GameObject dropdownObj = new GameObject("PatternDropdown");
            dropdownObj.transform.SetParent(panel.transform, false);
            
            var dropdown = dropdownObj.AddComponent<UnityEngine.UI.Dropdown>();
            dropdown.options.Add(new UnityEngine.UI.Dropdown.OptionData("constant"));
            dropdown.options.Add(new UnityEngine.UI.Dropdown.OptionData("sine"));
            dropdown.options.Add(new UnityEngine.UI.Dropdown.OptionData("pulse"));
            dropdown.options.Add(new UnityEngine.UI.Dropdown.OptionData("ramp"));
            
            var dropdownRect = dropdownObj.GetComponent<RectTransform>();
            dropdownRect.anchorMin = new Vector2(0.1f, 0.25f);
            dropdownRect.anchorMax = new Vector2(0.9f, 0.35f);
            dropdownRect.offsetMin = Vector2.zero;
            dropdownRect.offsetMax = Vector2.zero;
            
            // Add send button
            GameObject buttonObj = new GameObject("SendButton");
            buttonObj.transform.SetParent(panel.transform, false);
            
            var buttonImage = buttonObj.AddComponent<UnityEngine.UI.Image>();
            buttonImage.color = new Color(0.2f, 0.6f, 0.2f);
            
            var button = buttonObj.AddComponent<UnityEngine.UI.Button>();
            
            GameObject buttonTextObj = new GameObject("Text");
            buttonTextObj.transform.SetParent(buttonObj.transform, false);
            var buttonText = buttonTextObj.AddComponent<UnityEngine.UI.Text>();
            buttonText.text = "应用刺激";
            buttonText.font = GetPreferredFont();
            buttonText.fontSize = 14;
            buttonText.color = Color.white;
            buttonText.alignment = TextAnchor.MiddleCenter;
            
            var buttonRect = buttonObj.GetComponent<RectTransform>();
            buttonRect.anchorMin = new Vector2(0.2f, 0.05f);
            buttonRect.anchorMax = new Vector2(0.8f, 0.18f);
            buttonRect.offsetMin = Vector2.zero;
            buttonRect.offsetMax = Vector2.zero;
            
            // Add status text
            GameObject statusTextObj = new GameObject("StatusText");
            statusTextObj.transform.SetParent(panel.transform, false);
            var statusText = statusTextObj.AddComponent<UnityEngine.UI.Text>();
            statusText.text = "Ready";
            statusText.font = GetPreferredFont();
            statusText.fontSize = 10;
            statusText.color = Color.yellow;
            statusText.alignment = TextAnchor.MiddleCenter;
            
            var statusRect = statusTextObj.GetComponent<RectTransform>();
            statusRect.anchorMin = new Vector2(0.1f, 0.15f);
            statusRect.anchorMax = new Vector2(0.9f, 0.2f);
            statusRect.offsetMin = Vector2.zero;
            statusRect.offsetMax = Vector2.zero;
            
            // Wire up StimulationInput component
            GameObject brainManager = GameObject.Find("BrainManager");
            if (brainManager != null)
            {
                var stimInput = brainManager.GetComponent(System.Type.GetType("TwinBrain.StimulationInput"));
                if (stimInput != null)
                {
                    // Use reflection to set fields
                    var stimType = stimInput.GetType();
                    stimType.GetField("targetRegionsInput").SetValue(stimInput, inputField);
                    stimType.GetField("amplitudeSlider").SetValue(stimInput, slider);
                    stimType.GetField("amplitudeText").SetValue(stimInput, amplitudeText);
                    stimType.GetField("patternDropdown").SetValue(stimInput, dropdown);
                    stimType.GetField("sendButton").SetValue(stimInput, button);
                    stimType.GetField("statusText").SetValue(stimInput, statusText);
                    
                    Debug.Log("✓ 虚拟刺激UI组件已连接到StimulationInput");
                }
            }
            
            Debug.Log("✓ 虚拟刺激UI创建完成");
        }
        
        void SetupCamera()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                Debug.LogWarning("未找到主摄像机");
                return;
            }
            
            mainCamera.transform.position = new Vector3(0, 5, -10);
            mainCamera.transform.rotation = Quaternion.Euler(30, 0, 0);
            mainCamera.clearFlags = CameraClearFlags.SolidColor;
            mainCamera.backgroundColor = Color.black;
            
            Debug.Log("摄像机配置完成");
        }
        
        /// <summary>
        /// Get preferred font with fallback to built-in Arial
        /// Tries to load custom Aller font, falls back to Arial if not found
        /// </summary>
        Font GetPreferredFont()
        {
            Font customFont = Resources.Load<Font>("Fonts/Aller");
            if (customFont != null)
            {
                return customFont;
            }
            
            // Fallback to built-in Arial
            Debug.LogWarning("Aller font not found at Resources/Fonts/Aller, using built-in Arial. " +
                           "To use the Aller font, place it at Assets/Resources/Fonts/Aller in your Unity project.");
            return Resources.GetBuiltinResource<Font>("Arial.ttf");
        }
    }
}
