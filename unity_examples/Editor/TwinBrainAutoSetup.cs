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
                "5. 配置摄像机",
                MessageType.Info
            );
            
            EditorGUILayout.Space();
            
            GUILayout.Label("设置选项", EditorStyles.boldLabel);
            
            objFolderPath = EditorGUILayout.TextField("OBJ文件夹路径", objFolderPath);
            createBrainManager = EditorGUILayout.Toggle("创建BrainManager", createBrainManager);
            createExampleSphere = EditorGUILayout.Toggle("创建示例球体预制体", createExampleSphere);
            setupCamera = EditorGUILayout.Toggle("配置摄像机", setupCamera);
            
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
                
                EditorUtility.DisplayProgressBar("TwinBrain自动设置", "完成！", 1f);
                EditorUtility.DisplayDialog(
                    "设置完成",
                    "TwinBrain场景设置完成！\n\n请检查Hierarchy中的BrainManager对象。\n\n" +
                    "注意：OBJ文件已导入并自动设置，无需手动配置每个文件。\n" +
                    "在BrainVisualization组件中勾选'Use Obj Models'即可使用所有OBJ模型。",
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
            
            brainManager.AddComponent(System.Type.GetType("BrainVisualization"));
            
            Selection.activeGameObject = brainManager;
            
            Debug.Log("BrainManager创建完成");
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
    }
}
