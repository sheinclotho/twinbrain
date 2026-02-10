#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain Unity 一键式自动化设置脚本
===================================

完全自动化Unity工作流程：
1. 创建必需的文件夹结构（FreeSurfer文件夹 + 数据/缓存/模型输出文件夹）
2. 读取FreeSurfer文件并生成Unity配置
3. 自动生成所有必需的脚本和配置
4. 启动后端模型服务（支持加载训练好的模型）
5. 提供Unity中可用的按钮接口

使用方法:
    # 完整自动化设置
    python setup_unity_project.py --auto-setup
    
    # 使用FreeSurfer文件设置
    python setup_unity_project.py --freesurfer-dir ./freesurfer_files
    
    # 启动后端服务器（加载训练模型）
    python setup_unity_project.py --serve --model-path results/hetero_gnn_trained.pt
"""

import argparse
import json
import sys
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VERSION = "2.4"
DEFAULT_MODEL_FILENAME = "hetero_gnn_trained.pt"


class UnityWorkflowSetup:
    """Unity工作流一键式自动化设置"""
    
    def __init__(
        self,
        project_root: Path = None,
        output_base: Path = None
    ):
        """
        初始化设置管理器
        
        Args:
            project_root: 项目根目录
            output_base: 输出基础目录
        """
        self.project_root = project_root or Path(__file__).parent
        self.output_base = output_base or (self.project_root / "unity_project")
        
        # 定义标准文件夹结构
        self.freesurfer_dir = self.output_base / "freesurfer_files"
        self.data_dir = self.output_base / "brain_data"
        self.cache_dir = self.data_dir / "cache"
        self.model_output_dir = self.data_dir / "model_output"
        self.original_data_dir = self.data_dir / "original"
        self.unity_assets_dir = self.output_base / "Unity_Assets"
        self.unity_scripts_dir = self.unity_assets_dir / "Scripts"
        
    def create_folder_structure(self):
        """创建标准文件夹结构"""
        logger.info("="*80)
        logger.info("步骤 1: 创建文件夹结构")
        logger.info("="*80)
        
        folders = [
            (self.freesurfer_dir, "FreeSurfer表面数据文件"),
            (self.original_data_dir, "原始fMRI/EEG数据"),
            (self.cache_dir, "预处理缓存数据"),
            (self.model_output_dir, "模型预测输出"),
            (self.unity_assets_dir, "Unity资源文件"),
            (self.unity_scripts_dir, "Unity C#脚本"),
        ]
        
        for folder, description in folders:
            folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ 创建: {folder.relative_to(self.output_base)} - {description}")
        
        # 创建README文件
        self._create_folder_readmes()
        logger.info("  ✓ 文件夹结构创建完成")
        
    def _create_folder_readmes(self):
        """为每个文件夹创建README说明"""
        readmes = {
            self.freesurfer_dir: """# FreeSurfer 文件夹

放置FreeSurfer表面数据文件：
- lh.pial (左半球表面)
- rh.pial (右半球表面)  
- lh.Schaefer2018_200Parcels_7Networks_order.annot (左半球注释)
- rh.Schaefer2018_200Parcels_7Networks_order.annot (右半球注释)

这些文件用于一次性生成Unity前端的3D脑结构。
""",
            self.original_data_dir: """# 原始数据文件夹

放置原始脑成像数据：
- fMRI数据 (.nii.gz格式)
- EEG数据 (.edf, .set格式)
- DTI连接矩阵 (.npy格式)

这些数据将被预处理后用于可视化和模型训练。
""",
            self.cache_dir: """# 缓存文件夹

自动生成的预处理缓存数据：
- 预处理后的图数据 (.pt格式)
- 中间计算结果

可以安全删除以重新生成。
""",
            self.model_output_dir: """# 模型输出文件夹

模型预测输出和可视化数据：
- JSON格式的脑状态数据 (用于Unity加载)
- 预测轨迹数据
- 虚拟刺激响应数据

Unity通过读取这个文件夹中的JSON文件来显示动画。
""",
            self.unity_assets_dir: """# Unity资源文件夹

Unity项目资源文件：
- 3D模型 (.obj格式)
- 配置文件 (.json格式)
- 材质配置
- 脚本文件

将此文件夹导入Unity项目的Assets目录。
"""
        }
        
        for folder, content in readmes.items():
            readme_path = folder / "README.md"
            readme_path.write_text(content, encoding='utf-8')
    
    def process_freesurfer_files(
        self,
        lh_surface: Optional[Path] = None,
        rh_surface: Optional[Path] = None,
        lh_annot: Optional[Path] = None,
        rh_annot: Optional[Path] = None
    ):
        """
        处理FreeSurfer文件并生成Unity配置
        
        Args:
            lh_surface: 左半球表面文件路径
            rh_surface: 右半球表面文件路径
            lh_annot: 左半球注释文件路径
            rh_annot: 右半球注释文件路径
        """
        logger.info("\n" + "="*80)
        logger.info("步骤 2: 处理FreeSurfer文件")
        logger.info("="*80)
        
        # 如果提供了文件路径，复制到标准位置
        if any([lh_surface, rh_surface, lh_annot, rh_annot]):
            files_to_copy = [
                (lh_surface, self.freesurfer_dir / "lh.pial"),
                (rh_surface, self.freesurfer_dir / "rh.pial"),
                (lh_annot, self.freesurfer_dir / "lh.Schaefer2018_200Parcels_7Networks_order.annot"),
                (rh_annot, self.freesurfer_dir / "rh.Schaefer2018_200Parcels_7Networks_order.annot")
            ]
            
            for src, dst in files_to_copy:
                if src and Path(src).exists():
                    shutil.copy2(src, dst)
                    logger.info(f"  ✓ 复制: {src} -> {dst.name}")
        
        # 检查FreeSurfer文件是否存在
        required_files = {
            'lh_surface': self.freesurfer_dir / "lh.pial",
            'rh_surface': self.freesurfer_dir / "rh.pial",
            'lh_annot': self.freesurfer_dir / "lh.Schaefer2018_200Parcels_7Networks_order.annot",
            'rh_annot': self.freesurfer_dir / "rh.Schaefer2018_200Parcels_7Networks_order.annot"
        }
        
        missing_files = [name for name, path in required_files.items() if not path.exists()]
        
        if not missing_files:
            logger.info("  ✓ 所有FreeSurfer文件已就位")
            logger.info(f"    - lh.pial: {required_files['lh_surface']}")
            logger.info(f"    - rh.pial: {required_files['rh_surface']}")
            logger.info(f"    - lh.annot: {required_files['lh_annot']}")
            logger.info(f"    - rh.annot: {required_files['rh_annot']}")
            self._generate_unity_structure_from_freesurfer()
        else:
            logger.warning("  ⚠ FreeSurfer文件缺失，将使用默认配置")
            logger.info(f"  缺失文件: {', '.join(missing_files)}")
            logger.info("  提示: 将FreeSurfer文件放入以下位置:")
            for name, path in required_files.items():
                status = "✓" if path.exists() else "✗"
                logger.info(f"    {status} {path}")
            self._generate_default_unity_structure()
    
    def _generate_unity_structure_from_freesurfer(self):
        """从FreeSurfer文件生成Unity结构"""
        try:
            from unity_integration import WorkflowConfig, run_unity_workflow
            
            logger.info("  正在从FreeSurfer文件生成Unity配置...")
            
            config = WorkflowConfig(
                data_source='freesurfer',
                freesurfer_lh_surface=str(self.freesurfer_dir / "lh.pial"),
                freesurfer_rh_surface=str(self.freesurfer_dir / "rh.pial"),
                freesurfer_lh_annot=str(self.freesurfer_dir / "lh.Schaefer2018_200Parcels_7Networks_order.annot"),
                freesurfer_rh_annot=str(self.freesurfer_dir / "rh.Schaefer2018_200Parcels_7Networks_order.annot"),
                output_dir=str(self.unity_assets_dir),
                export_formats=['json', 'obj'],
                export_surface_mesh=True,
                generate_unity_config=True,
                generate_materials=True,
                start_time=0,
                end_time=0,  # 只生成结构，不生成时间序列数据
                time_step=1
            )
            
            results = run_unity_workflow(config)
            logger.info("  ✓ Unity结构生成完成")
            logger.info(f"  ✓ 生成文件: {len(results.get('output_files', []))} 个")
            
        except Exception as e:
            logger.error(f"  ❌ 生成Unity结构失败: {e}")
            logger.info("  回退到默认配置...")
            self._generate_default_unity_structure()
    
    def _generate_default_unity_structure(self):
        """生成默认Unity结构（无FreeSurfer）"""
        logger.info("  生成默认Unity配置...")
        
        # 生成默认配置文件
        default_config = {
            "atlas": "Schaefer200",
            "n_regions": 200,
            "description": "默认200脑区配置",
            "note": "请添加FreeSurfer文件以获取真实脑表面"
        }
        
        config_path = self.unity_assets_dir / "unity_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ 默认配置已保存: {config_path.name}")
    
    def generate_unity_scripts(self):
        """生成增强的Unity C#脚本"""
        logger.info("\n" + "="*80)
        logger.info("步骤 3: 生成Unity交互脚本")
        logger.info("="*80)
        
        # 复制基础脚本
        source_scripts_dir = self.project_root / "unity_examples"
        base_scripts = [
            "BrainVisualization.cs",
            "BrainConfigLoader.cs",
            "WebSocketClient.cs"
        ]
        
        for script in base_scripts:
            src = source_scripts_dir / script
            if src.exists():
                dst = self.unity_scripts_dir / script
                shutil.copy2(src, dst)
                logger.info(f"  ✓ 复制脚本: {script}")
        
        # 生成增强的交互控制脚本
        self._generate_data_loader_script()
        self._generate_animation_controller_script()
        self._generate_stimulation_input_script()
        self._generate_model_interface_script()
        
        logger.info("  ✓ Unity脚本生成完成")
    
    def _generate_data_loader_script(self):
        """生成数据加载脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// 数据加载器 - 从brain_data文件夹读取数据
/// 提供按钮接口用于加载和刷新数据
/// </summary>
public class BrainDataLoader : MonoBehaviour
{
    [Header("数据路径")]
    public string dataFolderPath = "../brain_data/model_output";
    
    [Header("UI引用")]
    public Button loadDataButton;
    public Button refreshButton;
    public Text statusText;
    
    [Header("可视化引用")]
    public BrainVisualization brainVis;
    
    private List<string> availableDataFiles = new List<string>();
    private int currentFileIndex = 0;
    
    void Start()
    {
        // 绑定按钮事件
        if (loadDataButton != null)
            loadDataButton.onClick.AddListener(OnLoadDataClicked);
        
        if (refreshButton != null)
            refreshButton.onClick.AddListener(OnRefreshClicked);
        
        // 初始扫描数据文件
        ScanDataFolder();
    }
    
    /// <summary>
    /// 扫描数据文件夹，获取可用的JSON文件列表
    /// </summary>
    public void ScanDataFolder()
    {
        availableDataFiles.Clear();
        
        string fullPath = Path.Combine(Application.dataPath, dataFolderPath);
        
        if (!Directory.Exists(fullPath))
        {
            UpdateStatus($"数据文件夹不存在: {fullPath}");
            return;
        }
        
        // 查找所有JSON文件
        string[] jsonFiles = Directory.GetFiles(fullPath, "*.json", SearchOption.AllDirectories);
        availableDataFiles = jsonFiles.OrderBy(f => f).ToList();
        
        UpdateStatus($"找到 {availableDataFiles.Count} 个数据文件");
    }
    
    /// <summary>
    /// 加载数据按钮点击事件
    /// </summary>
    public void OnLoadDataClicked()
    {
        if (availableDataFiles.Count == 0)
        {
            UpdateStatus("没有可用的数据文件");
            return;
        }
        
        string filePath = availableDataFiles[currentFileIndex];
        LoadDataFile(filePath);
        
        // 循环到下一个文件
        currentFileIndex = (currentFileIndex + 1) % availableDataFiles.Count;
    }
    
    /// <summary>
    /// 刷新按钮点击事件
    /// </summary>
    public void OnRefreshClicked()
    {
        ScanDataFolder();
        currentFileIndex = 0;
    }
    
    /// <summary>
    /// 加载指定的数据文件
    /// </summary>
    public void LoadDataFile(string filePath)
    {
        try
        {
            string jsonContent = File.ReadAllText(filePath);
            BrainStateData data = JsonUtility.FromJson<BrainStateData>(jsonContent);
            
            if (brainVis != null)
            {
                brainVis.LoadBrainStateData(data);
                UpdateStatus($"已加载: {Path.GetFileName(filePath)}");
            }
        }
        catch (System.Exception e)
        {
            UpdateStatus($"加载失败: {e.Message}");
        }
    }
    
    /// <summary>
    /// 加载所有数据文件作为时间序列
    /// </summary>
    public void LoadAllAsSequence()
    {
        if (availableDataFiles.Count == 0)
        {
            UpdateStatus("没有可用的数据文件");
            return;
        }
        
        UpdateStatus($"正在加载 {availableDataFiles.Count} 个时间点...");
        
        // 在协程中逐个加载
        StartCoroutine(LoadSequenceCoroutine());
    }
    
    private System.Collections.IEnumerator LoadSequenceCoroutine()
    {
        foreach (string filePath in availableDataFiles)
        {
            LoadDataFile(filePath);
            yield return new WaitForSeconds(0.1f); // 动画间隔
        }
        
        UpdateStatus("序列加载完成");
    }
    
    private void UpdateStatus(string message)
    {
        if (statusText != null)
            statusText.text = message;
        
        Debug.Log($"[DataLoader] {message}");
    }
}

/// <summary>
/// 脑状态数据结构
/// </summary>
[System.Serializable]
public class BrainStateData
{
    public int time_point;
    public float time_second;
    public RegionData[] regions;
    public ConnectionData[] connections;
}

[System.Serializable]
public class RegionData
{
    public int id;
    public string label;
    public float activity;
    public float[] position;
    public string network;
}

[System.Serializable]
public class ConnectionData
{
    public int source;
    public int target;
    public float strength;
}
''';
        
        script_path = self.unity_scripts_dir / "BrainDataLoader.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info("  ✓ 生成: BrainDataLoader.cs")
    
    def _generate_animation_controller_script(self):
        """生成动画控制脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 动画控制器 - 控制脑活动动画的播放
/// 时间轴显示演化过程
/// </summary>
public class AnimationController : MonoBehaviour
{
    [Header("UI控件")]
    public Slider timelineSlider;
    public Button playButton;
    public Button pauseButton;
    public Button stopButton;
    public Text timeLabel;
    
    [Header("动画设置")]
    public float fps = 10f; // 每秒帧数
    public bool loop = true;
    
    [Header("数据引用")]
    public BrainDataLoader dataLoader;
    
    private bool isPlaying = false;
    private float currentTime = 0f;
    private int totalFrames = 0;
    
    void Start()
    {
        // 绑定按钮
        if (playButton != null)
            playButton.onClick.AddListener(OnPlay);
        
        if (pauseButton != null)
            pauseButton.onClick.AddListener(OnPause);
        
        if (stopButton != null)
            stopButton.onClick.AddListener(OnStop);
        
        // 绑定时间轴滑块
        if (timelineSlider != null)
            timelineSlider.onValueChanged.AddListener(OnTimelineChanged);
        
        UpdateTimeLabel();
    }
    
    void Update()
    {
        if (isPlaying && totalFrames > 0)
        {
            currentTime += Time.deltaTime * fps;
            
            if (currentTime >= totalFrames)
            {
                if (loop)
                    currentTime = 0f;
                else
                {
                    currentTime = totalFrames - 1;
                    OnPause();
                }
            }
            
            UpdateVisualization();
        }
    }
    
    public void OnPlay()
    {
        isPlaying = true;
        Debug.Log("动画播放");
    }
    
    public void OnPause()
    {
        isPlaying = false;
        Debug.Log("动画暂停");
    }
    
    public void OnStop()
    {
        isPlaying = false;
        currentTime = 0f;
        UpdateVisualization();
        Debug.Log("动画停止");
    }
    
    public void OnTimelineChanged(float value)
    {
        if (!isPlaying && totalFrames > 0)
        {
            currentTime = value * (totalFrames - 1);
            UpdateVisualization();
        }
    }
    
    private void UpdateVisualization()
    {
        int frameIndex = Mathf.RoundToInt(currentTime);
        
        if (timelineSlider != null && !isPlaying)
            timelineSlider.value = (float)frameIndex / (totalFrames - 1);
        
        UpdateTimeLabel();
        
        // TODO: 通知可视化系统更新到指定帧
    }
    
    private void UpdateTimeLabel()
    {
        if (timeLabel != null)
        {
            int frame = Mathf.RoundToInt(currentTime);
            float seconds = currentTime / fps;
            timeLabel.text = $"时间: {seconds:F2}s (帧: {frame}/{totalFrames})";
        }
    }
    
    public void SetTotalFrames(int frames)
    {
        totalFrames = frames;
        
        if (timelineSlider != null)
        {
            timelineSlider.minValue = 0f;
            timelineSlider.maxValue = 1f;
        }
    }
}
''';
        
        script_path = self.unity_scripts_dir / "AnimationController.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info("  ✓ 生成: AnimationController.cs")
    
    def _generate_stimulation_input_script(self):
        """生成虚拟刺激输入脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 虚拟刺激输入控制器
/// 允许用户选择脑区并应用虚拟刺激
/// </summary>
public class StimulationInput : MonoBehaviour
{
    [Header("UI控件")]
    public Button selectRegionButton;
    public Button applyStimulationButton;
    public InputField amplitudeInput;
    public Dropdown patternDropdown;
    public Text selectedRegionsText;
    
    [Header("后端通信")]
    public ModelInterface modelInterface;
    
    private List<int> selectedRegionIds = new List<int>();
    
    void Start()
    {
        // 绑定按钮
        if (selectRegionButton != null)
            selectRegionButton.onClick.AddListener(OnSelectRegion);
        
        if (applyStimulationButton != null)
            applyStimulationButton.onClick.AddListener(OnApplyStimulation);
        
        UpdateSelectedRegionsDisplay();
    }
    
    void Update()
    {
        // 鼠标点击选择脑区
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit))
            {
                BrainRegion region = hit.collider.GetComponent<BrainRegion>();
                if (region != null)
                {
                    ToggleRegionSelection(region.regionId);
                }
            }
        }
    }
    
    public void OnSelectRegion()
    {
        Debug.Log("进入脑区选择模式 - 点击脑区进行选择");
    }
    
    public void ToggleRegionSelection(int regionId)
    {
        if (selectedRegionIds.Contains(regionId))
        {
            selectedRegionIds.Remove(regionId);
            Debug.Log($"取消选择脑区: {regionId}");
        }
        else
        {
            selectedRegionIds.Add(regionId);
            Debug.Log($"选择脑区: {regionId}");
        }
        
        UpdateSelectedRegionsDisplay();
    }
    
    public void OnApplyStimulation()
    {
        if (selectedRegionIds.Count == 0)
        {
            Debug.LogWarning("未选择任何脑区");
            return;
        }
        
        // 解析刺激参数
        float amplitude = 0.5f;
        if (amplitudeInput != null)
            float.TryParse(amplitudeInput.text, out amplitude);
        
        string pattern = "sine";
        if (patternDropdown != null && patternDropdown.options.Count > 0)
            pattern = patternDropdown.options[patternDropdown.value].text.ToLower();
        
        // 创建刺激配置
        StimulationConfig config = new StimulationConfig
        {
            targetRegions = selectedRegionIds.ToArray(),
            amplitude = amplitude,
            pattern = pattern,
            duration = 20,
            frequency = 10f
        };
        
        // 发送到后端
        if (modelInterface != null)
        {
            modelInterface.SendStimulation(config);
            Debug.Log($"发送虚拟刺激: {selectedRegionIds.Count} 个脑区, 强度: {amplitude}");
        }
    }
    
    public void ClearSelection()
    {
        selectedRegionIds.Clear();
        UpdateSelectedRegionsDisplay();
    }
    
    private void UpdateSelectedRegionsDisplay()
    {
        if (selectedRegionsText != null)
        {
            if (selectedRegionIds.Count == 0)
                selectedRegionsText.text = "未选择脑区";
            else
                selectedRegionsText.text = $"已选择: {string.Join(", ", selectedRegionIds)}";
        }
    }
}

/// <summary>
/// 刺激配置
/// </summary>
[System.Serializable]
public class StimulationConfig
{
    public int[] targetRegions;
    public float amplitude;
    public string pattern;
    public int duration;
    public float frequency;
}
''';
        
        script_path = self.unity_scripts_dir / "StimulationInput.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info("  ✓ 生成: StimulationInput.cs")
    
    def _generate_model_interface_script(self):
        """生成模型接口脚本"""
        script_content = '''using UnityEngine;
using System.Collections;
using System.Threading.Tasks;

/// <summary>
/// 模型接口 - 与后端Python模型通信
/// 支持发送刺激并接收预测结果
/// </summary>
public class ModelInterface : MonoBehaviour
{
    [Header("服务器设置")]
    public string serverUrl = "ws://localhost:8765";
    public bool autoConnect = true;
    
    [Header("数据保存")]
    public string outputFolder = "../brain_data/model_output";
    
    [Header("引用")]
    public WebSocketClient wsClient;
    public BrainDataLoader dataLoader;
    
    private bool isConnected = false;
    
    async void Start()
    {
        if (autoConnect)
        {
            await ConnectToServer();
        }
    }
    
    public async Task ConnectToServer()
    {
        if (wsClient == null)
        {
            wsClient = gameObject.AddComponent<WebSocketClient>();
            wsClient.serverUrl = serverUrl;
        }
        
        try
        {
            await wsClient.Connect();
            isConnected = true;
            
            // 监听消息
            wsClient.OnMessageReceived += OnModelResponse;
            
            Debug.Log($"✓ 已连接到模型服务器: {serverUrl}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"连接模型服务器失败: {e.Message}");
            isConnected = false;
        }
    }
    
    /// <summary>
    /// 发送虚拟刺激到后端模型
    /// </summary>
    public async void SendStimulation(StimulationConfig config)
    {
        if (!isConnected)
        {
            Debug.LogWarning("未连接到服务器");
            return;
        }
        
        // 构建请求
        var request = new
        {
            type = "simulate",
            stimulation = new
            {
                target_regions = config.targetRegions,
                amplitude = config.amplitude,
                pattern = config.pattern,
                duration = config.duration,
                frequency = config.frequency
            }
        };
        
        string jsonRequest = JsonUtility.ToJson(request);
        await wsClient.SendMessage(jsonRequest);
        
        Debug.Log("虚拟刺激已发送到后端");
    }
    
    /// <summary>
    /// 请求模型预测
    /// </summary>
    public async void RequestPrediction(int nSteps = 50)
    {
        if (!isConnected)
        {
            Debug.LogWarning("未连接到服务器");
            return;
        }
        
        var request = new
        {
            type = "predict",
            n_steps = nSteps
        };
        
        string jsonRequest = JsonUtility.ToJson(request);
        await wsClient.SendMessage(jsonRequest);
        
        Debug.Log($"请求预测 {nSteps} 步");
    }
    
    /// <summary>
    /// 处理模型响应
    /// </summary>
    private void OnModelResponse(string message)
    {
        try
        {
            // 解析响应
            ModelResponse response = JsonUtility.FromJson<ModelResponse>(message);
            
            if (response.type == "prediction" || response.type == "simulation")
            {
                Debug.Log($"收到模型输出: {response.type}");
                
                // 保存到文件
                SaveModelOutput(message, response.type);
                
                // 通知数据加载器刷新
                if (dataLoader != null)
                {
                    dataLoader.ScanDataFolder();
                }
            }
            else if (response.type == "error")
            {
                Debug.LogError($"模型错误: {response.message}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"处理响应失败: {e.Message}");
        }
    }
    
    /// <summary>
    /// 保存模型输出到文件
    /// </summary>
    private void SaveModelOutput(string jsonData, string dataType)
    {
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filename = $"{dataType}_{timestamp}.json";
        string fullPath = System.IO.Path.Combine(Application.dataPath, outputFolder, filename);
        
        try
        {
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(fullPath));
            System.IO.File.WriteAllText(fullPath, jsonData);
            Debug.Log($"✓ 模型输出已保存: {filename}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"保存失败: {e.Message}");
        }
    }
}

[System.Serializable]
public class ModelResponse
{
    public string type;
    public bool success;
    public string message;
}
''';
        
        script_path = self.unity_scripts_dir / "ModelInterface.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info("  ✓ 生成: ModelInterface.cs")
    
    def generate_startup_script(self):
        """生成启动脚本"""
        logger.info("\n" + "="*80)
        logger.info("步骤 4: 生成启动脚本")
        logger.info("="*80)
        
        # Python启动脚本
        startup_script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain 后端模型服务启动脚本
自动生成 - 请勿手动编辑
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unity_integration import BrainVisualizationServer
from unity_integration import BrainStateExporter, StimulationSimulator
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path):
    """加载训练好的模型"""
    if not Path(model_path).exists():
        logger.warning(f"模型文件不存在: {{model_path}}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(f"✓ 成功加载模型: {{model_path}}")
        return checkpoint
    except Exception as e:
        logger.error(f"加载模型失败: {{e}}")
        return None

def main():
    # 配置
    MODEL_PATH = "{self.project_root}/results/{DEFAULT_MODEL_FILENAME}"
    DATA_DIR = "{self.data_dir}"
    OUTPUT_DIR = "{self.model_output_dir}"
    PORT = 8765
    
    logger.info("="*80)
    logger.info("TwinBrain 后端模型服务")
    logger.info("="*80)
    
    # 加载模型
    model = load_trained_model(MODEL_PATH)
    
    # 创建导出器和模拟器
    exporter = BrainStateExporter(atlas_info=None, output_dir=str(OUTPUT_DIR))
    simulator = StimulationSimulator(n_regions=200)
    
    # 创建服务器
    server = BrainVisualizationServer(
        model=model,
        exporter=exporter,
        simulator=simulator,
        port=PORT
    )
    
    logger.info(f"数据目录: {{DATA_DIR}}")
    logger.info(f"输出目录: {{OUTPUT_DIR}}")
    logger.info(f"服务器端口: {{PORT}}")
    logger.info(f"Unity连接地址: ws://localhost:{{PORT}}")
    logger.info("")
    logger.info("服务器启动中...")
    logger.info("按 Ctrl+C 停止服务器")
    logger.info("="*80)
    
    # 启动服务器
    import asyncio
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\\n服务器已停止")

if __name__ == "__main__":
    main()
'''
        
        startup_path = self.output_base / "start_backend_server.py"
        startup_path.write_text(startup_script, encoding='utf-8')
        startup_path.chmod(0o755)  # Make executable
        logger.info(f"  ✓ 生成启动脚本: {startup_path.name}")
        
        # 生成批处理文件（Windows）
        batch_script = f'''@echo off
echo TwinBrain 后端服务器启动
echo ==============================
cd /d "%~dp0"
python start_backend_server.py
pause
'''
        batch_path = self.output_base / "start_backend_server.bat"
        batch_path.write_text(batch_script, encoding='utf-8')
        logger.info(f"  ✓ 生成Windows启动脚本: {batch_path.name}")
        
        # 生成Shell脚本（Linux/Mac）
        shell_script = f'''#!/bin/bash
echo "TwinBrain 后端服务器启动"
echo "=============================="
cd "$(dirname "$0")"
python3 start_backend_server.py
'''
        shell_path = self.output_base / "start_backend_server.sh"
        shell_path.write_text(shell_script, encoding='utf-8')
        shell_path.chmod(0o755)
        logger.info(f"  ✓ 生成Linux/Mac启动脚本: {shell_path.name}")
    
    def generate_documentation(self):
        """生成完整使用文档"""
        logger.info("\n" + "="*80)
        logger.info("步骤 5: 生成使用文档")
        logger.info("="*80)
        
        doc_content = f'''# TwinBrain Unity 全自动化工作流程

## 📁 文件夹结构

```
{self.output_base.name}/
├── freesurfer_files/          # FreeSurfer表面数据（一次性使用）
│   ├── lh.pial
│   ├── rh.pial
│   ├── lh.Schaefer2018_200Parcels_7Networks_order.annot
│   └── rh.Schaefer2018_200Parcels_7Networks_order.annot
│
├── brain_data/                # 数据文件夹
│   ├── original/              # 原始fMRI/EEG数据
│   ├── cache/                 # 预处理缓存
│   └── model_output/          # 模型输出（Unity读取此处）
│
├── Unity_Assets/              # Unity资源
│   ├── brain_structure.obj    # 3D脑模型
│   ├── unity_config.json      # 配置文件
│   └── Scripts/               # C#脚本
│       ├── BrainDataLoader.cs     # 数据加载器
│       ├── AnimationController.cs  # 动画控制
│       ├── StimulationInput.cs     # 刺激输入
│       └── ModelInterface.cs       # 模型接口
│
├── start_backend_server.py    # Python启动脚本
├── start_backend_server.bat   # Windows启动
├── start_backend_server.sh    # Linux/Mac启动
└── README_WORKFLOW.md         # 本文档
```

## 🚀 快速开始

### 1. 准备FreeSurfer文件（可选，一次性）

如果您有FreeSurfer文件，放入 `freesurfer_files/` 文件夹：
```bash
cp /path/to/lh.pial freesurfer_files/
cp /path/to/rh.pial freesurfer_files/
cp /path/to/lh.*.annot freesurfer_files/
cp /path/to/rh.*.annot freesurfer_files/
```

然后重新运行设置脚本以生成Unity结构：
```bash
python setup_unity_workflow.py --auto-setup
```

### 2. 启动后端模型服务器

**Windows:**
```
双击 start_backend_server.bat
```

**Linux/Mac:**
```bash
./start_backend_server.sh
```

**或直接使用Python:**
```bash
python start_backend_server.py
```

服务器启动后会显示：
```
✓ 服务器端口: 8765
✓ Unity连接地址: ws://localhost:8765
```

### 3. Unity中设置

#### 3.1 导入资源

1. 将 `Unity_Assets/` 文件夹内容复制到Unity项目的 `Assets/` 目录
2. Unity会自动导入所有脚本和模型

#### 3.2 创建场景

1. 创建空GameObject，命名 "BrainManager"
2. 添加以下组件：
   - `BrainVisualization`
   - `BrainDataLoader`
   - `AnimationController`
   - `StimulationInput`
   - `ModelInterface`

#### 3.3 配置组件

**BrainDataLoader:**
- Data Folder Path: `../brain_data/model_output`

**ModelInterface:**
- Server URL: `ws://localhost:8765`
- Output Folder: `../brain_data/model_output`
- Auto Connect: ✓

### 4. Unity中的交互按钮

创建UI Canvas，添加以下按钮：

#### 数据加载按钮
```
- 按钮: "加载数据"
- 功能: 从 model_output 文件夹读取JSON
- 脚本: BrainDataLoader.OnLoadDataClicked()
```

#### 动画控制按钮
```
- 按钮: "播放" / "暂停" / "停止"
- 滑块: 时间轴
- 功能: 控制大脑活动动画播放
- 脚本: AnimationController
```

#### 虚拟刺激按钮
```
- 按钮: "选择脑区" / "应用刺激"
- 输入框: 刺激强度
- 下拉框: 刺激模式（sine/pulse/ramp）
- 功能: 选择脑区并发送虚拟刺激
- 脚本: StimulationInput
```

#### 预测请求按钮
```
- 按钮: "请求预测"
- 功能: 请求模型预测未来脑状态
- 脚本: ModelInterface.RequestPrediction()
```

## 📊 完整工作流程

### 流程图

```
[FreeSurfer文件] → [一次性生成Unity结构] → [Unity前端准备完成]
                                                    ↓
[原始数据] → [预处理] → [缓存] → [模型训练/加载]
                                      ↓
    [Unity按钮: 请求预测/刺激] → [后端模型] → [输出JSON]
            ↑                                    ↓
            └───────── [保存到 model_output/] ←──┘
                               ↓
                    [Unity按钮: 加载数据] → [JSON动画]
                               ↓
                        [时间轴演化显示]
```

### 详细步骤

1. **初次设置（仅一次）**
   ```bash
   python setup_unity_workflow.py --auto-setup
   ```
   - 创建文件夹结构
   - 处理FreeSurfer文件（如果有）
   - 生成Unity配置和脚本

2. **启动后端服务器**
   ```bash
   python start_backend_server.py
   ```
   - 加载训练好的模型
   - 启动WebSocket服务器
   - 等待Unity连接

3. **Unity中操作**
   
   a) **加载现有数据并播放动画：**
   - 点击"加载数据"按钮
   - 点击"播放"开始动画
   - 使用时间轴滑块控制进度
   - 脑区颜色表示活动强度（蓝→黄→红）
   
   b) **输入虚拟刺激并获取预测：**
   - 点击鼠标选择目标脑区（可多选）
   - 设置刺激参数（强度、模式）
   - 点击"应用刺激"
   - 后端模型计算响应
   - 结果自动保存到 model_output/
   - 点击"刷新"按钮重新扫描文件
   - 点击"加载数据"查看预测结果
   
   c) **请求未来预测：**
   - 点击"请求预测"按钮
   - 后端生成多步预测
   - 结果保存为时间序列JSON
   - 加载后可播放动画查看演化

## 🎯 使用场景

### 场景1：查看历史数据
1. 将fMRI数据放入 `brain_data/original/`
2. 运行预处理（如需要）
3. Unity中点击"加载数据"
4. 播放动画查看脑活动演化

### 场景2：虚拟刺激实验
1. Unity中选择目标脑区
2. 设置刺激参数
3. 点击"应用刺激"
4. 等待后端计算（约数秒）
5. 点击"刷新" → "加载数据"
6. 查看刺激响应动画

### 场景3：未来预测
1. 确保模型已训练
2. Unity中点击"请求预测"
3. 后端生成50步预测
4. 自动保存到 model_output/
5. 加载并播放预测序列

## ⚙️ 高级配置

### 修改模型路径
编辑 `start_backend_server.py`:
```python
MODEL_PATH = "/path/to/your/trained_model.pt"
```

### 修改服务器端口
编辑 `start_backend_server.py`:
```python
PORT = 8765  # 改为其他端口
```

同时在Unity中更新 ModelInterface 的 Server URL。

### 自定义数据格式
编辑 `Unity_Assets/Scripts/BrainDataLoader.cs` 的 `BrainStateData` 结构。

## 📝 注意事项

1. **FreeSurfer文件是可选的**
   - 有：生成真实大脑表面
   - 无：使用默认球体布局

2. **后端服务器必须运行**
   - 虚拟刺激功能需要服务器
   - 数据加载不需要服务器

3. **数据文件夹路径**
   - 使用相对路径（../）
   - 确保Unity项目和数据文件夹位置关系正确

4. **模型文件**
   - 需要已训练的模型文件
   - 放在项目根目录的 results/ 文件夹

## 🔧 故障排除

### 问题：Unity无法连接服务器
**解决：**
1. 确认服务器正在运行
2. 检查端口是否正确（8765）
3. 检查防火墙设置

### 问题：找不到数据文件
**解决：**
1. 检查路径设置
2. 使用绝对路径进行测试
3. 确认文件夹结构正确

### 问题：动画不播放
**解决：**
1. 确认数据已加载
2. 检查时间轴设置
3. 查看Unity Console错误信息

## 📞 支持

如有问题，请查看：
- 项目主README: `../README_CN.md`
- Unity文档: `../docs/Unity工作流说明.md`
- GitHub Issues

---

**生成时间**: {self._get_timestamp()}
**工具版本**: TwinBrain v{VERSION}
'''
        
        doc_path = self.output_base / "README_WORKFLOW.md"
        doc_path.write_text(doc_content, encoding='utf-8')
        logger.info(f"  ✓ 使用文档已生成: {doc_path.name}")
    
    def _get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_summary(self):
        """打印完成摘要"""
        logger.info("\n" + "="*80)
        logger.info("✅ Unity全自动化工作流设置完成！")
        logger.info("="*80)
        logger.info("\n📁 生成的文件夹：")
        
        # Use absolute paths if can't get relative
        try:
            freesurfer_rel = self.freesurfer_dir.relative_to(self.project_root)
            data_rel = self.data_dir.relative_to(self.project_root)
            assets_rel = self.unity_assets_dir.relative_to(self.project_root)
        except ValueError:
            # If paths are not relative, use absolute paths
            freesurfer_rel = self.freesurfer_dir
            data_rel = self.data_dir
            assets_rel = self.unity_assets_dir
        
        logger.info(f"  - {freesurfer_rel}")
        logger.info(f"  - {data_rel}")
        logger.info(f"  - {assets_rel}")
        
        logger.info("\n📜 生成的脚本：")
        scripts = list(self.unity_scripts_dir.glob("*.cs"))
        for script in scripts:
            logger.info(f"  - {script.name}")
        
        logger.info("\n🚀 下一步操作：")
        logger.info("  1. [可选] 添加FreeSurfer文件到 freesurfer_files/ 并重新运行")
        
        try:
            server_rel = (self.output_base / "start_backend_server.py").relative_to(self.project_root)
            assets_rel2 = self.unity_assets_dir.relative_to(self.project_root)
            readme_rel = (self.output_base / "README_WORKFLOW.md").relative_to(self.project_root)
        except ValueError:
            server_rel = self.output_base / "start_backend_server.py"
            assets_rel2 = self.unity_assets_dir
            readme_rel = self.output_base / "README_WORKFLOW.md"
        
        logger.info(f"  2. 启动后端服务器: python {server_rel}")
        logger.info(f"  3. 在Unity中导入 {assets_rel2}/ 文件夹")
        logger.info("  4. 按照 README_WORKFLOW.md 配置Unity场景")
        logger.info("  5. 运行并测试！")
        
        logger.info(f"\n📖 详细文档: {readme_rel}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity 一键式自动化设置",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--auto-setup',
        action='store_true',
        help='运行完整自动化设置'
    )
    
    parser.add_argument(
        '--freesurfer-dir',
        type=Path,
        help='FreeSurfer文件目录路径'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='输出目录（默认: ./unity_project）'
    )
    
    parser.add_argument(
        '--serve',
        action='store_true',
        help='启动后端服务器'
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        help='训练模型文件路径'
    )
    
    args = parser.parse_args()
    
    # 创建设置管理器
    setup = UnityWorkflowSetup(output_base=args.output_dir)
    
    if args.auto_setup:
        # 完整自动化设置
        logger.info("开始Unity全自动化工作流设置...")
        logger.info("="*80)
        
        # 步骤1: 创建文件夹结构
        setup.create_folder_structure()
        
        # 步骤2: 处理FreeSurfer文件
        if args.freesurfer_dir and args.freesurfer_dir.exists():
            lh_surface = args.freesurfer_dir / "lh.pial"
            rh_surface = args.freesurfer_dir / "rh.pial"
            lh_annot = list(args.freesurfer_dir.glob("lh.*.annot"))
            rh_annot = list(args.freesurfer_dir.glob("rh.*.annot"))
            
            setup.process_freesurfer_files(
                lh_surface=lh_surface if lh_surface.exists() else None,
                rh_surface=rh_surface if rh_surface.exists() else None,
                lh_annot=lh_annot[0] if lh_annot else None,
                rh_annot=rh_annot[0] if rh_annot else None
            )
        else:
            setup.process_freesurfer_files()
        
        # 步骤3: 生成Unity脚本
        setup.generate_unity_scripts()
        
        # 步骤4: 生成启动脚本
        setup.generate_startup_script()
        
        # 步骤5: 生成文档
        setup.generate_documentation()
        
        # 打印摘要
        setup.print_summary()
        
    elif args.serve:
        # 启动服务器模式
        logger.info("启动后端模型服务器...")
        
        # 执行启动脚本
        startup_script = setup.output_base / "start_backend_server.py"
        if startup_script.exists():
            subprocess.run([sys.executable, str(startup_script)])
        else:
            logger.error(f"启动脚本不存在: {startup_script}")
            logger.info("请先运行 --auto-setup")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
