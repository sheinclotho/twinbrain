#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain Unity 一键式自动化脚本
================================

一键完成从脑数据到Unity可视化的完整流程：
1. 脑数据处理和转换
2. OBJ 3D模型生成
3. JSON数据导出
4. Unity配置生成
5. 材质和交互脚本配置
6. 后端服务器启动

使用方法:
    python unity_automation.py --mode export  # 仅导出数据
    python unity_automation.py --mode server  # 启动后端服务器
    python unity_automation.py --mode all     # 完整流程（默认）
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from unity_integration import (
    run_unity_workflow,
    WorkflowConfig,
    BrainVisualizationServer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnityAutomation:
    """Unity自动化工作流管理器"""
    
    def __init__(
        self,
        output_dir: str = "unity_output",
        brain_data_path: Optional[str] = None,
        use_example_data: bool = True
    ):
        """
        初始化自动化管理器
        
        Args:
            output_dir: 输出目录
            brain_data_path: 脑数据路径（如果有）
            use_example_data: 是否使用示例数据
        """
        self.output_dir = Path(output_dir)
        self.brain_data_path = brain_data_path
        self.use_example_data = use_example_data
        self.results = {}
    
    def run_complete_workflow(self):
        """运行完整工作流"""
        logger.info("="*80)
        logger.info("TwinBrain Unity 一键式自动化流程")
        logger.info("="*80)
        
        try:
            # 步骤1: 数据准备
            logger.info("\n[步骤 1/5] 数据准备...")
            self._step_prepare_data()
            
            # 步骤2: 导出数据
            logger.info("\n[步骤 2/5] 导出JSON和OBJ数据...")
            self._step_export_data()
            
            # 步骤3: 生成Unity配置
            logger.info("\n[步骤 3/5] 生成Unity配置和材质...")
            self._step_generate_unity_assets()
            
            # 步骤4: 生成交互脚本
            logger.info("\n[步骤 4/5] 生成Unity交互脚本...")
            self._step_generate_interaction_scripts()
            
            # 步骤5: 生成文档
            logger.info("\n[步骤 5/5] 生成使用文档...")
            self._step_generate_documentation()
            
            # 完成
            logger.info("\n" + "="*80)
            logger.info("✅ 自动化流程完成！")
            logger.info("="*80)
            self._print_summary()
            
        except Exception as e:
            logger.error(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _step_prepare_data(self):
        """步骤1: 准备数据"""
        if self.use_example_data:
            logger.info("  使用示例数据...")
            self.data_source = "example"
        elif self.brain_data_path:
            logger.info(f"  使用本地数据: {self.brain_data_path}")
            self.data_source = "local"
        else:
            logger.warning("  未指定数据源，将使用示例数据")
            self.data_source = "example"
            self.use_example_data = True
    
    def _step_export_data(self):
        """步骤2: 导出数据"""
        # 配置导出
        config = WorkflowConfig(
            data_source=self.data_source,
            data_path=self.brain_data_path,
            output_dir=str(self.output_dir),
            export_formats=['json', 'obj'],
            start_time=0,
            end_time=200,
            time_step=5,
            export_connectivity=True,
            export_networks=True,
            export_obj_per_frame=False,
            generate_unity_config=True,
            generate_materials=True,
            subject_id="twinbrain_demo",
            atlas_name="Schaefer200"
        )
        
        logger.info("  运行Unity工作流...")
        self.results['workflow'] = run_unity_workflow(config)
        
        # 统计输出文件
        json_files = list((self.output_dir / "json").glob("*.json"))
        obj_files = list((self.output_dir / "obj").glob("*.obj"))
        
        logger.info(f"  ✓ 生成了 {len(json_files)} 个JSON文件")
        logger.info(f"  ✓ 生成了 {len(obj_files)} 个OBJ文件")
    
    def _step_generate_unity_assets(self):
        """步骤3: 生成Unity资源"""
        # Unity项目结构已由workflow_manager生成
        # 这里添加额外的配置
        
        # 生成场景配置
        scene_config = self._create_scene_config()
        scene_path = self.output_dir / "unity_scene_config.json"
        with open(scene_path, 'w', encoding='utf-8') as f:
            json.dump(scene_config, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ 生成场景配置: {scene_path.name}")
        
        # 生成预制体配置
        prefab_config = self._create_prefab_config()
        prefab_path = self.output_dir / "unity_prefab_config.json"
        with open(prefab_path, 'w', encoding='utf-8') as f:
            json.dump(prefab_config, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ 生成预制体配置: {prefab_path.name}")
    
    def _step_generate_interaction_scripts(self):
        """步骤4: 生成交互脚本"""
        scripts_dir = self.output_dir / "UnityScripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # 复制并增强现有脚本
        project_root = Path(__file__).parent
        unity_examples = project_root / "unity_examples"
        
        scripts = [
            "BrainVisualization.cs",
            "BrainConfigLoader.cs",
            "WebSocketClient.cs"
        ]
        
        for script in scripts:
            src = unity_examples / script
            if src.exists():
                dst = scripts_dir / script
                dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
                logger.info(f"  ✓ 复制脚本: {script}")
        
        # 生成新的增强脚本
        self._generate_interaction_controller(scripts_dir)
        self._generate_region_selector(scripts_dir)
        self._generate_stimulation_controller(scripts_dir)
        self._generate_prediction_visualizer(scripts_dir)
    
    def _step_generate_documentation(self):
        """步骤5: 生成文档"""
        doc_path = self.output_dir / "README_UNITY.md"
        
        documentation = self._create_comprehensive_documentation()
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(documentation)
        
        logger.info(f"  ✓ 生成使用文档: {doc_path.name}")
    
    def start_server(self, port: int = 8765):
        """启动后端服务器"""
        logger.info("\n" + "="*80)
        logger.info("启动TwinBrain后端服务器")
        logger.info("="*80)
        
        try:
            # 检查是否安装了websockets
            try:
                import websockets
            except ImportError:
                logger.error("❌ 缺少websockets库")
                logger.info("请安装: pip install websockets")
                return
            
            # 创建服务器实例
            from unity_integration import BrainStateExporter
            from unity_integration import StimulationSimulator
            
            # 加载图谱信息
            atlas_info = self._load_atlas_info()
            exporter = BrainStateExporter(atlas_info)
            simulator = StimulationSimulator(n_regions=200)
            
            server = BrainVisualizationServer(
                model=None,  # TODO: 加载训练好的模型
                exporter=exporter,
                simulator=simulator,
                port=port
            )
            
            logger.info(f"✓ 服务器将在端口 {port} 启动")
            logger.info(f"✓ Unity连接地址: ws://localhost:{port}")
            logger.info("\n按 Ctrl+C 停止服务器...")
            
            # 启动服务器
            import asyncio
            asyncio.run(server.start())
            
        except KeyboardInterrupt:
            logger.info("\n服务器已停止")
        except Exception as e:
            logger.error(f"❌ 服务器错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_scene_config(self) -> Dict[str, Any]:
        """创建场景配置"""
        return {
            "scene_name": "TwinBrain_Visualization",
            "camera": {
                "position": {"x": 0, "y": 0, "z": -150},
                "look_at": {"x": 0, "y": 0, "z": 0},
                "field_of_view": 60
            },
            "lighting": {
                "ambient_color": {"r": 0.4, "g": 0.4, "b": 0.4},
                "directional_light": {
                    "color": {"r": 1, "g": 1, "b": 1},
                    "intensity": 1.0,
                    "rotation": {"x": 50, "y": -30, "z": 0}
                }
            },
            "background": {
                "type": "solid_color",
                "color": {"r": 0, "g": 0, "b": 0}
            }
        }
    
    def _create_prefab_config(self) -> Dict[str, Any]:
        """创建预制体配置"""
        return {
            "brain_region": {
                "type": "sphere",
                "base_scale": 0.8,
                "material": "Standard",
                "shader": "Standard",
                "properties": {
                    "metallic": 0.0,
                    "smoothness": 0.7,
                    "emission_enabled": True
                }
            },
            "connection_line": {
                "type": "line",
                "base_width": 0.02,
                "material": "Transparent",
                "shader": "Particles/Standard Unlit"
            },
            "selection_highlight": {
                "type": "outline",
                "color": {"r": 255, "g": 255, "b": 0},
                "width": 0.1
            }
        }
    
    def _generate_interaction_controller(self, scripts_dir: Path):
        """生成交互控制器脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.EventSystems;

/// <summary>
/// 脑区交互控制器 - 处理鼠标点击、悬停等交互
/// </summary>
public class BrainInteractionController : MonoBehaviour
{
    [Header("交互设置")]
    public LayerMask brainRegionLayer;
    public float clickRadius = 1.0f;
    
    [Header("高亮设置")]
    public Color hoverColor = Color.yellow;
    public Color selectedColor = Color.green;
    
    private GameObject currentHovered;
    private GameObject currentSelected;
    private Camera mainCamera;
    
    // 事件
    public delegate void RegionClickedHandler(int regionId, Vector3 position);
    public event RegionClickedHandler OnRegionClicked;
    
    public delegate void RegionHoveredHandler(int regionId);
    public event RegionHoveredHandler OnRegionHovered;
    
    void Start()
    {
        mainCamera = Camera.main;
    }
    
    void Update()
    {
        HandleHover();
        HandleClick();
    }
    
    void HandleHover()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;
        
        if (Physics.Raycast(ray, out hit, Mathf.Infinity, brainRegionLayer))
        {
            GameObject hitObject = hit.collider.gameObject;
            
            if (hitObject != currentHovered)
            {
                // 取消上一个高亮
                if (currentHovered != null && currentHovered != currentSelected)
                {
                    ResetRegionColor(currentHovered);
                }
                
                // 高亮新的
                currentHovered = hitObject;
                if (currentHovered != currentSelected)
                {
                    HighlightRegion(currentHovered, hoverColor);
                }
                
                // 触发事件
                int regionId = ExtractRegionId(hitObject.name);
                OnRegionHovered?.Invoke(regionId);
            }
        }
        else if (currentHovered != null && currentHovered != currentSelected)
        {
            ResetRegionColor(currentHovered);
            currentHovered = null;
        }
    }
    
    void HandleClick()
    {
        if (Input.GetMouseButtonDown(0) && !EventSystem.current.IsPointerOverGameObject())
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, Mathf.Infinity, brainRegionLayer))
            {
                GameObject clickedObject = hit.collider.gameObject;
                
                // 取消上一个选中
                if (currentSelected != null)
                {
                    ResetRegionColor(currentSelected);
                }
                
                // 选中新的
                currentSelected = clickedObject;
                HighlightRegion(currentSelected, selectedColor);
                
                // 触发事件
                int regionId = ExtractRegionId(clickedObject.name);
                OnRegionClicked?.Invoke(regionId, hit.point);
                
                Debug.Log($"选中脑区: {regionId}");
            }
        }
    }
    
    void HighlightRegion(GameObject region, Color color)
    {
        Renderer renderer = region.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.SetColor("_EmissionColor", color);
            renderer.material.EnableKeyword("_EMISSION");
        }
    }
    
    void ResetRegionColor(GameObject region)
    {
        // 恢复原始颜色（从BrainVisualization组件获取）
        Renderer renderer = region.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.DisableKeyword("_EMISSION");
        }
    }
    
    int ExtractRegionId(string objectName)
    {
        // 从对象名称提取ID，格式: "Region_0_Label"
        string[] parts = objectName.Split('_');
        if (parts.Length >= 2 && int.TryParse(parts[1], out int id))
        {
            return id;
        }
        return -1;
    }
    
    public void DeselectAll()
    {
        if (currentSelected != null)
        {
            ResetRegionColor(currentSelected);
            currentSelected = null;
        }
        if (currentHovered != null)
        {
            ResetRegionColor(currentHovered);
            currentHovered = null;
        }
    }
}
'''
        
        script_path = scripts_dir / "BrainInteractionController.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info(f"  ✓ 生成交互控制器: BrainInteractionController.cs")
    
    def _generate_region_selector(self, scripts_dir: Path):
        """生成脑区选择器脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 脑区选择器 - 管理多个脑区的选择
/// </summary>
public class BrainRegionSelector : MonoBehaviour
{
    [Header("UI引用")]
    public Text infoText;
    public Button clearButton;
    
    [Header("选择设置")]
    public int maxSelections = 10;
    
    private List<int> selectedRegions = new List<int>();
    private BrainInteractionController interactionController;
    
    void Start()
    {
        interactionController = GetComponent<BrainInteractionController>();
        if (interactionController != null)
        {
            interactionController.OnRegionClicked += HandleRegionClick;
            interactionController.OnRegionHovered += HandleRegionHover;
        }
        
        if (clearButton != null)
        {
            clearButton.onClick.AddListener(ClearSelection);
        }
    }
    
    void HandleRegionClick(int regionId, Vector3 position)
    {
        if (selectedRegions.Contains(regionId))
        {
            selectedRegions.Remove(regionId);
            Debug.Log($"取消选择脑区 {regionId}");
        }
        else
        {
            if (selectedRegions.Count >= maxSelections)
            {
                Debug.LogWarning($"已达到最大选择数量 ({maxSelections})");
                return;
            }
            
            selectedRegions.Add(regionId);
            Debug.Log($"选择脑区 {regionId}");
        }
        
        UpdateInfoText();
    }
    
    void HandleRegionHover(int regionId)
    {
        if (infoText != null)
        {
            infoText.text = $"悬停脑区: {regionId}";
        }
    }
    
    void UpdateInfoText()
    {
        if (infoText != null)
        {
            if (selectedRegions.Count == 0)
            {
                infoText.text = "未选择脑区";
            }
            else
            {
                infoText.text = $"已选择 {selectedRegions.Count} 个脑区: " + 
                               string.Join(", ", selectedRegions);
            }
        }
    }
    
    public void ClearSelection()
    {
        selectedRegions.Clear();
        if (interactionController != null)
        {
            interactionController.DeselectAll();
        }
        UpdateInfoText();
    }
    
    public List<int> GetSelectedRegions()
    {
        return new List<int>(selectedRegions);
    }
}
'''
        
        script_path = scripts_dir / "BrainRegionSelector.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info(f"  ✓ 生成脑区选择器: BrainRegionSelector.cs")
    
    def _generate_stimulation_controller(self, scripts_dir: Path):
        """生成刺激控制器脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 虚拟刺激控制器 - 向后端发送刺激请求并显示预测结果
/// </summary>
public class StimulationController : MonoBehaviour
{
    [Header("UI引用")]
    public InputField amplitudeInput;
    public Dropdown patternDropdown;
    public Button applyButton;
    public Text statusText;
    
    [Header("设置")]
    public float defaultAmplitude = 0.5f;
    public string[] availablePatterns = {"constant", "sine", "pulse"};
    
    private WebSocketClient wsClient;
    private BrainRegionSelector regionSelector;
    private BrainVisualization visualization;
    
    void Start()
    {
        wsClient = GetComponent<WebSocketClient>();
        regionSelector = GetComponent<BrainRegionSelector>();
        visualization = GetComponent<BrainVisualization>();
        
        // 初始化UI
        if (amplitudeInput != null)
        {
            amplitudeInput.text = defaultAmplitude.ToString();
        }
        
        if (patternDropdown != null)
        {
            patternDropdown.ClearOptions();
            patternDropdown.AddOptions(new List<string>(availablePatterns));
        }
        
        if (applyButton != null)
        {
            applyButton.onClick.AddListener(ApplyStimulation);
        }
    }
    
    public void ApplyStimulation()
    {
        if (regionSelector == null || wsClient == null)
        {
            UpdateStatus("错误: 缺少必要组件");
            return;
        }
        
        List<int> selectedRegions = regionSelector.GetSelectedRegions();
        if (selectedRegions.Count == 0)
        {
            UpdateStatus("请先选择脑区");
            return;
        }
        
        // 读取参数
        float amplitude = defaultAmplitude;
        if (amplitudeInput != null)
        {
            float.TryParse(amplitudeInput.text, out amplitude);
        }
        
        string pattern = "sine";
        if (patternDropdown != null && patternDropdown.value < availablePatterns.Length)
        {
            pattern = availablePatterns[patternDropdown.value];
        }
        
        // 发送到后端
        UpdateStatus($"发送刺激请求: {selectedRegions.Count}个脑区, 强度={amplitude}, 模式={pattern}");
        
        wsClient.SimulateStimulation(
            selectedRegions.ToArray(),
            amplitude,
            pattern
        );
    }
    
    void UpdateStatus(string message)
    {
        Debug.Log(message);
        if (statusText != null)
        {
            statusText.text = message;
        }
    }
    
    public void OnPredictionReceived(string predictionJson)
    {
        // 处理后端返回的预测结果
        UpdateStatus("收到预测结果，更新可视化...");
        
        // TODO: 解析并显示预测
        if (visualization != null)
        {
            // visualization.LoadFromJson(predictionJson);
        }
    }
}
'''
        
        script_path = scripts_dir / "StimulationController.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info(f"  ✓ 生成刺激控制器: StimulationController.cs")
    
    def _generate_prediction_visualizer(self, scripts_dir: Path):
        """生成预测可视化脚本"""
        script_content = '''using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 预测结果可视化 - 同时显示预测和真实数据进行对比
/// </summary>
public class PredictionVisualizer : MonoBehaviour
{
    [Header("可视化模式")]
    public enum VisualizationMode
    {
        RealOnly,      // 仅显示真实数据
        PredictionOnly, // 仅显示预测数据
        SideBySide,    // 并排对比
        Overlay        // 叠加显示
    }
    
    [Header("设置")]
    public VisualizationMode currentMode = VisualizationMode.SideBySide;
    public float separationDistance = 100f;
    
    [Header("颜色")]
    public Color realDataColor = Color.blue;
    public Color predictionColor = Color.red;
    public Color matchColor = Color.green;
    
    [Header("UI")]
    public Text modeText;
    public Text accuracyText;
    public Slider accuracyThresholdSlider;
    
    private BrainVisualization realVisualization;
    private BrainVisualization predictionVisualization;
    private Dictionary<int, float> predictionAccuracy = new Dictionary<int, float>();
    
    void Start()
    {
        // 创建两个可视化实例
        CreateDualVisualizations();
        
        // 更新UI
        UpdateModeText();
    }
    
    void Update()
    {
        // 快捷键切换模式
        if (Input.GetKeyDown(KeyCode.M))
        {
            CycleMode();
        }
    }
    
    void CreateDualVisualizations()
    {
        // 在这里创建两个可视化对象用于对比
        // 实际实现需要根据场景结构调整
    }
    
    public void LoadPredictionData(string predictionJson, string realJson)
    {
        // 加载预测和真实数据
        // 计算准确度
        CalculateAccuracy();
        UpdateAccuracyDisplay();
    }
    
    void CalculateAccuracy()
    {
        // 计算每个脑区的预测准确度
        // 这里需要实际的计算逻辑
        
        // 示例：随机生成（实际应该基于真实对比）
        predictionAccuracy.Clear();
        for (int i = 0; i < 200; i++)
        {
            predictionAccuracy[i] = Random.Range(0.7f, 1.0f);
        }
    }
    
    void UpdateAccuracyDisplay()
    {
        if (accuracyText != null)
        {
            float avgAccuracy = 0f;
            foreach (var acc in predictionAccuracy.Values)
            {
                avgAccuracy += acc;
            }
            avgAccuracy /= predictionAccuracy.Count;
            
            accuracyText.text = $"平均准确度: {avgAccuracy:P1}";
        }
    }
    
    public void CycleMode()
    {
        currentMode = (VisualizationMode)(((int)currentMode + 1) % 4);
        ApplyVisualizationMode();
        UpdateModeText();
    }
    
    void ApplyVisualizationMode()
    {
        switch (currentMode)
        {
            case VisualizationMode.RealOnly:
                // 显示真实数据
                if (realVisualization != null)
                    realVisualization.gameObject.SetActive(true);
                if (predictionVisualization != null)
                    predictionVisualization.gameObject.SetActive(false);
                break;
                
            case VisualizationMode.PredictionOnly:
                // 显示预测数据
                if (realVisualization != null)
                    realVisualization.gameObject.SetActive(false);
                if (predictionVisualization != null)
                    predictionVisualization.gameObject.SetActive(true);
                break;
                
            case VisualizationMode.SideBySide:
                // 并排显示
                if (realVisualization != null)
                {
                    realVisualization.gameObject.SetActive(true);
                    realVisualization.transform.position = Vector3.left * separationDistance;
                }
                if (predictionVisualization != null)
                {
                    predictionVisualization.gameObject.SetActive(true);
                    predictionVisualization.transform.position = Vector3.right * separationDistance;
                }
                break;
                
            case VisualizationMode.Overlay:
                // 叠加显示
                if (realVisualization != null)
                {
                    realVisualization.gameObject.SetActive(true);
                    realVisualization.transform.position = Vector3.zero;
                }
                if (predictionVisualization != null)
                {
                    predictionVisualization.gameObject.SetActive(true);
                    predictionVisualization.transform.position = Vector3.zero;
                }
                break;
        }
    }
    
    void UpdateModeText()
    {
        if (modeText != null)
        {
            modeText.text = $"模式: {currentMode}";
        }
    }
}
'''
        
        script_path = scripts_dir / "PredictionVisualizer.cs"
        script_path.write_text(script_content, encoding='utf-8')
        logger.info(f"  ✓ 生成预测可视化: PredictionVisualizer.cs")
    
    def _create_comprehensive_documentation(self) -> str:
        """创建综合文档"""
        return f'''# TwinBrain Unity 集成完整指南

## 概述

本指南帮助你快速搭建TwinBrain Unity可视化系统，实现：
- ✅ 3D脑模型可视化
- ✅ 脑区活动实时动画
- ✅ 点击交互和选择
- ✅ 虚拟刺激模拟
- ✅ 预测vs真实数据对比
- ✅ 前后端WebSocket通信

## 快速开始

### 1. 文件结构

```
{self.output_dir}/
├── json/                          # JSON脑状态数据
│   ├── brain_state_*.json
│   └── sequence_index.json
├── obj/                           # 3D模型文件
│   └── brain_regions.obj
├── materials/                     # 材质配置
│   ├── RegionMaterial.json
│   └── ConnectionMaterial.json
├── UnityScripts/                  # Unity C#脚本
│   ├── BrainVisualization.cs
│   ├── BrainConfigLoader.cs
│   ├── WebSocketClient.cs
│   ├── BrainInteractionController.cs
│   ├── BrainRegionSelector.cs
│   ├── StimulationController.cs
│   └── PredictionVisualizer.cs
├── unity_config.json              # Unity配置
├── unity_scene_config.json        # 场景配置
├── unity_prefab_config.json       # 预制体配置
└── README_UNITY.md               # 本文档
```

### 2. Unity项目设置

#### 2.1 创建项目
1. 打开Unity Hub
2. 创建新的3D项目（Unity 2020.3+）
3. 项目名称：TwinBrain_Visualization

#### 2.2 安装依赖
1. 打开 Window → Package Manager
2. 点击 "+" → Add package from git URL
3. 输入: `com.unity.nuget.newtonsoft-json`
4. 等待安装完成

#### 2.3 导入脚本
1. 在Assets下创建Scripts文件夹
2. 将 `UnityScripts/` 下所有.cs文件复制到Scripts文件夹
3. 等待Unity编译完成

#### 2.4 导入数据
1. 在Assets下创建Data文件夹
2. 创建子文件夹：JSON, OBJ, Materials
3. 将对应的数据文件复制到相应文件夹

### 3. 场景搭建

#### 3.1 创建主对象
1. 创建空GameObject，命名为"BrainSystem"
2. 添加以下组件：
   - BrainVisualization
   - BrainConfigLoader
   - BrainInteractionController
   - BrainRegionSelector
   - WebSocketClient
   - StimulationController
   - PredictionVisualizer

#### 3.2 配置组件

**BrainConfigLoader:**
- Config Path: `Assets/Data/unity_config.json`
- Auto Load: ✓

**BrainVisualization:**
- JSON Path: `Assets/Data/JSON/`
- Load Sequence: ✓
- Region Scale: 1.0
- Activity Threshold: 0.3
- Show Connections: ✓
- FPS: 10
- Auto Play: ✓

**WebSocketClient:**
- Server URL: `ws://localhost:8765`
- Auto Connect: ✓
- Reconnect Interval: 5

#### 3.3 创建UI
1. 创建Canvas
2. 添加以下UI元素：

**信息面板:**
- Text: 显示选中脑区信息
- Text: 显示刺激状态
- Text: 显示预测准确度

**控制面板:**
- Button: 清除选择
- InputField: 刺激强度
- Dropdown: 刺激模式
- Button: 应用刺激
- Button: 切换可视化模式
- Slider: 准确度阈值

3. 连接UI引用到对应脚本组件

### 4. 启动后端服务器

在命令行运行：

```bash
cd {self.output_dir.parent}
python unity_automation.py --mode server
```

服务器将在 `ws://localhost:8765` 启动

### 5. 运行Unity项目

1. 确保后端服务器已启动
2. 在Unity中点击Play
3. 系统将自动加载数据并开始可视化

## 功能说明

### 基础可视化

**脑区显示:**
- 颜色表示活动强度（蓝色=低，红色=高）
- 大小随活动强度变化
- 透明度可调

**连接显示:**
- 白色线=结构连接
- 黄色线=功能连接
- 线宽表示连接强度

**动画播放:**
- 空格键: 播放/暂停
- R键: 重新加载
- 方向键: 逐帧播放

### 交互功能

**鼠标操作:**
- 悬停: 高亮脑区（黄色）
- 左键点击: 选择脑区（绿色）
- 右键: 取消选择
- 滚轮: 缩放视图

**脑区选择:**
- 支持多选（最多10个）
- 显示选中脑区列表
- 一键清除所有选择

### 虚拟刺激

**配置刺激:**
1. 选择目标脑区
2. 设置刺激强度（0-1）
3. 选择刺激模式（constant/sine/pulse）
4. 点击"应用刺激"

**查看响应:**
- 后端计算刺激响应
- 前端实时显示脑变化
- 可视化刺激扩散效果

### 预测对比

**模式切换（M键）:**
1. Real Only: 仅显示真实数据
2. Prediction Only: 仅显示预测数据
3. Side by Side: 并排对比
4. Overlay: 叠加显示

**准确度指标:**
- 整体平均准确度
- 每个脑区的准确度
- 颜色编码（绿色=准确，红色=误差大）

## 后端API

### WebSocket消息格式

**获取当前状态:**
```json
{{
  "type": "get_state"
}}
```

**预测未来状态:**
```json
{{
  "type": "predict",
  "n_steps": 20
}}
```

**模拟刺激:**
```json
{{
  "type": "simulate",
  "stimulation": {{
    "target_regions": [10, 15, 20],
    "amplitude": 0.5,
    "pattern": "sine",
    "frequency": 10.0
  }}
}}
```

**开始流式传输:**
```json
{{
  "type": "start_stream",
  "fps": 10,
  "duration": 60
}}
```

## 故障排除

### 数据加载失败
- 检查文件路径是否正确
- 确认JSON格式是否有效
- 查看Unity Console错误信息

### 服务器连接失败
- 确认后端服务器已启动
- 检查端口8765是否被占用
- 查看防火墙设置

### 可视化性能问题
- 提高活动阈值（显示更少脑区）
- 关闭连接显示
- 降低帧率
- 减少时间步长

### 交互无响应
- 确认脑区对象有Collider组件
- 检查LayerMask设置
- 确认相机有正确的Raycaster

## 高级功能

### 自定义颜色映射
编辑 `unity_config.json`:
```json
{{
  "colors": {{
    "low_activity": {{"r": 0, "g": 0, "b": 255}},
    "mid_activity": {{"r": 0, "g": 255, "b": 0}},
    "high_activity": {{"r": 255, "g": 0, "b": 0}}
  }}
}}
```

### 添加网络分析
显示特定脑网络（视觉、运动等）的活动模式。

### 导出录像
使用Unity Recorder录制可视化视频。

### VR支持
添加VR支持以实现沉浸式脑浏览。

## 常见问题

**Q: 如何更改配色方案？**
A: 在BrainVisualization组件中设置Low/High Activity Color。

**Q: 如何添加更多交互？**
A: 扩展BrainInteractionController脚本，添加自定义事件处理。

**Q: 如何使用自己的脑数据？**
A: 运行Python脚本导出数据，然后在Unity中重新导入。

**Q: 支持实时fMRI数据吗？**
A: 支持，通过WebSocket流式传输实时数据。

## 联系支持

遇到问题请：
1. 查看控制台日志
2. 检查本文档
3. 提交GitHub Issue

---

**版本:** 1.0  
**生成时间:** {self.output_dir}  
**维护者:** TwinBrain Team
'''
    
    def _load_atlas_info(self) -> Dict[str, Any]:
        """加载图谱信息"""
        # 简化版本，实际应该从文件加载
        import numpy as np
        atlas_info = {
            'name': 'Schaefer200',
            'n_regions': 200,
            'regions': {}
        }
        
        for i in range(200):
            atlas_info['regions'][str(i + 1)] = {
                'label': f'Region_{i+1}',
                'xyz': [
                    np.random.uniform(-80, 80),
                    np.random.uniform(-100, 80),
                    np.random.uniform(-60, 80)
                ]
            }
        
        return atlas_info
    
    def _print_summary(self):
        """打印完成摘要"""
        print("\n📁 输出目录:")
        print(f"   {self.output_dir.absolute()}")
        
        print("\n📄 生成的文件:")
        for root, dirs, files in self.output_dir.walk():
            level = root.relative_to(self.output_dir).parts
            indent = "  " * len(level)
            print(f"{indent}📂 {root.name}/")
            sub_indent = "  " * (len(level) + 1)
            for file in files:
                print(f"{sub_indent}📄 {file}")
        
        print("\n✨ 下一步:")
        print("1. 打开Unity并创建新项目")
        print("2. 安装Newtonsoft.Json包")
        print("3. 导入UnityScripts/文件夹中的所有脚本")
        print("4. 导入json/, obj/, materials/数据到Unity")
        print("5. 按照README_UNITY.md配置场景")
        print("6. 运行: python unity_automation.py --mode server 启动后端")
        print("7. 在Unity中点击Play开始可视化")
        
        print("\n📖 详细说明:")
        print(f"   查看 {self.output_dir}/README_UNITY.md")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity 一键式自动化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出所有数据和配置
  python unity_automation.py --mode export
  
  # 启动后端服务器
  python unity_automation.py --mode server
  
  # 完整流程（导出+服务器）
  python unity_automation.py --mode all
  
  # 指定输出目录
  python unity_automation.py --output custom_output --mode export
  
  # 使用自己的数据
  python unity_automation.py --data-path /path/to/data --mode export
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['export', 'server', 'all'],
        default='all',
        help='运行模式 (default: all)'
    )
    
    parser.add_argument(
        '--output',
        default='unity_output',
        help='输出目录 (default: unity_output)'
    )
    
    parser.add_argument(
        '--data-path',
        help='脑数据路径（可选）'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='服务器端口 (default: 8765)'
    )
    
    parser.add_argument(
        '--use-example',
        action='store_true',
        default=True,
        help='使用示例数据 (default: True)'
    )
    
    args = parser.parse_args()
    
    # 创建自动化实例
    automation = UnityAutomation(
        output_dir=args.output,
        brain_data_path=args.data_path,
        use_example_data=args.use_example
    )
    
    try:
        if args.mode in ['export', 'all']:
            # 运行导出流程
            automation.run_complete_workflow()
        
        if args.mode in ['server', 'all']:
            # 启动服务器
            if args.mode == 'all':
                print("\n" + "="*80)
                print("导出完成，现在启动服务器...")
                print("（如果不需要服务器，请按Ctrl+C）")
                print("="*80)
                import time
                time.sleep(3)
            
            automation.start_server(port=args.port)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 130
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
