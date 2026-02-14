using UnityEngine;
using UnityEngine.UI;
using Newtonsoft.Json.Linq;

namespace TwinBrain
{
    /// <summary>
    /// TwinBrain 虚拟刺激输入控制器
    /// Unity 2019+ 兼容版本
    /// 
    /// 提供用户界面用于输入虚拟刺激参数并发送到后端模型
    /// 
    /// 功能:
    /// - 选择目标脑区（通过点击或手动输入）
    /// - 设置刺激参数（振幅、模式、持续时间）
    /// - 发送刺激请求到后端
    /// - 显示刺激结果
    /// </summary>
    [RequireComponent(typeof(WebSocketClient))]
    public class StimulationInput : MonoBehaviour
    {
        [Header("UI References")]
        [Tooltip("Input field for target region IDs (comma separated)")]
        public InputField targetRegionsInput;
        
        [Tooltip("Slider for stimulation amplitude")]
        public Slider amplitudeSlider;
        
        [Tooltip("Text showing current amplitude value")]
        public Text amplitudeText;
        
        [Tooltip("Dropdown for stimulation pattern")]
        public Dropdown patternDropdown;
        
        [Tooltip("Button to send stimulation")]
        public Button sendButton;
        
        [Tooltip("Text for status messages")]
        public Text statusText;
        
        [Header("Default Values")]
        [Tooltip("Default stimulation amplitude")]
        [Range(0f, 5f)]
        public float defaultAmplitude = 1.0f;
        
        [Tooltip("Available stimulation patterns")]
        public string[] patterns = new string[] { "constant", "sine", "pulse", "ramp" };
        
        [Header("References")]
        [Tooltip("BrainVisualization component")]
        public BrainVisualization brainVis;
        
        private WebSocketClient wsClient;
        private int[] selectedRegions = new int[0];
        private float currentAmplitude;
        private string currentPattern = "constant";
        
        void Start()
        {
            wsClient = GetComponent<WebSocketClient>();
            
            // Initialize UI
            if (amplitudeSlider != null)
            {
                amplitudeSlider.value = defaultAmplitude;
                amplitudeSlider.onValueChanged.AddListener(OnAmplitudeChanged);
            }
            
            if (patternDropdown != null)
            {
                patternDropdown.ClearOptions();
                patternDropdown.AddOptions(new System.Collections.Generic.List<string>(patterns));
                patternDropdown.onValueChanged.AddListener(OnPatternChanged);
            }
            
            if (sendButton != null)
            {
                sendButton.onClick.AddListener(OnSendClicked);
            }
            
            if (brainVis != null)
            {
                brainVis.OnRegionClicked += OnRegionClicked;
            }
            
            currentAmplitude = defaultAmplitude;
            UpdateAmplitudeText();
            UpdateStatusText("Ready");
        }
        
        void OnDestroy()
        {
            if (brainVis != null)
            {
                brainVis.OnRegionClicked -= OnRegionClicked;
            }
        }
        
        /// <summary>
        /// 处理脑区点击事件
        /// </summary>
        void OnRegionClicked(int regionId, RegionData regionData)
        {
            // Add to selected regions
            System.Collections.Generic.List<int> regionsList = new System.Collections.Generic.List<int>(selectedRegions);
            
            if (!regionsList.Contains(regionId))
            {
                regionsList.Add(regionId);
                selectedRegions = regionsList.ToArray();
                UpdateTargetRegionsInput();
                UpdateStatusText(string.Format("Selected region {0}: {1}", regionId, regionData.label));
            }
        }
        
        /// <summary>
        /// 更新目标脑区输入框
        /// </summary>
        void UpdateTargetRegionsInput()
        {
            if (targetRegionsInput != null && selectedRegions.Length > 0)
            {
                string[] ids = new string[selectedRegions.Length];
                for (int i = 0; i < selectedRegions.Length; i++)
                {
                    ids[i] = selectedRegions[i].ToString();
                }
                targetRegionsInput.text = string.Join(", ", ids);
            }
        }
        
        /// <summary>
        /// 振幅滑块值改变
        /// </summary>
        void OnAmplitudeChanged(float value)
        {
            currentAmplitude = value;
            UpdateAmplitudeText();
        }
        
        /// <summary>
        /// 更新振幅显示文本
        /// </summary>
        void UpdateAmplitudeText()
        {
            if (amplitudeText != null)
            {
                amplitudeText.text = string.Format("Amplitude: {0:F2}", currentAmplitude);
            }
        }
        
        /// <summary>
        /// 模式下拉框值改变
        /// </summary>
        void OnPatternChanged(int index)
        {
            if (index >= 0 && index < patterns.Length)
            {
                currentPattern = patterns[index];
                UpdateStatusText(string.Format("Pattern: {0}", currentPattern));
            }
        }
        
        /// <summary>
        /// 发送按钮点击
        /// </summary>
        void OnSendClicked()
        {
            ParseTargetRegions();
            
            if (selectedRegions.Length == 0)
            {
                UpdateStatusText("Error: No target regions selected");
                return;
            }
            
            if (!wsClient.isConnected)
            {
                UpdateStatusText("Error: Not connected to server");
                return;
            }
            
            SendStimulation();
        }
        
        /// <summary>
        /// 解析目标脑区输入
        /// </summary>
        void ParseTargetRegions()
        {
            if (targetRegionsInput == null || string.IsNullOrEmpty(targetRegionsInput.text))
            {
                return;
            }
            
            try
            {
                string[] parts = targetRegionsInput.text.Split(',');
                System.Collections.Generic.List<int> regions = new System.Collections.Generic.List<int>();
                
                foreach (string part in parts)
                {
                    string trimmed = part.Trim();
                    if (!string.IsNullOrEmpty(trimmed))
                    {
                        int regionId;
                        if (int.TryParse(trimmed, out regionId))
                        {
                            regions.Add(regionId);
                        }
                    }
                }
                
                selectedRegions = regions.ToArray();
            }
            catch (System.Exception e)
            {
                Debug.LogError(string.Format("Failed to parse target regions: {0}", e.Message));
                UpdateStatusText("Error: Invalid region format");
            }
        }
        
        /// <summary>
        /// 发送刺激请求到后端
        /// </summary>
        void SendStimulation()
        {
            UpdateStatusText(string.Format("Sending stimulation to {0} regions...", selectedRegions.Length));
            
            wsClient.SimulateStimulation(selectedRegions, currentAmplitude, currentPattern);
            
            UpdateStatusText(string.Format("Stimulation sent: {0} regions, amplitude={1:F2}, pattern={2}", 
                selectedRegions.Length, currentAmplitude, currentPattern));
        }
        
        /// <summary>
        /// 更新状态文本
        /// </summary>
        void UpdateStatusText(string message)
        {
            if (statusText != null)
            {
                statusText.text = message;
            }
            Debug.Log(string.Format("[StimulationInput] {0}", message));
        }
        
        /// <summary>
        /// 清除选中的脑区
        /// </summary>
        public void ClearSelectedRegions()
        {
            selectedRegions = new int[0];
            if (targetRegionsInput != null)
            {
                targetRegionsInput.text = "";
            }
            UpdateStatusText("Cleared selected regions");
        }
        
        /// <summary>
        /// 设置振幅值
        /// </summary>
        public void SetAmplitude(float amplitude)
        {
            currentAmplitude = Mathf.Clamp(amplitude, 0f, 5f);
            if (amplitudeSlider != null)
            {
                amplitudeSlider.value = currentAmplitude;
            }
            UpdateAmplitudeText();
        }
        
        /// <summary>
        /// 设置刺激模式
        /// </summary>
        public void SetPattern(string pattern)
        {
            for (int i = 0; i < patterns.Length; i++)
            {
                if (patterns[i] == pattern)
                {
                    currentPattern = pattern;
                    if (patternDropdown != null)
                    {
                        patternDropdown.value = i;
                    }
                    break;
                }
            }
        }
    }
}
