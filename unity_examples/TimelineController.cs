using UnityEngine;
using UnityEngine.UI;

namespace TwinBrain
{
    /// <summary>
    /// TwinBrain 时间序列控制器
    /// Unity 2019+ 兼容版本
    /// 
    /// 提供进度条和播放控制，用于时间序列可视化
    /// 
    /// 功能:
    /// - 进度条滑块，可拖动到任意时间点
    /// - 播放/暂停按钮
    /// - 显示当前时间点和总帧数
    /// - 与BrainVisualization组件集成
    /// 
    /// 使用方法:
    /// 1. 在Canvas下创建UI元素（Slider, Button, Text）
    /// 2. 将此脚本添加到Canvas或子对象
    /// 3. 配置UI引用和BrainVisualization引用
    /// 
    /// 注意: BrainVisualization组件可以在同一GameObject或其他GameObject上
    /// </summary>
    public class TimelineController : MonoBehaviour
    {
        [Header("UI References")]
        [Tooltip("Slider for timeline scrubbing")]
        public Slider timelineSlider;
        
        [Tooltip("Button to play/pause animation")]
        public Button playPauseButton;
        
        [Tooltip("Text showing current frame")]
        public Text currentFrameText;
        
        [Tooltip("Text showing total frames")]
        public Text totalFramesText;
        
        [Tooltip("Text on play/pause button")]
        public Text playPauseButtonText;
        
        [Header("References")]
        [Tooltip("BrainVisualization component (can be on this or another GameObject)")]
        public BrainVisualization brainVisualization;
        [Header("Settings")]
        [Tooltip("Update UI every N frames")]
        public int updateInterval = 1;
        
        private BrainVisualization brainVis;
        private bool isPlaying = false;
        private int lastFrame = -1;
        
        void Start()
        {
            // Get BrainVisualization from assigned reference or same GameObject
            brainVis = brainVisualization != null ? brainVisualization : GetComponent<BrainVisualization>();
            
            if (brainVis == null)
            {
                Debug.LogError("TimelineController requires BrainVisualization component!");
                enabled = false;
                return;
            }
            
            // Setup slider
            if (timelineSlider != null)
            {
                timelineSlider.onValueChanged.AddListener(OnSliderChanged);
            }
            
            // Setup play/pause button
            if (playPauseButton != null)
            {
                playPauseButton.onClick.AddListener(OnPlayPauseClicked);
            }
            
            // Initialize UI
            UpdateUI();
        }
        
        void Update()
        {
            // Update UI periodically
            int currentFrame = brainVis.GetCurrentFrame();
            if (currentFrame != lastFrame)
            {
                if ((currentFrame % updateInterval) == 0)
                {
                    UpdateUI();
                }
                lastFrame = currentFrame;
            }
        }
        
        /// <summary>
        /// 滑块值改变事件
        /// </summary>
        void OnSliderChanged(float value)
        {
            if (brainVis == null || timelineSlider == null)
            {
                return;
            }
            
            int totalFrames = brainVis.GetTotalFrames();
            if (totalFrames == 0)
            {
                return;
            }
            
            // Convert slider value (0-1) to frame index
            int targetFrame = Mathf.RoundToInt(value * (totalFrames - 1));
            
            // Only update if different from current frame
            if (targetFrame != brainVis.GetCurrentFrame())
            {
                brainVis.SetFrame(targetFrame);
                UpdateFrameText();
            }
        }
        
        /// <summary>
        /// 播放/暂停按钮点击事件
        /// </summary>
        void OnPlayPauseClicked()
        {
            if (brainVis == null)
            {
                return;
            }
            
            // Toggle play/pause
            if (isPlaying)
            {
                brainVis.Pause();
                isPlaying = false;
            }
            else
            {
                brainVis.Play();
                isPlaying = true;
            }
            
            UpdatePlayPauseButtonText();
        }
        
        /// <summary>
        /// 更新所有UI元素
        /// </summary>
        void UpdateUI()
        {
            UpdateSlider();
            UpdateFrameText();
            UpdatePlayPauseButtonText();
        }
        
        /// <summary>
        /// 更新滑块位置
        /// </summary>
        void UpdateSlider()
        {
            if (timelineSlider == null || brainVis == null)
            {
                return;
            }
            
            int totalFrames = brainVis.GetTotalFrames();
            if (totalFrames == 0)
            {
                timelineSlider.value = 0;
                return;
            }
            
            int currentFrame = brainVis.GetCurrentFrame();
            float sliderValue = (float)currentFrame / (totalFrames - 1);
            
            // Update slider without triggering onValueChanged
            timelineSlider.SetValueWithoutNotify(sliderValue);
        }
        
        /// <summary>
        /// 更新帧数显示
        /// </summary>
        void UpdateFrameText()
        {
            if (brainVis == null)
            {
                return;
            }
            
            int currentFrame = brainVis.GetCurrentFrame();
            int totalFrames = brainVis.GetTotalFrames();
            
            if (currentFrameText != null)
            {
                currentFrameText.text = string.Format("Frame: {0}", currentFrame + 1);
            }
            
            if (totalFramesText != null)
            {
                totalFramesText.text = string.Format("/ {0}", totalFrames);
            }
        }
        
        /// <summary>
        /// 更新播放/暂停按钮文本
        /// </summary>
        void UpdatePlayPauseButtonText()
        {
            if (playPauseButtonText != null)
            {
                playPauseButtonText.text = isPlaying ? "Pause" : "Play";
            }
        }
        
        /// <summary>
        /// 跳到第一帧
        /// </summary>
        public void GoToFirstFrame()
        {
            if (brainVis != null)
            {
                brainVis.SetFrame(0);
                UpdateUI();
            }
        }
        
        /// <summary>
        /// 跳到最后一帧
        /// </summary>
        public void GoToLastFrame()
        {
            if (brainVis != null)
            {
                int totalFrames = brainVis.GetTotalFrames();
                if (totalFrames > 0)
                {
                    brainVis.SetFrame(totalFrames - 1);
                    UpdateUI();
                }
            }
        }
        
        /// <summary>
        /// 前进一帧
        /// </summary>
        public void StepForward()
        {
            if (brainVis != null)
            {
                int currentFrame = brainVis.GetCurrentFrame();
                int totalFrames = brainVis.GetTotalFrames();
                if (currentFrame < totalFrames - 1)
                {
                    brainVis.SetFrame(currentFrame + 1);
                    UpdateUI();
                }
            }
        }
        
        /// <summary>
        /// 后退一帧
        /// </summary>
        public void StepBackward()
        {
            if (brainVis != null)
            {
                int currentFrame = brainVis.GetCurrentFrame();
                if (currentFrame > 0)
                {
                    brainVis.SetFrame(currentFrame - 1);
                    UpdateUI();
                }
            }
        }
    }
}
