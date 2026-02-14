using UnityEngine;
using System.Collections.Generic;

// TwinBrain数据结构定义
// Unity 2019+ 兼容版本
// 所有数据类匹配TwinBrain JSON格式

namespace TwinBrain
{
    [System.Serializable]
    public class BrainStateData
    {
        public string version;
        public string timestamp;
        public MetadataData metadata;
        public BrainState brain_state;
        public StimulationData stimulation;
    }

    [System.Serializable]
    public class MetadataData
    {
        public string subject;
        public string atlas;
        public string model_version;
        public int time_point;
        public float time_second;
    }

    [System.Serializable]
    public class BrainState
    {
        public int time_point;
        public float time_second;
        public List<RegionData> regions;
        public List<ConnectionData> connections;
        public GlobalMetrics global_metrics;
    }

    [System.Serializable]
    public class RegionData
    {
        public int id;
        public string label;
        public PositionData position;
        public ActivityData activity;
    }

    [System.Serializable]
    public class PositionData
    {
        public float x;
        public float y;
        public float z;
    }

    [System.Serializable]
    public class ActivityData
    {
        public FMRIActivity fmri;
        public EEGActivity eeg;
        
        // C# convention: Use PascalCase for properties
        // Note: JSON deserialization will still work with snake_case JSON fields
        public float predictionValue;
        public bool isPredicted;
    }

    [System.Serializable]
    public class FMRIActivity
    {
        public float amplitude;
        public float raw_value;
    }

    [System.Serializable]
    public class EEGActivity
    {
        public float amplitude;
        public float raw_value;
    }

    [System.Serializable]
    public class ConnectionData
    {
        public int source;
        public int target;
        public float strength;
        public string type;
    }

    [System.Serializable]
    public class GlobalMetrics
    {
        public float mean_activity;
        public float std_activity;
        public float max_activity;
        public int active_regions;
    }

    [System.Serializable]
    public class StimulationData
    {
        public bool active;
        public List<int> target_regions;
        public float amplitude;
        
        /// <summary>
        /// Stimulation pattern type
        /// Valid values: "constant", "sine", "pulse", "ramp"
        /// </summary>
        public string pattern;
    }

    [System.Serializable]
    public class SequenceIndex
    {
        public string subject;
        public int start;
        public int end;
        public int step;
        public int n_frames;
        public List<string> files;
    }

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
        public bool use_obj_models;
    }

    [System.Serializable]
    public class ColorSettings
    {
        public ColorRGB low_activity;
        public ColorRGB high_activity;
        public ColorRGB predicted_signal;
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

    [System.Serializable]
    public class PredictionRequest
    {
        public string type;
        public int n_steps;
        public List<float> current_state;
    }

    [System.Serializable]
    public class PredictionResponse
    {
        public string type;
        public bool success;
        public List<BrainStateData> predictions;
        public string message;
    }

    [System.Serializable]
    public class StimulationRequest
    {
        public string type;
        public StimulationData stimulation;
    }

    [System.Serializable]
    public class StimulationResponse
    {
        public string type;
        public bool success;
        public BrainStateData result;
        public string message;
    }
}
