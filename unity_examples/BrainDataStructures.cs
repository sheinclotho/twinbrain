using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// TwinBrain数据结构定义
/// 
/// 这个文件包含了TwinBrain JSON数据的所有数据类定义。
/// 被BrainVisualization和WebSocketClient共享使用，避免重复定义。
/// </summary>
namespace TwinBrain
{
    // 数据类匹配TwinBrain JSON格式
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

    // Unity配置数据结构
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
}
