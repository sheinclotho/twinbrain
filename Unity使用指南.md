# TwinBrain Unity 使用指南

## 简介

本指南提供最简单、最实用的Unity可视化使用方法。遵循本指南，您可以快速搭建大脑3D可视化。

## 前提条件

- Python 3.8+
- Unity 2019.1+ (推荐2019/2020 LTS)
- （可选）FreeSurfer表面数据

## 快速开始（三步走）

### 第一步：初始化项目

```bash
# 克隆仓库
git clone https://github.com/sheinclotho/twinbrain.git
cd twinbrain

# 安装依赖
pip install -r requirements.txt

# 使用FreeSurfer数据创建项目（推荐）
python setup_unity_project.py --freesurfer-dir /path/to/freesurfer_files

# 或创建基础项目结构（不使用FreeSurfer）
python setup_unity_project.py --auto-setup
```

**生成的文件**:
```
unity_project/
├── freesurfer_files/      # FreeSurfer数据
├── brain_data/           
│   ├── original/          # 原始数据放这里
│   ├── cache/             # 预处理缓存放这里
│   └── model_output/      # JSON状态文件（Unity读取）
├── Unity_Assets/
│   ├── Scripts/           # Unity C#脚本
│   └── Models/            # OBJ 3D模型（如果使用FreeSurfer）
└── unity_config.json      # 配置文件
```

### 第二步：准备数据

#### 方式A：直接使用JSON文件（最简单）

如果你已经有处理好的JSON数据，直接复制到输出目录：

```bash
# 复制JSON文件到model_output目录
cp your_brain_state.json unity_project/brain_data/model_output/
```

**JSON文件格式示例** (最简版本):
```json
{
  "timestamp": "2024-02-13T12:00:00",
  "brain_state": {
    "regions": [
      {
        "id": 1,
        "position": {"x": -10.5, "y": 20.3, "z": 30.1},
        "activity": {"fmri": {"amplitude": 0.75}}
      },
      {
        "id": 2,
        "position": {"x": -12.1, "y": 22.5, "z": 31.8},
        "activity": {"fmri": {"amplitude": 0.45}}
      }
    ]
  }
}
```

**字段说明**:
- `id`: 脑区ID，从1开始
- `position`: 脑区3D坐标（单位：毫米）
- `activity.fmri.amplitude`: 活动强度，范围0-1

#### 方式B：从预处理数据生成JSON

如果你有预处理的数据文件（.pkl, .npy格式）：

```bash
# 将数据文件放入cache目录
cp preprocessed_data.pkl unity_project/brain_data/cache/

# 使用brain_state_exporter生成JSON
python -c "
from unity_integration import BrainStateExporter
import numpy as np

# 加载数据
data = np.load('unity_project/brain_data/cache/preprocessed_data.pkl', allow_pickle=True)

# 导出为JSON
exporter = BrainStateExporter()
exporter.export_from_array(
    fmri_data=data['fmri'],  # shape: (n_regions, n_timepoints, n_features)
    output_dir='unity_project/brain_data/model_output',
    region_positions=data.get('positions')  # 可选
)
"
```

#### 方式C：从原始fMRI数据生成JSON

```bash
# 1. 将原始数据放入original目录
cp your_fmri.nii unity_project/brain_data/original/

# 2. 使用主项目的预处理工具（需要先训练模型）
python main.py --mode preprocess --data-dir unity_project/brain_data/original

# 3. 导出JSON
python -m unity_integration.brain_state_exporter \
    --data-dir unity_project/brain_data/cache \
    --output unity_project/brain_data/model_output
```

### 第三步：在Unity中设置

#### 1. 创建Unity项目

1. 打开Unity Hub
2. 新建3D项目
3. 命名（例如：TwinBrain_Viz）

#### 2. 导入资源

**复制脚本**:
```bash
cp unity_project/Unity_Assets/Scripts/* <Unity项目路径>/Assets/Scripts/
```

**复制模型**（如果有OBJ文件）:
```bash
cp unity_project/Unity_Assets/Models/* <Unity项目路径>/Assets/Models/
```

**复制数据**:
```bash
# 在Unity项目中创建StreamingAssets目录
mkdir <Unity项目路径>/Assets/StreamingAssets

# 复制JSON数据
cp -r unity_project/brain_data/model_output <Unity项目路径>/Assets/StreamingAssets/brain_states

# 复制配置文件
cp unity_project/unity_config.json <Unity项目路径>/Assets/StreamingAssets/
```

#### 3. 安装JSON解析包

在Unity中：
1. 打开 `Window > Package Manager`
2. 点击 `+` > `Add package from git URL`
3. 输入: `com.unity.nuget.newtonsoft-json`
4. 点击 `Add`

#### 4. 设置场景

**创建管理器对象**:
1. 在Hierarchy中，右键 > `Create Empty`
2. 命名为 `BrainManager`
3. 在Inspector中，点击 `Add Component`
4. 添加 `BrainDataLoader` 脚本

**配置BrainDataLoader**:
- **Json Directory**: `StreamingAssets/brain_states`
- **Config Path**: `StreamingAssets/unity_config.json`
- **Auto Load On Start**: ✓（勾选）

**创建脑区预制体**（使用简单球体）:
1. Hierarchy > 3D Object > Sphere
2. 缩放到合适大小（Scale: 0.5, 0.5, 0.5）
3. 拖到Project窗口创建Prefab，命名为 `BrainRegion`
4. 在BrainDataLoader中，将此Prefab拖到 `Region Prefab` 字段
5. 删除Hierarchy中的原始Sphere

#### 5. 运行

点击Unity的 `Play` 按钮，应该可以看到根据JSON数据渲染的大脑活动可视化。

## 简单实用的JSON加载方法

### 方法1：使用Python直接生成（推荐）

创建一个简单的脚本 `generate_json.py`:

```python
#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

def generate_brain_state_json(output_path, n_regions=200):
    """
    生成简单的大脑状态JSON文件
    
    Args:
        output_path: 输出文件路径
        n_regions: 脑区数量
    """
    # 生成随机位置和活动值（实际使用时替换为真实数据）
    regions = []
    for i in range(1, n_regions + 1):
        region = {
            "id": i,
            "label": f"Region_{i}",
            "position": {
                "x": float(np.random.uniform(-50, 50)),
                "y": float(np.random.uniform(-50, 50)),
                "z": float(np.random.uniform(-50, 50))
            },
            "activity": {
                "fmri": {
                    "amplitude": float(np.random.uniform(0, 1))
                }
            }
        }
        regions.append(region)
    
    # 创建完整JSON结构
    brain_state = {
        "timestamp": "2024-02-13T12:00:00",
        "metadata": {
            "atlas": "Schaefer200",
            "n_regions": n_regions
        },
        "brain_state": {
            "regions": regions
        }
    }
    
    # 保存文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(brain_state, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON文件已生成: {output_path}")
    print(f"  包含 {n_regions} 个脑区")

# 使用示例
if __name__ == "__main__":
    generate_brain_state_json(
        output_path="unity_project/brain_data/model_output/brain_state_001.json",
        n_regions=200
    )
```

运行：
```bash
python generate_json.py
```

### 方法2：从numpy数组加载

如果你的数据是numpy数组：

```python
import json
import numpy as np

# 加载你的数据
fmri_data = np.load('your_data.npy')  # shape: (n_regions, n_timepoints, n_features)
positions = np.load('positions.npy')   # shape: (n_regions, 3)

# 生成JSON（单个时间点）
timepoint = 0
regions = []
for i in range(len(fmri_data)):
    regions.append({
        "id": i + 1,
        "position": {
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "z": float(positions[i, 2])
        },
        "activity": {
            "fmri": {
                "amplitude": float(fmri_data[i, timepoint, 0])
            }
        }
    })

brain_state = {
    "timestamp": f"2024-02-13T12:00:{timepoint:02d}",
    "brain_state": {"regions": regions}
}

# 保存
with open(f'brain_state_{timepoint:03d}.json', 'w') as f:
    json.dump(brain_state, f, indent=2)
```

### 方法3：批量生成时间序列

```python
def generate_time_series(fmri_data, positions, output_dir):
    """
    批量生成时间序列JSON文件
    
    Args:
        fmri_data: numpy array, shape (n_regions, n_timepoints, n_features)
        positions: numpy array, shape (n_regions, 3)
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_regions, n_timepoints, _ = fmri_data.shape
    
    for t in range(n_timepoints):
        regions = []
        for i in range(n_regions):
            regions.append({
                "id": i + 1,
                "position": {
                    "x": float(positions[i, 0]),
                    "y": float(positions[i, 1]),
                    "z": float(positions[i, 2])
                },
                "activity": {
                    "fmri": {"amplitude": float(fmri_data[i, t, 0])}
                }
            })
        
        brain_state = {
            "timestamp": f"2024-02-13T12:00:{t:02d}",
            "brain_state": {"regions": regions}
        }
        
        output_file = output_dir / f"brain_state_t{t:04d}.json"
        with open(output_file, 'w') as f:
            json.dump(brain_state, f)
        
        if t % 10 == 0:
            print(f"已生成 {t}/{n_timepoints} 个文件")
    
    print(f"✓ 完成！共生成 {n_timepoints} 个JSON文件")
```

## 常见问题

### Q: 没有FreeSurfer数据怎么办？

A: 不影响！可以使用简单的球体（Sphere）作为脑区显示，只需要JSON数据即可。

### Q: 如何自定义脑区位置？

A: 在JSON的`position`字段中直接指定3D坐标：
```json
"position": {"x": -10.5, "y": 20.3, "z": 30.1}
```

### Q: 数据范围应该是多少？

A: 
- `activity.amplitude`: 建议归一化到0-1范围
- `position`: 单位为毫米，典型范围 -100到100

### Q: 如何显示时间序列动画？

A: 生成多个JSON文件，命名为 `brain_state_t0000.json`, `brain_state_t0001.json` 等，Unity会自动按序列播放。

### Q: 性能慢怎么办？

A: 
1. 减少脑区数量（从200减到100或50）
2. 降低JSON文件更新频率
3. 使用Unity的LOD系统
4. 只显示活动值高于阈值的脑区

### Q: 可以不用OBJ模型吗？

A: 可以！直接使用Unity的Sphere或Cube primitive即可，简单稳定。

## 配置选项

编辑 `unity_config.json` 调整可视化参数：

```json
{
  "visualization": {
    "region_scale": 1.0,        // 脑区大小，越大越明显
    "activity_threshold": 0.2,  // 活动阈值，低于此值不显示
    "fps": 10,                  // 动画帧率
    "use_obj_models": false     // 是否使用OBJ模型（false使用简单球体）
  }
}
```

## 下一步

- 查看 `Unity架构说明.md` 了解技术细节
- 查看 `unity_integration/` 目录了解Python模块
- 查看 `setup_unity_project.py` 了解项目生成过程

---

**提示**: Unity部分设计简单稳定，完成设置后基本不需要修改。如有问题，优先检查JSON文件格式是否正确。
