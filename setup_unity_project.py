#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain Unity 一键式自动化设置脚本（简化版）
===========================================

完全自动化Unity项目设置，无需手动配置。

功能：
1. 自动创建Unity项目文件夹结构
2. 复制Unity C#脚本到项目
3. 生成示例大脑数据
4. 创建Unity配置文件
5. 提供详细的使用说明

使用方法：
    # 基本使用（自动生成示例数据）
    python setup_unity_project.py
    
    # 指定输出目录
    python setup_unity_project.py --output unity_project
    
    # 包含后端服务器设置
    python setup_unity_project.py --with-server

完成后：
    1. 在Unity中创建新项目
    2. 将生成的Scripts文件夹复制到Unity的Assets目录
    3. 将生成的JSON数据复制到Unity的StreamingAssets目录
    4. 按照生成的README_UNITY.md说明操作
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


class UnityProjectSetup:
    """Unity项目一键式设置"""
    
    def __init__(self, output_dir="Unity_TwinBrain", with_server=False):
        self.project_root = Path(__file__).parent
        self.output_dir = Path(output_dir)
        self.with_server = with_server
        
        # 定义输出目录结构
        self.scripts_dir = self.output_dir / "Scripts"
        self.data_dir = self.output_dir / "BrainData"
        self.json_dir = self.data_dir / "JSON"
        self.config_dir = self.data_dir / "Config"
        
    def setup(self):
        """执行完整设置流程"""
        print_header("TwinBrain Unity 一键式项目设置")
        
        try:
            # 步骤1: 创建目录结构
            self.create_directory_structure()
            
            # 步骤2: 复制Unity脚本
            self.copy_unity_scripts()
            
            # 步骤3: 生成示例数据
            self.generate_example_data()
            
            # 步骤4: 生成配置文件
            self.generate_config_files()
            
            # 步骤5: 生成使用说明
            self.generate_documentation()
            
            # 步骤6: 可选 - 生成服务器脚本
            if self.with_server:
                self.generate_server_scripts()
            
            # 完成
            self.print_completion_summary()
            
        except Exception as e:
            print_error(f"设置过程中出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def create_directory_structure(self):
        """创建目录结构"""
        print_info("步骤 1/6: 创建目录结构...")
        
        directories = [
            self.scripts_dir,
            self.json_dir,
            self.config_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"创建目录: {directory.relative_to(self.output_dir)}")
    
    def copy_unity_scripts(self):
        """复制Unity C#脚本"""
        print_info("\n步骤 2/6: 复制Unity C#脚本...")
        
        unity_examples = self.project_root / "unity_examples"
        
        scripts = [
            "BrainDataStructures.cs",
            "BrainVisualization.cs",
            "BrainConfigLoader.cs",
            "WebSocketClient.cs",
        ]
        
        for script in scripts:
            src = unity_examples / script
            if src.exists():
                dst = self.scripts_dir / script
                shutil.copy2(src, dst)
                print_success(f"复制脚本: {script}")
            else:
                print_warning(f"脚本不存在: {script}")
    
    def generate_example_data(self):
        """生成示例大脑数据"""
        print_info("\n步骤 3/6: 生成示例大脑数据...")
        
        # 生成示例大脑状态数据
        example_data = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "subject": "示例数据",
                "atlas": "Schaefer200",
                "model_version": "1.0",
                "time_point": 0,
                "time_second": 0.0
            },
            "brain_state": {
                "time_point": 0,
                "time_second": 0.0,
                "regions": [],
                "connections": [],
                "global_metrics": {
                    "mean_activity": 0.5,
                    "std_activity": 0.1,
                    "max_activity": 0.8,
                    "active_regions": 100
                }
            },
            "stimulation": {
                "active": False,
                "target_regions": [],
                "amplitude": 0.0
            }
        }
        
        # 生成100个示例脑区
        import math
        n_regions = 100
        for i in range(n_regions):
            # 将脑区排列成球形
            theta = 2 * math.pi * i / n_regions
            phi = math.acos(1 - 2 * (i + 0.5) / n_regions)
            r = 50  # 半径
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            region = {
                "id": i,
                "label": f"Region_{i:03d}",
                "position": {"x": x, "y": y, "z": z},
                "activity": {
                    "fmri": {
                        "amplitude": 0.3 + 0.4 * math.sin(i * 0.1),
                        "raw_value": 0.5
                    },
                    "eeg": {
                        "amplitude": 0.2,
                        "raw_value": 0.3
                    }
                }
            }
            example_data["brain_state"]["regions"].append(region)
        
        # 生成一些连接
        for i in range(0, n_regions - 1, 5):
            connection = {
                "source": i,
                "target": i + 1,
                "strength": 0.6 + 0.3 * math.sin(i * 0.2),
                "type": "structural" if i % 2 == 0 else "functional"
            }
            example_data["brain_state"]["connections"].append(connection)
        
        # 保存示例数据
        example_file = self.json_dir / "brain_state_example.json"
        with open(example_file, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, indent=2, ensure_ascii=False)
        print_success(f"生成示例数据: {example_file.name}")
        
        # 生成序列索引
        sequence_index = {
            "subject": "示例数据",
            "start": 0,
            "end": 1,
            "step": 1,
            "n_frames": 1,
            "files": ["brain_state_example.json"]
        }
        
        index_file = self.json_dir / "sequence_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(sequence_index, f, indent=2, ensure_ascii=False)
        print_success(f"生成序列索引: {index_file.name}")
    
    def generate_config_files(self):
        """生成配置文件"""
        print_info("\n步骤 4/6: 生成Unity配置文件...")
        
        unity_config = {
            "project_name": "TwinBrain Unity Demo",
            "atlas": "Schaefer200",
            "data_paths": {
                "json_dir": "BrainData/JSON",
                "obj_dir": "BrainData/OBJ",
                "materials_dir": "Materials"
            },
            "visualization": {
                "region_scale": 0.5,
                "activity_threshold": 0.2,
                "connection_threshold": 0.4,
                "show_connections": True,
                "fps": 10,
                "auto_play": False
            },
            "colors": {
                "low_activity": {"r": 0, "g": 0, "b": 255},
                "high_activity": {"r": 255, "g": 0, "b": 0},
                "connection_structural": {"r": 255, "g": 255, "b": 255, "a": 128},
                "connection_functional": {"r": 255, "g": 255, "b": 0, "a": 128}
            },
            "animation": {
                "start_frame": 0,
                "end_frame": 100,
                "frame_step": 1
            }
        }
        
        config_file = self.config_dir / "unity_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(unity_config, f, indent=2, ensure_ascii=False)
        print_success(f"生成Unity配置: {config_file.name}")
    
    def generate_documentation(self):
        """生成使用说明文档"""
        print_info("\n步骤 5/6: 生成使用说明...")
        
        readme_content = """# TwinBrain Unity 项目设置完成！

## 📁 项目结构

```
Unity_TwinBrain/
├── Scripts/                    # Unity C#脚本（复制到Unity的Assets/Scripts/）
│   ├── BrainDataStructures.cs  # 数据结构定义
│   ├── BrainVisualization.cs   # 可视化主脚本
│   ├── BrainConfigLoader.cs    # 配置加载器
│   └── WebSocketClient.cs      # WebSocket客户端（可选）
├── BrainData/                  # 大脑数据（复制到Unity的StreamingAssets/）
│   ├── JSON/                   # JSON数据文件
│   │   ├── brain_state_example.json
│   │   └── sequence_index.json
│   └── Config/                 # 配置文件
│       └── unity_config.json
└── README_UNITY.md            # 本文件
```

## 🚀 快速开始（3步完成）

### 步骤1: 创建Unity项目

1. 打开Unity Hub
2. 创建新的3D项目（推荐Unity 2021或更新版本）
3. 项目名称可以任意，例如"TwinBrain_Demo"

### 步骤2: 导入脚本和数据

1. **导入C#脚本**：
   - 将 `Scripts/` 文件夹整个复制到Unity项目的 `Assets/` 目录
   - Unity会自动编译这些脚本

2. **导入数据文件**：
   - 在Unity的 `Assets/` 目录下创建 `StreamingAssets` 文件夹（如果不存在）
   - 将 `BrainData/` 文件夹整个复制到 `Assets/StreamingAssets/`

3. **安装依赖**：
   - 在Unity中打开 Package Manager (Window > Package Manager)
   - 搜索并安装 "Newtonsoft Json" (或者从 https://github.com/jilleJr/Newtonsoft.Json-for-Unity/releases 下载)

### 步骤3: 设置场景

1. **创建空GameObject**：
   - 在Hierarchy中右键 > Create Empty
   - 命名为 "BrainManager"

2. **附加脚本**：
   - 选中 BrainManager
   - 在Inspector中点击 "Add Component"
   - 搜索并添加 "Brain Visualization"
   - 再添加 "Brain Config Loader"

3. **配置路径**：
   - 在 Brain Visualization 组件中：
     - Json Path: `StreamingAssets/BrainData/JSON`
     - Load Sequence: 勾选
   - 在 Brain Config Loader 组件中：
     - Config Path: `StreamingAssets/BrainData/Config/unity_config.json`

4. **创建脑区预制体（可选）**：
   - 在Hierarchy中创建一个Sphere (GameObject > 3D Object > Sphere)
   - 调整大小为 (0.1, 0.1, 0.1)
   - 将其拖到Project窗口创建Prefab
   - 将Prefab赋值给 Brain Visualization 的 Region Prefab 字段
   - 删除Hierarchy中的Sphere

5. **点击播放**：
   - 按Unity的播放按钮
   - 你应该能看到大脑区域被可视化出来！

## ⌨️ 快捷键

- **Space**: 播放/暂停动画
- **R**: 重新加载数据

## 🎨 可视化设置

在 Brain Visualization 组件中可以调整：

- **Region Scale**: 脑区大小
- **Activity Threshold**: 显示活动的阈值（0-1）
- **Show Connections**: 是否显示连接
- **Connection Threshold**: 显示连接的阈值（0-1）
- **FPS**: 动画帧率
- **Colors**: 高/低活动颜色

## 🔧 高级功能

### 使用自己的数据

将你自己的TwinBrain导出的JSON文件放到 `StreamingAssets/BrainData/JSON/` 目录，然后：

1. 更新 `sequence_index.json` 文件，列出你的JSON文件
2. 在Unity中调整 Brain Visualization 的 Json Path

### WebSocket实时连接（可选）

如果需要实时连接到TwinBrain后端：

1. 启动TwinBrain WebSocket服务器：
   ```bash
   cd [TwinBrain项目根目录]
   python -m unity_integration.realtime_server
   ```

2. 在Unity中添加 WebSocket Client 组件到 BrainManager
3. 配置服务器URL（默认 ws://localhost:8765）
4. 使用脚本的API方法获取实时数据

## 📚 脚本API

### BrainVisualization 主要方法

```csharp
// 加载单个大脑状态
LoadSingleState(string path)

// 加载序列
LoadSequence()

// 播放/暂停
Play()
Pause()

// 重新加载
Reload()
```

### WebSocketClient 主要方法

```csharp
// 连接/断开
Connect()
Disconnect()

// 获取当前状态
GetBrainState()

// 请求预测
RequestPrediction(int nSteps)

// 模拟刺激
SimulateStimulation(int[] regions, float amplitude)
```

## ❓ 常见问题

### Q: 脚本编译错误？
A: 确保已安装 Newtonsoft.Json 包

### Q: 找不到JSON文件？
A: 检查文件是否在 `Assets/StreamingAssets/` 目录下

### Q: 看不到任何可视化？
A: 检查以下几点：
   - Json Path 是否正确
   - Load Sequence 是否勾选
   - Activity Threshold 是否过高

### Q: 连接线不显示？
A: 需要创建或指定 Connection Material

## 📞 获取帮助

如有问题，请查看TwinBrain主文档或提issue。

---
生成时间: {timestamp}
"""
        
        readme_file = self.output_dir / "README_UNITY.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print_success(f"生成使用说明: {readme_file.name}")
    
    def generate_server_scripts(self):
        """生成服务器脚本（可选）"""
        print_info("\n步骤 6/6: 生成后端服务器脚本...")
        
        server_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
TwinBrain Unity 后端服务器启动脚本
\"\"\"

import sys
from pathlib import Path

# 添加TwinBrain路径
twinbrain_root = Path(__file__).parent.parent
sys.path.insert(0, str(twinbrain_root))

from unity_integration.realtime_server import main

if __name__ == "__main__":
    print("启动TwinBrain WebSocket服务器...")
    print("服务器地址: ws://localhost:8765")
    print("按 Ctrl+C 停止服务器")
    main()
"""
        
        server_file = self.output_dir / "start_server.py"
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(server_script)
        server_file.chmod(0o755)
        print_success(f"生成服务器启动脚本: {server_file.name}")
    
    def print_completion_summary(self):
        """打印完成摘要"""
        print_header("✅ 设置完成！")
        
        print(f"{Colors.BOLD}生成的文件位置：{Colors.END}")
        print(f"  📁 {self.output_dir.absolute()}")
        print()
        
        print(f"{Colors.BOLD}接下来的步骤：{Colors.END}")
        print(f"  1. 阅读 {Colors.GREEN}{self.output_dir}/README_UNITY.md{Colors.END}")
        print(f"  2. 在Unity中创建新项目")
        print(f"  3. 将Scripts文件夹复制到Unity的Assets目录")
        print(f"  4. 将BrainData文件夹复制到Unity的Assets/StreamingAssets目录")
        print(f"  5. 按照README说明配置场景")
        print()
        
        if self.with_server:
            print(f"{Colors.BOLD}启动后端服务器：{Colors.END}")
            print(f"  {Colors.GREEN}python {self.output_dir}/start_server.py{Colors.END}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity 一键式项目设置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python setup_unity_project.py
  python setup_unity_project.py --output my_unity_project
  python setup_unity_project.py --with-server
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='Unity_TwinBrain',
        help='输出目录（默认: Unity_TwinBrain）'
    )
    
    parser.add_argument(
        '--with-server',
        action='store_true',
        help='同时生成后端服务器启动脚本'
    )
    
    args = parser.parse_args()
    
    # 执行设置
    setup = UnityProjectSetup(
        output_dir=args.output,
        with_server=args.with_server
    )
    setup.setup()


if __name__ == "__main__":
    main()
