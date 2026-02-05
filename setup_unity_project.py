#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain Unity 一键式自动化设置脚本
===========================================

完全自动化Unity项目设置，基于真实数据。

功能：
1. 自动创建Unity项目文件夹结构（freesurfer/, data/, state/）
2. 复制Unity C#脚本到项目
3. 如果有FreeSurfer文件，生成真实的200个脑区OBJ模型
4. 基于实际atlas文件创建配置
5. 提供详细的使用说明

使用方法：
    # 基本使用（创建项目结构，不生成示例数据）
    python setup_unity_project.py
    
    # 使用FreeSurfer数据自动生成OBJ模型
    python setup_unity_project.py --freesurfer-dir /path/to/freesurfer/files
    
    # 指定atlas（必须与freesurfer文件匹配）
    python setup_unity_project.py --freesurfer-dir /path/to/fs --atlas Schaefer200

完成后：
    1. 将FreeSurfer文件放入生成的freesurfer/目录（如果还没有）
    2. 将脑数据放入data/目录（原始或预处理格式）
    3. 在Unity中创建新项目
    4. 将Scripts文件夹复制到Unity的Assets目录
    5. 按照生成的README_UNITY.md说明操作
"""

import argparse
import json
import shutil
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

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
    """Unity项目一键式设置 - 基于真实数据"""
    
    def __init__(
        self,
        output_dir="Unity_TwinBrain",
        freesurfer_dir=None,
        atlas_name="Schaefer200",
        with_server=False
    ):
        self.project_root = Path(__file__).parent
        self.output_dir = Path(output_dir)
        self.freesurfer_dir = Path(freesurfer_dir) if freesurfer_dir else None
        self.atlas_name = atlas_name
        self.with_server = with_server
        
        # 定义标准文件夹结构（按照用户要求）
        self.freesurfer_folder = self.output_dir / "freesurfer"
        self.data_folder = self.output_dir / "data"
        self.state_folder = self.output_dir / "state"
        self.scripts_dir = self.output_dir / "Scripts"
        self.obj_dir = self.output_dir / "OBJ"  # 3D模型输出
        
        # 数据子文件夹
        self.data_raw = self.data_folder / "raw"  # 原始数据
        self.data_cache = self.data_folder / "cache"  # 预处理缓存
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup(self):
        """执行完整设置流程"""
        print_header("TwinBrain Unity 一键式项目设置（基于真实数据）")
        
        try:
            # 步骤1: 创建目录结构
            self.create_directory_structure()
            
            # 步骤2: 复制Unity脚本
            self.copy_unity_scripts()
            
            # 步骤3: 加载atlas信息
            atlas_info = self.load_atlas_info()
            
            # 步骤4: 如果提供了FreeSurfer文件，生成OBJ模型
            if self.freesurfer_dir and self.freesurfer_dir.exists():
                self.generate_freesurfer_models(atlas_info)
            else:
                print_warning("未提供FreeSurfer文件，跳过OBJ模型生成")
                print_info(f"将FreeSurfer文件放入: {self.freesurfer_folder}")
            
            # 步骤5: 生成配置文件
            self.generate_config_files(atlas_info)
            
            # 步骤6: 生成使用说明
            self.generate_documentation(atlas_info)
            
            # 步骤7: 可选 - 生成服务器脚本
            if self.with_server:
                self.generate_server_scripts()
            
            # 完成
            self.print_completion_summary()
            
        except Exception as e:
            print_error(f"设置过程中出错: {e}")
            self.logger.exception("详细错误:")
            sys.exit(1)
    
    def create_directory_structure(self):
        """创建目录结构"""
        print_info("步骤 1/7: 创建目录结构...")
        
        directories = [
            (self.freesurfer_folder, "FreeSurfer表面数据文件（.pial, .annot）"),
            (self.data_raw, "原始fMRI/EEG数据"),
            (self.data_cache, "预处理缓存数据"),
            (self.state_folder, "处理后的JSON状态文件"),
            (self.scripts_dir, "Unity C#脚本"),
            (self.obj_dir, "3D脑区模型（OBJ格式）"),
        ]
        
        for directory, description in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"创建: {directory.relative_to(self.output_dir)} - {description}")
        
        # 创建README文件说明各文件夹用途
        self._create_folder_readmes()
    
    def _create_folder_readmes(self):
        """为每个文件夹创建README说明"""
        readmes = {
            self.freesurfer_folder / "README.md": """# FreeSurfer 文件夹

存放FreeSurfer处理后的表面文件和标注文件。

## 所需文件

如果使用FreeSurfer数据，请将以下文件放入此文件夹：

- `lh.pial` - 左半球表面文件
- `rh.pial` - 右半球表面文件  
- `lh.Schaefer2018_200Parcels_7Networks_order.annot` - 左半球标注文件
- `rh.Schaefer2018_200Parcels_7Networks_order.annot` - 右半球标注文件

## 生成模型

运行以下命令生成OBJ模型：
```bash
python setup_unity_project.py --freesurfer-dir freesurfer --atlas Schaefer200
```

这将自动：
1. 读取FreeSurfer表面和标注文件
2. 生成200个脑区的3D OBJ模型
3. 保存到OBJ/文件夹
""",
            self.data_folder / "README.md": """# Data 文件夹

存放实际的脑数据文件。

## 数据格式

### raw/ - 原始数据
- fMRI数据（NIfTI格式）
- EEG数据（可选）
- 其他原始格式数据

### cache/ - 预处理缓存
- 预处理后的数据
- 提取的时间序列
- 连接矩阵

## 数据处理流程

1. 将原始数据放入 `raw/` 文件夹
2. 运行预处理脚本（TwinBrain主项目）
3. 缓存数据自动保存到 `cache/` 文件夹
4. 处理后的状态保存到 `state/` 文件夹（JSON格式）

## 在Unity中使用

Unity脚本可以从 `state/` 文件夹读取处理好的JSON文件进行可视化。
""",
            self.state_folder / "README.md": """# State 文件夹

存放处理后的大脑状态JSON文件，供Unity可视化使用。

## 文件格式

每个JSON文件包含：
- 脑区活动值（fMRI BOLD信号）
- 脑区连接强度
- 全局指标
- 时间戳

## 生成方式

### 方式1: 从数据处理生成
```python
# 使用TwinBrain处理data/文件夹中的数据
python -m unity_integration.brain_state_exporter \\
    --data-dir Unity_TwinBrain/data \\
    --output Unity_TwinBrain/state
```

### 方式2: 从模型预测生成
```python
# 使用训练好的模型生成预测状态
python -m unity_integration.brain_state_exporter \\
    --model results/hetero_gnn_trained.pt \\
    --output Unity_TwinBrain/state
```

### 方式3: 后端实时生成
启动WebSocket服务器，后端预测结果自动保存到此文件夹。

## Unity使用

在Unity的BrainVisualization组件中，设置JSON路径为此文件夹。
"""
        }
        
        for path, content in readmes.items():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def copy_unity_scripts(self):
        """复制Unity C#脚本"""
        print_info("\n步骤 2/7: 复制Unity C#脚本...")
        
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
    
    def load_atlas_info(self) -> Dict[str, Any]:
        """加载atlas信息（从实际atlas文件）"""
        print_info("\n步骤 3/7: 加载Atlas信息...")
        
        # 根据atlas名称加载对应的文件
        atlas_dir = self.project_root / "atlases"
        
        if self.atlas_name.startswith("Schaefer"):
            # 从Schaefer atlas文件加载
            n_parcels = 200  # 默认
            if "100" in self.atlas_name:
                n_parcels = 100
            elif "400" in self.atlas_name:
                n_parcels = 400
            
            atlas_file = atlas_dir / "schaefer_2018" / f"Schaefer2018_{n_parcels}Parcels_7Networks_order.txt"
            
            if atlas_file.exists():
                atlas_info = self._load_schaefer_atlas(atlas_file, n_parcels)
                print_success(f"加载Schaefer atlas: {n_parcels}个脑区")
                return atlas_info
            else:
                print_warning(f"Atlas文件不存在: {atlas_file}")
        
        # 如果找不到atlas文件，返回基本信息
        print_warning(f"使用{self.atlas_name}的基本信息（请提供atlas文件以获得完整信息）")
        return {
            'name': self.atlas_name,
            'n_regions': 200,  # 默认
            'regions': {}
        }
    
    def _load_schaefer_atlas(self, atlas_file: Path, n_parcels: int) -> Dict[str, Any]:
        """从Schaefer atlas文件加载脑区信息"""
        atlas_info = {
            'name': f'Schaefer{n_parcels}',
            'n_regions': n_parcels,
            'regions': {}
        }
        
        try:
            with open(atlas_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析格式: region_id x y z region_name network
                    parts = line.split()
                    if len(parts) >= 5:
                        region_id = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        region_name = parts[4] if len(parts) > 4 else f"Region_{region_id}"
                        network = parts[5] if len(parts) > 5 else "Unknown"
                        
                        atlas_info['regions'][region_id] = {
                            'label': region_name,
                            'xyz': [x, y, z],
                            'network': network
                        }
        except Exception as e:
            self.logger.warning(f"解析atlas文件时出错: {e}")
        
        return atlas_info
    
    def generate_freesurfer_models(self, atlas_info: Dict[str, Any]):
        """从FreeSurfer文件生成OBJ模型"""
        print_info("\n步骤 4/7: 从FreeSurfer生成OBJ模型...")
        
        try:
            from unity_integration.freesurfer_loader import FreeSurferLoader
            from unity_integration.obj_generator import BrainOBJGenerator
            
            # 检查FreeSurfer文件
            lh_pial = self.freesurfer_dir / "lh.pial"
            rh_pial = self.freesurfer_dir / "rh.pial"
            
            # 根据atlas名称构造标注文件名
            if "Schaefer" in self.atlas_name:
                n_parcels = 200  # 默认
                if "100" in self.atlas_name:
                    n_parcels = 100
                elif "400" in self.atlas_name:
                    n_parcels = 400
                annot_base = f"Schaefer2018_{n_parcels}Parcels_7Networks_order"
            else:
                print_warning(f"未知的atlas类型: {self.atlas_name}，使用默认标注")
                annot_base = "Schaefer2018_200Parcels_7Networks_order"
            
            lh_annot = self.freesurfer_dir / f"lh.{annot_base}.annot"
            rh_annot = self.freesurfer_dir / f"rh.{annot_base}.annot"
            
            # 检查文件是否存在
            missing_files = []
            for f in [lh_pial, rh_pial, lh_annot, rh_annot]:
                if not f.exists():
                    missing_files.append(f.name)
            
            if missing_files:
                print_warning(f"缺少FreeSurfer文件: {', '.join(missing_files)}")
                print_info("请将以下文件放入freesurfer/文件夹:")
                for f in missing_files:
                    print_info(f"  - {f}")
                return
            
            # 加载FreeSurfer数据
            print_info("加载FreeSurfer表面和标注...")
            loader = FreeSurferLoader()
            
            # 加载双侧半球数据
            loader.load_bilateral_surfaces(
                lh_pial, rh_pial,
                lh_annot, rh_annot
            )
            
            # 转换为atlas格式
            atlas_info = loader.to_atlas_info(atlas_name=self.atlas_name)
            print_success(f"加载了{atlas_info['n_regions']}个脑区")
            
            # 生成OBJ模型
            print_info("生成OBJ 3D模型...")
            obj_gen = BrainOBJGenerator(
                atlas_info=atlas_info,
                sphere_resolution=16  # 16x16网格
            )
            
            # 为每个脑区生成OBJ文件
            for region_id, region_data in atlas_info['regions'].items():
                region_label = region_data['label']
                centroid = region_data['xyz']
                
                # 生成并保存OBJ
                obj_file = self.obj_dir / f"region_{region_id}_{region_label}.obj"
                obj_gen.export_region_obj(
                    region_id=int(region_id),
                    output_path=obj_file
                )
            
            print_success(f"生成了{atlas_info['n_regions']}个OBJ模型到: {self.obj_dir}")
            
        except ImportError as e:
            print_error(f"无法导入FreeSurfer加载器: {e}")
            print_info("请确保安装了nibabel: pip install nibabel")
        except Exception as e:
            print_error(f"生成OBJ模型时出错: {e}")
            self.logger.exception("详细错误:")
    
    def generate_config_files(self, atlas_info: Dict[str, Any]):
        """生成配置文件"""
        print_info("\n步骤 5/7: 生成Unity配置文件...")
        
        config = {
            "project_name": "TwinBrain Unity Project",
            "atlas": atlas_info['name'],
            "n_regions": atlas_info['n_regions'],
            "data_paths": {
                "freesurfer_dir": "freesurfer",
                "data_raw_dir": "data/raw",
                "data_cache_dir": "data/cache",
                "state_dir": "state",
                "obj_dir": "OBJ"
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
            "workflow": {
                "use_freesurfer": self.freesurfer_dir is not None,
                "auto_process_data": True,
                "save_predictions_to_state": True
            }
        }
        
        config_file = self.output_dir / "unity_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print_success(f"生成配置文件: {config_file.name}")
    
    def generate_documentation(self, atlas_info: Dict[str, Any]):
        """生成使用说明文档"""
        print_info("\n步骤 6/7: 生成使用说明...")
        
        has_freesurfer = self.freesurfer_dir and self.freesurfer_dir.exists()
        n_regions = atlas_info.get('n_regions', 200)
        
        readme_content = f"""# TwinBrain Unity 项目 - 使用说明

## 📁 项目结构

```
{self.output_dir.name}/
├── freesurfer/          # FreeSurfer表面文件（.pial, .annot）
├── data/
│   ├── raw/            # 原始fMRI/EEG数据
│   └── cache/          # 预处理缓存
├── state/              # 处理后的JSON状态文件
├── Scripts/            # Unity C#脚本（复制到Unity Assets/）
├── OBJ/                # 3D脑区模型（复制到Unity Assets/）
└── unity_config.json   # Unity配置文件
```

## 🎯 工作流程

### 阶段1: 构建（一次性）

{'✅ **已完成**: 从FreeSurfer生成了' + str(n_regions) + '个脑区OBJ模型' if has_freesurfer else '⚠️ **需要完成**: 将FreeSurfer文件放入freesurfer/文件夹'}

1. **准备FreeSurfer文件**（如果还没有）
   ```bash
   # 将以下文件复制到 freesurfer/ 文件夹:
   - lh.pial, rh.pial
   - lh.{self.atlas_name}.annot, rh.{self.atlas_name}.annot
   ```

2. **生成OBJ模型**（如果还没有）
   ```bash
   python setup_unity_project.py \\
       --freesurfer-dir {self.output_dir}/freesurfer \\
       --atlas {self.atlas_name}
   ```

3. **在Unity中导入**
   - 创建新Unity 3D项目
   - 将 `Scripts/` 复制到 `Assets/Scripts/`
   - 将 `OBJ/` 复制到 `Assets/Models/`
   - 安装Newtonsoft.Json包

### 阶段2: 使用（可重复）

**准备数据:**

1. 将实际脑数据放入 `data/raw/` 或 `data/cache/`
   - 支持原始fMRI数据（需预处理）
   - 支持预处理的时间序列（直接使用）

2. 处理数据生成状态JSON
   ```bash
   # 方法1: 从数据处理
   python -m unity_integration.brain_state_exporter \\
       --data-dir {self.output_dir}/data \\
       --output {self.output_dir}/state
   
   # 方法2: 从模型预测
   python -m unity_integration.brain_state_exporter \\
       --model results/hetero_gnn_trained.pt \\
       --output {self.output_dir}/state
   ```

3. 在Unity中加载状态JSON进行可视化

**实时模式:**

1. 启动后端服务器
   ```bash
   python -m unity_integration.realtime_server
   ```

2. Unity通过WebSocket连接
3. 后端预测自动保存到 `state/` 文件夹

## 🎮 Unity设置

### 1. 创建场景

1. Hierarchy > Create Empty > 命名"BrainManager"
2. Add Component > Brain Visualization
3. Add Component > Brain Config Loader

### 2. 配置组件

**Brain Visualization:**
- Json Path: `StreamingAssets/state`
- Load Sequence: ✓
- Region Prefab: 从OBJ模型创建预制体（或使用Sphere）

**Brain Config Loader:**
- Config Path: `StreamingAssets/unity_config.json`
- Auto Load: ✓

### 3. 复制数据到Unity

将以下文件夹复制到 `Assets/StreamingAssets/`:
- `state/` (JSON文件)
- `unity_config.json`

## 📊 数据处理流程

```
原始数据 (data/raw/) 
    ↓ [预处理]
缓存数据 (data/cache/)
    ↓ [提取状态]
JSON状态 (state/)
    ↓ [Unity加载]
可视化
```

## 🔄 后端交互

1. **启动后端服务器**
   ```bash
   python -m unity_integration.realtime_server
   ```

2. **Unity连接**
   - Add Component > WebSocket Client
   - Server URL: ws://localhost:8765

3. **预测流程**
   - Unity请求预测
   - 后端处理并生成JSON
   - 自动保存到 `state/` 文件夹
   - Unity加载新状态

## ⚙️ 配置说明

编辑 `unity_config.json` 调整:
- 可视化参数（颜色、阈值）
- 数据路径
- 工作流设置

## ❓ 常见问题

### Q: 没有FreeSurfer文件怎么办？
A: 可以使用Sphere预制体代替OBJ模型，仅需JSON状态文件即可可视化。

### Q: 如何添加更多数据？
A: 将数据放入 `data/` 文件夹，运行处理脚本生成新的JSON到 `state/`。

### Q: 预测结果在哪里？
A: 后端预测自动保存为JSON到 `state/` 文件夹，Unity可自动加载。

### Q: 支持其他atlas吗？
A: 支持，只需提供对应的FreeSurfer标注文件并指定atlas名称。

## 🎓 Atlas信息

当前使用: **{atlas_info['name']}**
脑区数量: **{n_regions}个**

---
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        readme_file = self.output_dir / "README_UNITY.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print_success(f"生成使用说明: {readme_file.name}")
    
    def generate_server_scripts(self):
        """生成服务器脚本"""
        print_info("\n步骤 7/7: 生成后端服务器脚本...")
        
        server_script = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
TwinBrain Unity 后端服务器启动脚本

预测结果自动保存到: {self.output_dir}/state/
\"\"\"

import sys
from pathlib import Path

# 添加TwinBrain路径
twinbrain_root = Path(__file__).parent.parent
sys.path.insert(0, str(twinbrain_root))

from unity_integration.realtime_server import main

if __name__ == "__main__":
    print("启动TwinBrain WebSocket服务器...")
    print(f"预测结果保存到: {self.output_dir}/state/")
    print("服务器地址: ws://localhost:8765")
    print("按 Ctrl+C 停止服务器")
    main()
"""
        
        server_file = self.output_dir / "start_server.py"
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(server_script)
        server_file.chmod(0o755)
        print_success(f"生成服务器脚本: {server_file.name}")
    
    def print_completion_summary(self):
        """打印完成摘要"""
        print_header("✅ 设置完成！")
        
        print(f"{Colors.BOLD}项目位置：{Colors.END}")
        print(f"  📁 {self.output_dir.absolute()}")
        print()
        
        print(f"{Colors.BOLD}文件夹结构：{Colors.END}")
        print(f"  freesurfer/ - FreeSurfer表面文件")
        print(f"  data/       - 实际脑数据（raw + cache）")
        print(f"  state/      - 处理后的JSON状态")
        print(f"  Scripts/    - Unity C#脚本")
        print(f"  OBJ/        - 3D脑区模型")
        print()
        
        if not (self.freesurfer_dir and self.freesurfer_dir.exists()):
            print(f"{Colors.BOLD}⚠️ 下一步（重要）：{Colors.END}")
            print(f"  1. 将FreeSurfer文件放入: {Colors.GREEN}{self.freesurfer_folder}{Colors.END}")
            print(f"  2. 运行: {Colors.GREEN}python setup_unity_project.py --freesurfer-dir {self.freesurfer_folder}{Colors.END}")
            print(f"  3. 这将生成真实的200个脑区OBJ模型")
            print()
        
        print(f"{Colors.BOLD}数据处理流程：{Colors.END}")
        print(f"  1. 将实际数据放入: {Colors.GREEN}data/raw/{Colors.END} 或 {Colors.GREEN}data/cache/{Colors.END}")
        print(f"  2. 运行数据处理生成JSON到: {Colors.GREEN}state/{Colors.END}")
        print(f"  3. Unity从state/文件夹加载JSON可视化")
        print()
        
        print(f"{Colors.BOLD}Unity设置：{Colors.END}")
        print(f"  1. 创建Unity 3D项目")
        print(f"  2. 将Scripts/复制到Assets/Scripts/")
        print(f"  3. 将OBJ/复制到Assets/Models/（如果有）")
        print(f"  4. 将state/和unity_config.json复制到Assets/StreamingAssets/")
        print(f"  5. 参考: {Colors.GREEN}README_UNITY.md{Colors.END}")
        print()
        
        if self.with_server:
            print(f"{Colors.BOLD}启动后端服务器：{Colors.END}")
            print(f"  {Colors.GREEN}python {self.output_dir}/start_server.py{Colors.END}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity 一键式项目设置（基于真实数据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 创建项目结构（不生成示例数据）
  python setup_unity_project.py
  
  # 使用FreeSurfer文件生成真实OBJ模型
  python setup_unity_project.py --freesurfer-dir /path/to/freesurfer --atlas Schaefer200
  
  # 指定输出目录
  python setup_unity_project.py --output my_project --freesurfer-dir /path/to/fs
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='Unity_TwinBrain',
        help='输出目录（默认: Unity_TwinBrain）'
    )
    
    parser.add_argument(
        '--freesurfer-dir',
        help='FreeSurfer文件目录（包含.pial和.annot文件）'
    )
    
    parser.add_argument(
        '--atlas',
        default='Schaefer200',
        help='Atlas名称（默认: Schaefer200）'
    )
    
    parser.add_argument(
        '--with-server',
        action='store_true',
        help='生成后端服务器启动脚本'
    )
    
    args = parser.parse_args()
    
    # 执行设置
    setup = UnityProjectSetup(
        output_dir=args.output,
        freesurfer_dir=args.freesurfer_dir,
        atlas_name=args.atlas,
        with_server=args.with_server
    )
    setup.setup()


if __name__ == "__main__":
    main()
