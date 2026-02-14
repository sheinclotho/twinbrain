#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Package Installer and Validator
======================================

自动化Unity项目设置工具，提供一键式安装和验证功能。

功能:
1. 验证Unity项目结构
2. 自动安装C#脚本到Unity Assets
3. 检查和创建必要的文件夹
4. 生成Unity Package Manager (UPM) 包定义
5. 创建Assembly Definition文件
6. 验证依赖项
7. 生成使用说明

使用方法:
    python unity_package_installer.py --unity-project /path/to/UnityProject
    python unity_package_installer.py --unity-project /path/to/UnityProject --validate-only
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnityPackageInstaller:
    """Unity包安装和验证工具"""
    
    def __init__(self, unity_project_path: Path, twinbrain_root: Path = None):
        """
        初始化安装器
        
        Args:
            unity_project_path: Unity项目路径
            twinbrain_root: TwinBrain项目根目录
        """
        self.unity_project = unity_project_path
        self.twinbrain_root = twinbrain_root or Path(__file__).parent
        
        # Unity路径
        self.assets_dir = self.unity_project / "Assets"
        self.scripts_dir = self.assets_dir / "TwinBrain" / "Scripts"
        self.streaming_assets = self.assets_dir / "StreamingAssets"
        self.packages_dir = self.unity_project / "Packages"
        
        # TwinBrain源文件路径
        self.source_scripts = self.twinbrain_root / "unity_examples"
        
    def validate_unity_project(self) -> Tuple[bool, List[str]]:
        """
        验证Unity项目结构
        
        Returns:
            (is_valid, issues): 验证结果和问题列表
        """
        logger.info("验证Unity项目结构...")
        
        issues = []
        
        # 检查是否是Unity项目
        if not (self.unity_project / "ProjectSettings").exists():
            issues.append("不是有效的Unity项目（缺少ProjectSettings目录）")
            return False, issues
        
        # 检查Assets目录
        if not self.assets_dir.exists():
            issues.append("Assets目录不存在")
            return False, issues
        
        # 检查Packages目录
        if not self.packages_dir.exists():
            issues.append("Packages目录不存在（这不是严重问题，但可能是旧版Unity）")
        
        # 检查TwinBrain脚本
        if self.scripts_dir.exists():
            script_files = list(self.scripts_dir.glob("*.cs"))
            if len(script_files) > 0:
                logger.info(f"✓ 找到 {len(script_files)} 个TwinBrain脚本")
            else:
                issues.append("TwinBrain脚本目录存在但为空")
        else:
            issues.append("TwinBrain脚本未安装")
        
        # 检查StreamingAssets
        if not self.streaming_assets.exists():
            issues.append("StreamingAssets目录不存在（将创建）")
        
        # 检查Newtonsoft.Json包
        manifest_file = self.packages_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    dependencies = manifest.get("dependencies", {})
                    
                    if "com.unity.nuget.newtonsoft-json" not in dependencies:
                        issues.append("Newtonsoft.Json包未安装（需要手动安装）")
                    else:
                        logger.info("✓ Newtonsoft.Json包已安装")
            except Exception as e:
                logger.warning(f"无法读取packages manifest: {e}")
        
        if len(issues) == 0:
            logger.info("✓ Unity项目验证通过")
            return True, []
        else:
            logger.warning(f"发现 {len(issues)} 个问题")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False, issues
    
    def install_scripts(self) -> bool:
        """
        安装C#脚本到Unity项目
        
        Returns:
            是否成功
        """
        logger.info("安装TwinBrain C#脚本...")
        
        if not self.source_scripts.exists():
            logger.error(f"源脚本目录不存在: {self.source_scripts}")
            return False
        
        # 创建目标目录
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有C#脚本
        script_files = list(self.source_scripts.glob("*.cs"))
        
        if len(script_files) == 0:
            logger.error("未找到C#脚本文件")
            return False
        
        # 复制脚本
        installed_count = 0
        for script_file in script_files:
            dest_file = self.scripts_dir / script_file.name
            
            try:
                shutil.copy2(script_file, dest_file)
                logger.info(f"  ✓ 安装: {script_file.name}")
                installed_count += 1
            except Exception as e:
                logger.error(f"  ✗ 安装失败 {script_file.name}: {e}")
        
        logger.info(f"✓ 成功安装 {installed_count}/{len(script_files)} 个脚本")
        return installed_count > 0
    
    def create_assembly_definition(self) -> bool:
        """
        创建Assembly Definition文件
        
        Returns:
            是否成功
        """
        logger.info("创建Assembly Definition...")
        
        asmdef_content = {
            "name": "TwinBrain.Scripts",
            "rootNamespace": "TwinBrain",
            "references": [],
            "includePlatforms": [],
            "excludePlatforms": [],
            "allowUnsafeCode": False,
            "overrideReferences": False,
            "precompiledReferences": [
                "Newtonsoft.Json.dll"
            ],
            "autoReferenced": True,
            "defineConstraints": [],
            "versionDefines": [],
            "noEngineReferences": False
        }
        
        asmdef_file = self.scripts_dir / "TwinBrain.Scripts.asmdef"
        
        try:
            with open(asmdef_file, 'w', encoding='utf-8') as f:
                json.dump(asmdef_content, f, indent=2)
            logger.info(f"✓ 创建: {asmdef_file.name}")
            return True
        except Exception as e:
            logger.error(f"创建Assembly Definition失败: {e}")
            return False
    
    def setup_streaming_assets(self, unity_project_data: Optional[Path] = None) -> bool:
        """
        设置StreamingAssets目录结构
        
        Args:
            unity_project_data: unity_project数据目录路径
            
        Returns:
            是否成功
        """
        logger.info("设置StreamingAssets...")
        
        # 创建必要的子目录
        subdirs = [
            "brain_states",      # JSON状态文件
            "config",            # 配置文件
            "OBJ"               # 3D模型（可选）
        ]
        
        for subdir in subdirs:
            dir_path = self.streaming_assets / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ 创建: {subdir}/")
        
        # 复制配置文件
        if unity_project_data:
            config_source = unity_project_data / "unity_config.json"
            if config_source.exists():
                config_dest = self.streaming_assets / "config" / "unity_config.json"
                try:
                    shutil.copy2(config_source, config_dest)
                    logger.info("  ✓ 复制配置文件")
                except Exception as e:
                    logger.warning(f"复制配置文件失败: {e}")
        
        # 创建README
        readme_content = """# TwinBrain StreamingAssets

此目录包含Unity运行时需要的数据文件。

## 目录结构

- **brain_states/**: JSON格式的大脑状态文件
- **config/**: Unity配置文件（unity_config.json）
- **OBJ/**: 3D脑区模型文件（可选）

## 数据准备

1. 运行 TwinBrain 后端服务器生成 brain_states/*.json 文件
2. 或使用 brain_state_exporter 工具转换预处理数据
3. 将 unity_config.json 放入 config/ 目录

## 更多信息

查看项目文档: Unity一键使用指南.md
"""
        
        readme_file = self.streaming_assets / "README.md"
        try:
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info("  ✓ 创建README.md")
        except Exception as e:
            logger.warning(f"创建README失败: {e}")
        
        return True
    
    def create_package_json(self) -> bool:
        """
        创建UPM包定义（用于未来支持）
        
        Returns:
            是否成功
        """
        logger.info("创建Package定义...")
        
        package_json = {
            "name": "com.twinbrain.unity",
            "version": "2.4.0",
            "displayName": "TwinBrain Unity Integration",
            "description": "Unity integration for TwinBrain digital twin brain system. Provides visualization and real-time communication with TwinBrain backend.",
            "unity": "2019.1",
            "keywords": [
                "brain",
                "neuroscience",
                "visualization",
                "digital-twin"
            ],
            "author": {
                "name": "TwinBrain Team"
            },
            "dependencies": {
                "com.unity.nuget.newtonsoft-json": "3.0.2"
            }
        }
        
        # 在TwinBrain/Scripts同级目录创建package.json
        package_dir = self.assets_dir / "TwinBrain"
        package_dir.mkdir(parents=True, exist_ok=True)
        package_file = package_dir / "package.json"
        
        try:
            with open(package_file, 'w', encoding='utf-8') as f:
                json.dump(package_json, f, indent=2)
            logger.info(f"✓ 创建: package.json")
            return True
        except Exception as e:
            logger.error(f"创建package.json失败: {e}")
            return False
    
    def generate_usage_guide(self) -> bool:
        """
        在Unity项目中生成使用指南
        
        Returns:
            是否成功
        """
        logger.info("生成使用指南...")
        
        guide_content = """# TwinBrain Unity 使用指南

## 快速开始

### 1. 检查依赖

确保已安装 Newtonsoft.Json:
1. Window > Package Manager
2. "+" > "Add package from git URL"
3. 输入: `com.unity.nuget.newtonsoft-json`

### 2. 创建场景

1. 创建空GameObject，命名为 "BrainManager"
2. 添加组件:
   - `BrainVisualization` (主可视化)
   - `WebSocketClientImproved` (通信，可选)
   - `BrainConfigLoader` (配置加载)

### 3. 配置组件

**BrainVisualization:**
- Json Path: `StreamingAssets/brain_states`
- Region Prefab: 脑区预制体（Sphere或OBJ模型）

**WebSocketClientImproved:**
- Server URL: `http://localhost:8765`
- Auto Connect: ✓

### 4. 准备数据

将JSON状态文件放入 `Assets/StreamingAssets/brain_states/`

### 5. 运行

按Play键启动可视化！

## 进阶功能

### Cache自动转换

1. 创建UI Canvas和Button
2. 添加 `CacheToJsonConverter` 组件
3. 连接UI元素
4. 点击按钮自动转换cache文件

### 实时通信

确保后端服务器运行:
```bash
python unity_startup.py --model results/model.pt
```

然后在Unity中:
```csharp
// 获取WebSocket客户端
var wsClient = GetComponent<WebSocketClientImproved>();

// 请求预测
wsClient.RequestPrediction(10, (response) => {
    Debug.Log("收到预测结果");
});

// 模拟刺激
int[] regions = {10, 20, 30};
wsClient.SimulateStimulation(regions, 0.5f, "sine", (response) => {
    Debug.Log("收到刺激模拟结果");
});
```

## 故障排除

### 找不到Newtonsoft.Json

确保已通过Package Manager安装，或手动添加DLL。

### WebSocket连接失败

1. 检查后端服务器是否运行
2. 确认URL和端口正确
3. 查看Unity Console的错误信息

### JSON文件加载失败

1. 确认文件在StreamingAssets目录
2. 检查文件格式是否正确
3. 查看日志中的具体错误

## 更多帮助

查看完整文档:
- Unity一键使用指南.md
- Unity架构说明.md
- GitHub Issues: https://github.com/sheinclotho/twinbrain/issues
"""
        
        guide_file = self.assets_dir / "TwinBrain" / "USAGE_GUIDE.md"
        
        try:
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            logger.info(f"✓ 创建: USAGE_GUIDE.md")
            return True
        except Exception as e:
            logger.error(f"创建使用指南失败: {e}")
            return False
    
    def run_installation(self, unity_project_data: Optional[Path] = None) -> bool:
        """
        执行完整安装流程
        
        Args:
            unity_project_data: unity_project数据目录
            
        Returns:
            是否成功
        """
        logger.info("="*80)
        logger.info("TwinBrain Unity Package 安装")
        logger.info("="*80)
        
        # 1. 验证Unity项目
        is_valid, issues = self.validate_unity_project()
        if not is_valid:
            logger.error("Unity项目验证失败，无法继续安装")
            return False
        
        # 2. 安装脚本
        if not self.install_scripts():
            logger.error("脚本安装失败")
            return False
        
        # 3. 创建Assembly Definition
        self.create_assembly_definition()
        
        # 4. 设置StreamingAssets
        self.setup_streaming_assets(unity_project_data)
        
        # 5. 创建Package定义
        self.create_package_json()
        
        # 6. 生成使用指南
        self.generate_usage_guide()
        
        logger.info("="*80)
        logger.info("✓ 安装完成！")
        logger.info("="*80)
        logger.info("\n后续步骤:")
        logger.info("1. 在Unity中打开项目")
        logger.info("2. 确保安装 Newtonsoft.Json (Window > Package Manager)")
        logger.info("3. 查看 Assets/TwinBrain/USAGE_GUIDE.md 了解使用方法")
        logger.info("4. 准备数据文件到 Assets/StreamingAssets/brain_states/")
        logger.info("\n")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TwinBrain Unity Package 安装和验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 安装到Unity项目
  python unity_package_installer.py --unity-project /path/to/UnityProject
  
  # 仅验证Unity项目
  python unity_package_installer.py --unity-project /path/to/UnityProject --validate-only
  
  # 指定数据目录
  python unity_package_installer.py --unity-project /path/to/UnityProject --data-dir unity_project
        """
    )
    
    parser.add_argument(
        '--unity-project',
        type=str,
        required=True,
        help='Unity项目路径'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='TwinBrain unity_project数据目录路径（可选）'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证Unity项目，不执行安装'
    )
    
    parser.add_argument(
        '--twinbrain-root',
        type=str,
        help='TwinBrain项目根目录（默认为脚本所在目录）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    unity_project = Path(args.unity_project).resolve()
    twinbrain_root = Path(args.twinbrain_root).resolve() if args.twinbrain_root else None
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    
    if not unity_project.exists():
        logger.error(f"Unity项目不存在: {unity_project}")
        return 1
    
    # 创建安装器
    installer = UnityPackageInstaller(unity_project, twinbrain_root)
    
    # 验证项目
    is_valid, issues = installer.validate_unity_project()
    
    if args.validate_only:
        if is_valid:
            logger.info("\n✓ Unity项目验证通过")
            return 0
        else:
            logger.error("\n✗ Unity项目验证失败")
            return 1
    
    # 执行安装
    if installer.run_installation(data_dir):
        return 0
    else:
        logger.error("安装失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
