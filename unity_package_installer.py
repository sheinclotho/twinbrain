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
        
    def validate_unity_project(self) -> Tuple[bool, List[str], List[str]]:
        """
        验证Unity项目结构
        
        Returns:
            (is_valid, fatal_issues, warnings): 验证结果、致命问题列表和警告列表
        """
        logger.info("验证Unity项目结构...")
        
        fatal_issues = []
        warnings = []
        
        # 检查是否是Unity项目
        if not (self.unity_project / "ProjectSettings").exists():
            fatal_issues.append("不是有效的Unity项目（缺少ProjectSettings目录）")
            return False, fatal_issues, warnings
        
        # 检查Assets目录
        if not self.assets_dir.exists():
            fatal_issues.append("Assets目录不存在")
            return False, fatal_issues, warnings
        
        # 检查Packages目录（可选，旧版Unity可能没有）
        if not self.packages_dir.exists():
            warnings.append("Packages目录不存在（可能是旧版Unity，将尝试创建）")
        
        # 检查TwinBrain脚本（可自动安装）
        if self.scripts_dir.exists():
            script_files = list(self.scripts_dir.glob("*.cs"))
            if len(script_files) > 0:
                logger.info(f"✓ 找到 {len(script_files)} 个TwinBrain脚本")
            else:
                warnings.append("TwinBrain脚本目录存在但为空（将安装脚本）")
        else:
            warnings.append("TwinBrain脚本未安装（将自动安装）")
        
        # 检查StreamingAssets（可自动创建）
        if not self.streaming_assets.exists():
            warnings.append("StreamingAssets目录不存在（将自动创建）")
        
        # 检查Newtonsoft.Json包（可自动安装）
        manifest_file = self.packages_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    dependencies = manifest.get("dependencies", {})
                    
                    if "com.unity.nuget.newtonsoft-json" not in dependencies:
                        warnings.append("Newtonsoft.Json包未安装（将尝试自动添加到manifest.json）")
                    else:
                        logger.info("✓ Newtonsoft.Json包已安装")
            except Exception as e:
                logger.warning(f"无法读取packages manifest: {e}")
                warnings.append("无法读取packages manifest（将尝试创建）")
        else:
            warnings.append("packages manifest不存在（将尝试创建）")
        
        # 只有致命问题才导致验证失败
        if len(fatal_issues) == 0:
            if len(warnings) == 0:
                logger.info("✓ Unity项目验证通过，无需修复")
            else:
                logger.info(f"✓ Unity项目基本有效，发现 {len(warnings)} 个可修复的问题")
                for warning in warnings:
                    logger.info(f"  ⚠ {warning}")
            return True, [], warnings
        else:
            logger.error(f"✗ Unity项目验证失败，发现 {len(fatal_issues)} 个致命问题")
            for issue in fatal_issues:
                logger.error(f"  ✗ {issue}")
            return False, fatal_issues, warnings
    
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
        
        # 获取所有C#脚本（排除Editor子目录）
        script_files = [f for f in self.source_scripts.glob("*.cs") if f.is_file()]
        
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
        
        # 安装Editor脚本
        editor_source = self.source_scripts / "Editor"
        if editor_source.exists() and editor_source.is_dir():
            editor_dest = self.scripts_dir / "Editor"
            editor_dest.mkdir(parents=True, exist_ok=True)
            
            editor_scripts = list(editor_source.glob("*.cs"))
            for editor_script in editor_scripts:
                dest_file = editor_dest / editor_script.name
                try:
                    shutil.copy2(editor_script, dest_file)
                    logger.info(f"  ✓ 安装Editor脚本: {editor_script.name}")
                    installed_count += 1
                except Exception as e:
                    logger.error(f"  ✗ 安装Editor脚本失败 {editor_script.name}: {e}")
        
        logger.info(f"✓ 成功安装 {installed_count} 个脚本")
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

查看项目文档: Unity使用指南.md
"""
        
        readme_file = self.streaming_assets / "README.md"
        try:
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info("  ✓ 创建README.md")
        except Exception as e:
            logger.warning(f"创建README失败: {e}")
        
        return True
    
    def install_newtonsoft_json(self) -> bool:
        """
        自动添加Newtonsoft.Json包到Unity项目
        
        Returns:
            是否成功
        """
        logger.info("安装Newtonsoft.Json依赖...")
        
        # 确保Packages目录存在
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_file = self.packages_dir / "manifest.json"
        
        # 创建或更新manifest.json
        manifest = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            except Exception as e:
                logger.warning(f"无法读取现有manifest.json: {e}，将创建新的")
        
        # 确保dependencies字段存在
        if "dependencies" not in manifest:
            manifest["dependencies"] = {}
        
        # 添加Newtonsoft.Json包
        newtonsoft_package = "com.unity.nuget.newtonsoft-json"
        if newtonsoft_package not in manifest["dependencies"]:
            manifest["dependencies"][newtonsoft_package] = "3.2.1"
            logger.info(f"  ✓ 添加 {newtonsoft_package} 到 manifest.json")
        else:
            logger.info(f"  ✓ {newtonsoft_package} 已存在于 manifest.json")
        
        # 写入manifest.json
        try:
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            logger.info("✓ manifest.json 更新成功")
            logger.info("  注意：Unity将在下次打开项目时自动下载包")
            return True
        except Exception as e:
            logger.error(f"更新manifest.json失败: {e}")
            return False
    
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
                "com.unity.nuget.newtonsoft-json": "3.2.1"
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
        在Unity项目中生成简化的使用指南，指向主文档
        
        Returns:
            是否成功
        """
        logger.info("生成使用指南...")
        
        guide_content = """# TwinBrain Unity 快速参考

> **完整文档**: 请查看TwinBrain仓库中的 **Unity使用指南.md**  
> 本文件仅作为快速参考，详细说明请参阅主文档。

---

## ✅ 自动化安装已完成

安装程序已经自动完成：
- ✅ C#脚本已复制到 `Assets/TwinBrain/Scripts/`
- ✅ 目录结构已创建
- ✅ Assembly Definition已生成
- ✅ Newtonsoft.Json依赖已配置（Unity将自动下载）

---

## 🚀 下一步操作

### 方法1: 使用自动化设置工具（推荐）

1. 在Unity Hub中打开项目
2. 等待Newtonsoft.Json包下载完成（1-2分钟）
3. 在Unity菜单中选择：**TwinBrain → 自动设置场景**
4. 点击"开始自动设置"按钮
5. 完成！所有OBJ文件和组件将自动配置

### 方法2: 手动配置

如果需要手动配置，请参阅完整文档中的详细步骤。

---

## 📚 文档链接

**主要文档** (在TwinBrain仓库根目录):
- **Unity使用指南.md** - Unity集成完整指南（最新版本v4.0）
- **UNIFIED_GUIDE.md** - 完整系统使用指南
- **TROUBLESHOOTING.md** - 问题排查指南

---

## ❓ 常见问题快速解答

**Q: 找不到Newtonsoft.Json类型？**
A: 等待Unity自动下载完成，或在Package Manager中手动添加: `com.unity.nuget.newtonsoft-json`

**Q: 场景中看不到脑区？**
A: 使用自动设置工具（TwinBrain → 自动设置场景）自动配置所有对象

**Q: OBJ文件无法拖动到场景？**
A: 使用自动设置工具会自动导入并配置所有OBJ文件，无需手动拖动

---

**安装版本**: 2.6  
**更新日期**: 2024-02-15  
**请查阅 Unity使用指南.md 获取完整文档**
"""
        
        guide_file = self.assets_dir / "TwinBrain" / "USAGE_GUIDE.md"
        
        try:
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            logger.info(f"✓ 创建: USAGE_GUIDE.md (指向主文档)")
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
        is_valid, fatal_issues, warnings = self.validate_unity_project()
        if not is_valid:
            logger.error("Unity项目验证失败，存在无法自动修复的致命问题：")
            for issue in fatal_issues:
                logger.error(f"  ✗ {issue}")
            logger.error("请先修复以上问题后再运行安装程序")
            return False
        
        # 如果有警告，显示将要执行的修复
        if warnings:
            logger.info("\n将自动修复以下问题：")
            for warning in warnings:
                logger.info(f"  → {warning}")
            logger.info("")
        
        # 2. 安装脚本
        if not self.install_scripts():
            logger.error("脚本安装失败")
            return False
        
        # 3. 安装Newtonsoft.Json依赖
        self.install_newtonsoft_json()
        
        # 4. 创建Assembly Definition
        self.create_assembly_definition()
        
        # 5. 设置StreamingAssets
        self.setup_streaming_assets(unity_project_data)
        
        # 6. 创建Package定义
        self.create_package_json()
        
        # 7. 生成使用指南
        self.generate_usage_guide()
        
        logger.info("="*80)
        logger.info("✓ 自动安装完成！")
        logger.info("="*80)
        logger.info("\n后续步骤：")
        logger.info("")
        logger.info("1. 在Unity中打开项目")
        logger.info("   - Unity会自动下载Newtonsoft.Json包（需要1-2分钟）")
        logger.info("   - 等待进度条完成")
        logger.info("")
        logger.info("2. 手动创建场景对象（由于Unity编辑器限制，无法自动化）：")
        logger.info("   a. 创建空GameObject，命名为 'BrainManager'")
        logger.info("   b. 添加组件：BrainVisualization, WebSocketClientImproved, BrainConfigLoader")
        logger.info("   c. 创建脑区预制体（Sphere或OBJ模型）")
        logger.info("   d. 配置组件参数")
        logger.info("")
        logger.info("3. 准备数据文件")
        logger.info("   - 将JSON文件复制到 Assets/StreamingAssets/brain_states/")
        logger.info("   - 或使用Cache转换功能（需创建UI）")
        logger.info("")
        logger.info("4. 查看详细使用说明")
        logger.info("   - Assets/TwinBrain/USAGE_GUIDE.md（Unity项目内）")
        logger.info("   - Unity使用指南.md（TwinBrain仓库）")
        logger.info("")
        logger.info("提示：手动步骤只需完成一次，之后可保存为场景模板重复使用")
        logger.info("")
        
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
    is_valid, fatal_issues, warnings = installer.validate_unity_project()
    
    if args.validate_only:
        if is_valid:
            logger.info("\n✓ Unity项目验证通过")
            if warnings:
                logger.info("\n发现以下可修复的问题（运行安装程序将自动修复）：")
                for warning in warnings:
                    logger.info(f"  ⚠ {warning}")
            return 0
        else:
            logger.error("\n✗ Unity项目验证失败")
            logger.error("致命问题：")
            for issue in fatal_issues:
                logger.error(f"  ✗ {issue}")
            return 1
    
    # 执行安装
    if installer.run_installation(data_dir):
        return 0
    else:
        logger.error("安装失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
