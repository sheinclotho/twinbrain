# TwinBrain Unity 安装使用说明

> **重要**: 本指南提供真正的一键安装，无需手动拖拽200+个OBJ文件

## 🚀 快速开始（3步完成）

### 1. 创建Unity项目

在Unity Hub中创建新的3D项目，名称任意（如 `TwinBrainDemo`）

### 2. 运行一键安装脚本

```bash
cd twinbrain

# 如果有FreeSurfer数据（推荐）
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo \
    --freesurfer-dir /path/to/freesurfer

# 如果没有FreeSurfer数据
python unity_one_click_install.py \
    --unity-project /path/to/TwinBrainDemo
```

**这一步会自动完成**:
- ✅ 生成200+个OBJ文件（如果提供FreeSurfer）
- ✅ 复制所有脚本到Unity项目
- ✅ 复制所有OBJ文件到Unity项目
- ✅ 配置依赖和设置
- ✅ 安装自动化工具

### 3. 在Unity中自动设置

1. 在Unity Hub中打开项目
2. 等待Unity自动下载包（1-2分钟）
3. 在Unity菜单中: **TwinBrain -> 自动设置场景**
4. 点击**"开始自动设置"**按钮
5. 等待完成（会自动配置所有OBJ文件）

**完成！** 点击Play按钮测试

---

## 📖 详细说明

### 什么是一键安装？

以前需要：
1. ❌ 运行setup_unity_project.py生成OBJ
2. ❌ 运行unity_package_installer.py安装脚本
3. ❌ 手动复制OBJ文件
4. ❌ 在Unity中手动拖拽200+个OBJ文件
5. ❌ 手动设置每个OBJ的缩放、材质
6. ❌ 手动创建BrainManager
7. ❌ 手动添加组件

现在只需：
1. ✅ 运行`unity_one_click_install.py`（一行命令）
2. ✅ 在Unity中点击菜单按钮（自动完成所有配置）

### 自动化工具说明

安装脚本会在Unity项目中添加一个Editor工具（TwinBrainAutoSetup），它会：

- ✅ 自动导入所有OBJ文件
- ✅ 自动设置每个OBJ的缩放比例（0.01）
- ✅ 自动设置每个OBJ的材质导入
- ✅ 自动创建BrainManager GameObject
- ✅ 自动添加必要的组件
- ✅ 自动创建预制体
- ✅ 自动配置摄像机

**不需要手动拖拽或配置任何OBJ文件！**

### FreeSurfer数据说明

如果您有FreeSurfer处理的脑表面数据，需要以下文件：
- `lh.pial` - 左半球表面
- `rh.pial` - 右半球表面
- `lh.Schaefer2018_200Parcels_7Networks_order.annot` - 左半球标注
- `rh.Schaefer2018_200Parcels_7Networks_order.annot` - 右半球标注

如果没有这些文件，脚本会使用默认球体作为脑区模型。

---

## ❓ 常见问题

### Q: 我还需要运行其他脚本吗？

**A**: 不需要！`unity_one_click_install.py`已经整合了所有功能。

### Q: OBJ文件需要手动设置吗？

**A**: 不需要！Unity中的自动设置工具会处理所有OBJ文件的配置。

### Q: 找不到"TwinBrain"菜单？

**A**: 
1. 确保Unity已完全加载项目
2. 等待Newtonsoft.Json包下载完成
3. 如果仍然没有，检查Console是否有编译错误

### Q: 自动设置工具报错？

**A**: 
1. 确保所有包已下载完成
2. 保存当前场景后再运行
3. 查看Unity Console的详细错误信息

### Q: setup_unity_project.py 和 unity_package_installer.py 还需要吗？

**A**: 不需要了！`unity_one_click_install.py`已经整合了它们的所有功能。如果您愿意，仍然可以分别运行它们，但推荐使用新的一键安装脚本。

---

## 🔧 高级选项

### 指定中间文件输出目录

```bash
python unity_one_click_install.py \
    --unity-project /path/to/UnityProject \
    --freesurfer-dir /path/to/freesurfer \
    --output-dir ./my_output
```

### 仅生成资源（不安装到Unity）

```bash
python setup_unity_project.py --auto-setup \
    --freesurfer-dir /path/to/freesurfer
```

### 仅安装到Unity（不生成资源）

```bash
python unity_package_installer.py \
    --unity-project /path/to/UnityProject \
    --data-dir unity_project
```

---

## 📞 获取帮助

如果遇到问题：
1. 查看Unity Console的错误信息
2. 查看Python终端的日志输出
3. 创建GitHub Issue并附上错误信息

---

**版本**: 4.0  
**更新日期**: 2024-02-15  
**改进**: 真正的一键安装，无需手动配置OBJ文件
