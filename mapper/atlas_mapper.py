import json
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


class BrainAtlas:
    def __init__(self, file_path: str = None):
        self.regions = {}
        self.connection_hints = {}

        if file_path:
            self.load_from_path(file_path)
        else:
            self.load_from_file()

    def load_from_path(self, file_path: str):
        """从给定路径加载脑图谱"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Atlas file not found: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        self._parse_data(data)
        print(f"Loaded atlas from {file_path}")

    def load_from_file(self):
        """通过文件选择器加载脑图谱"""
        root = Tk()
        root.withdraw()
        file_path = askopenfilename(
            title="Select Brain Atlas JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            print("No atlas file selected.")
            root.destroy()
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._parse_data(data)
            print(f"Loaded atlas from {file_path}")
        except Exception as e:
            print(f"Error loading atlas file: {e}")
        finally:
            root.destroy()

    def _parse_data(self, data: dict):
        """解析 atlas JSON 数据"""
        # regions
        if "regions" not in data or not isinstance(data["regions"], dict):
            raise ValueError("Invalid JSON: 'regions' must be a dictionary")

        for region_id, region_data in data["regions"].items():
            pos = np.array(region_data.get("position"), dtype=np.float32)
            if pos.shape != (3,):
                raise ValueError(f"Invalid position format for region {region_id}")
            self.regions[region_id] = {
                "position": pos,
                "function": region_data.get("function", "unknown"),
                "label_id": region_data.get("label_id")
            }

        # connection_hints
        if "connection_hints" in data:
            for key, weight in data["connection_hints"].items():
                src, tgt = key.split("-")
                self.connection_hints[(src, tgt)] = float(weight)

    def get_region_by_label(self, label):
        """根据label_id或名称模糊匹配脑区"""
        for rid, rdata in self.regions.items():
            if str(rdata.get("label_id")) == str(label) or str(label).lower() in str(rdata.get("function", "")).lower():
                return rid, rdata
        return None, None

    def get_region_by_position(self, xyz, tol=3.0):
        """
        根据3D坐标查找最接近的脑区
        tol: 允许的距离阈值（mm）
        """
        xyz = np.array(xyz, dtype=np.float32)
        best_rid, best_dist = None, float("inf")
        for rid, rdata in self.regions.items():
            dist = np.linalg.norm(rdata["position"] - xyz)
            if dist < best_dist and dist <= tol:
                best_rid, best_dist = rid, dist
        return best_rid, self.regions.get(best_rid) if best_rid else (None, None)

    def get_region_id_list(self):
        """返回所有region_id列表"""
        return list(self.regions.keys())
