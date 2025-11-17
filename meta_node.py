import numpy as np
from typing import Dict, Any, Optional, List

class MetaNode:
    """
    元节点模板，支持静态和动态特征。
    动态化适合数字孪生脑仿真。
    """
    def __init__(self, node_id: str, position: np.ndarray,
                 function_label: str, initial_weight: float = 1.0,
                 region_id: Optional[str] = None):
        self.node_id = node_id
        self.position = np.array(position, dtype=np.float32)
        self.function_label = function_label
        self.initial_weight = float(np.clip(initial_weight, 0.0, 1.0))
        self.region_id = region_id

        # 静态/动态属性
        self.features: Dict[str, Any] = {}
        self.dynamic_features: Dict[str, float] = {}
        self.activation_state: float = self.initial_weight
        self.history: List[float] = []

    # ---- 静态特征 ----
    def add_feature(self, key: str, value: Any):
        self.features[key] = value

    def add_features_bulk(self, features: Dict[str, Any]):
        self.features.update(features)

    # ---- 动态状态 ----
    def update_state(self, delta: float, decay: float = 0.0, record_history: bool = True):
        self.activation_state += delta - decay * self.activation_state
        self.activation_state = float(np.clip(self.activation_state, 0.0, 1.0))
        if record_history:
            self.history.append(self.activation_state)

    def step_update(self, delta_dict: Dict[str, float] = None, decay: float = 0.0,
                    record_history: bool = True):
        """
        一次更新多个动态特征
        delta_dict keys 可以是 'activation' 或 dynamic_features 的名称
        """
        if delta_dict is None:
            delta_dict = {}
        # 更新 activation_state
        delta_act = delta_dict.get('activation', 0.0)
        self.update_state(delta_act, decay=decay, record_history=record_history)
        # 更新其他 dynamic_features
        for key, delta in delta_dict.items():
            if key == 'activation':
                continue
            self.dynamic_features[key] = self.dynamic_features.get(key, 0.0) + delta

    def reset_state(self, value: Optional[float] = None):
        self.activation_state = self.initial_weight if value is None else value
        self.history = []

    def set_dynamic_feature(self, key: str, value: float):
        self.dynamic_features[key] = value

    def get_dynamic_feature(self, key: str) -> float:
        return self.dynamic_features.get(key, 0.0)

    # ---- 序列化/克隆 ----
    def to_dict(self, include_state: bool = True) -> Dict[str, Any]:
        data = {
            'node_id': self.node_id,
            'position': self.position.tolist(),
            'function_label': self.function_label,
            'initial_weight': self.initial_weight,
            'region_id': self.region_id,
            'features': self.features,
            'dynamic_features': self.dynamic_features
        }
        if include_state:
            data['activation_state'] = self.activation_state
            data['history'] = self.history
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaNode':
        node = cls(
            node_id=data['node_id'],
            position=np.array(data['position']),
            function_label=data['function_label'],
            initial_weight=data.get('initial_weight', 1.0),
            region_id=data.get('region_id', None)
        )
        node.features = data.get('features', {})
        node.dynamic_features = data.get('dynamic_features', {})
        node.activation_state = data.get('activation_state', node.initial_weight)
        node.history = data.get('history', [])
        return node

    def clone(self, new_id: Optional[str] = None) -> 'MetaNode':
        clone = MetaNode(
            node_id=new_id if new_id else self.node_id,
            position=self.position.copy(),
            function_label=self.function_label,
            initial_weight=self.initial_weight,
            region_id=self.region_id
        )
        clone.features = self.features.copy()
        clone.dynamic_features = self.dynamic_features.copy()
        clone.activation_state = self.activation_state
        clone.history = self.history.copy()
        return clone

    # ---- 可解释性 ----
    def explain(self, show_history: bool = False):
        print(f"[Node {self.node_id}] Region: {self.region_id}, Label: {self.function_label}")
        print(f" Position: {self.position}")
        print(f" InitWeight: {self.initial_weight}, CurrentAct: {self.activation_state}")
        print(f" Features: {self.features}")
        print(f" DynamicFeatures: {self.dynamic_features}")
        if show_history:
            print(f" History: {self.history}")

    # ---- 兼容旧接口 ----
    def get_data(self, include_state: bool = True) -> Dict[str, Any]:
        """
        等价于 self.to_dict()。
        """
        return self.to_dict(include_state=include_state)
