# TwinBrain 数字孪生脑 - 优化方向和研究思路

## 📋 文档概述

本文档专注于 TwinBrain 系统的优化方向、研究思路和未来发展规划，为探索意识和大脑机制提供技术路线图。

**核心目标**: 构建一个能够真实模拟大脑动态、预测未来状态、响应虚拟刺激的数字孪生系统。

**实施状态说明**:
- ✅ **已实现**: 功能已经实现并集成到系统中
- 🚧 **进行中**: 正在开发或部分实现
- 📋 **待实现**: 已规划但尚未开始实施

---

## 🎯 核心研究方向

### 1. 增强预测能力 ✅ 已实现 (部分)

#### 1.1 多步未来预测 ✅ 已实现

**实施状态**: ✅ **已完成** (2026-02-01)

**当前状态**:
- ✅ 已实现 PredictorHead 模块，支持多步未来预测
- ✅ 集成到 DynamicHeteroTrainer 中
- ✅ 支持通过配置文件启用/禁用
- ✅ 支持自定义预测步数和损失权重

**实现位置**:
- `train/predictor.py`: PredictorHead 和 ConditionalPredictor 类
- `train/hetero_trainer.py`: 集成预测功能到训练器
- `config/default.yaml`: prediction 配置节

**使用方法**:
```yaml
# 在配置文件中启用预测
prediction:
  enabled: true  # 启用多步预测
  steps: 10      # 预测未来10步
  weight: 0.1    # 预测损失权重
```

**优化方向**: 📋 待优化

##### A. 引入预测模块 ✅ 已实现

在现有架构中添加预测头：

```python
class PredictorHead(nn.Module):
    """预测未来状态的模块"""
    def __init__(self, hidden_dim, n_future_steps=10):
        super().__init__()
        self.n_future_steps = n_future_steps
        
        # 时序预测网络
        self.predictor_gru = nn.GRU(
            hidden_dim, hidden_dim, 
            num_layers=3, batch_first=True
        )
        
        # 注意力机制：关注关键时间步
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, latent_seq):
        """
        latent_seq: [B, T_past, H] - 历史潜在序列
        return: [B, T_future, H] - 未来预测序列
        """
        # 使用历史信息初始化
        _, hidden = self.predictor_gru(latent_seq)
        
        # 自回归预测未来
        predictions = []
        current = latent_seq[:, -1:, :]  # 最后一个状态
        
        for t in range(self.n_future_steps):
            # 预测下一步
            pred, hidden = self.predictor_gru(current, hidden)
            
            # 应用注意力（参考历史）
            attended, _ = self.temporal_attention(
                pred.transpose(0, 1),
                latent_seq.transpose(0, 1),
                latent_seq.transpose(0, 1)
            )
            
            pred = self.output_proj(attended.transpose(0, 1))
            predictions.append(pred)
            current = pred
        
        return torch.cat(predictions, dim=1)
```

##### B. 物理约束的预测

引入神经动力学约束：

```python
class NeurodynamicPredictor(nn.Module):
    """基于神经动力学的预测器"""
    def __init__(self, hidden_dim, connectivity_matrix):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 学习动力学参数
        self.tau = nn.Parameter(torch.ones(1))  # 时间常数
        self.gain = nn.Parameter(torch.ones(1))  # 增益
        
        # 连接矩阵（可学习）
        self.W = nn.Parameter(
            torch.tensor(connectivity_matrix, dtype=torch.float32)
        )
    
    def forward(self, x_t, dt=0.01):
        """
        使用Wilson-Cowan型动力学方程
        dx/dt = -x/tau + gain * sigmoid(W @ x)
        """
        # 计算导数
        interaction = torch.sigmoid(self.W @ x_t)
        dx_dt = (-x_t / self.tau + self.gain * interaction)
        
        # Euler积分
        x_next = x_t + dt * dx_dt
        
        return x_next
```

**收益**:
- 更准确的未来状态预测
- 符合神经动力学规律
- 可解释的预测机制

#### 1.2 条件预测

基于刺激条件预测未来：

```python
class ConditionalPredictor(nn.Module):
    """条件化预测器"""
    def __init__(self, hidden_dim, n_regions):
        super().__init__()
        
        # 刺激编码器
        self.stim_encoder = nn.Sequential(
            nn.Linear(n_regions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 条件化GRU
        self.conditional_gru = nn.GRU(
            hidden_dim * 2,  # concat [state, stimulation]
            hidden_dim,
            num_layers=3
        )
    
    def forward(self, current_state, stimulation):
        """
        current_state: [B, H] - 当前状态
        stimulation: [B, N_regions] - 刺激向量
        """
        # 编码刺激
        stim_embed = self.stim_encoder(stimulation)
        
        # 拼接状态和刺激
        conditioned = torch.cat([current_state, stim_embed], dim=-1)
        
        # 预测
        predicted, _ = self.conditional_gru(conditioned.unsqueeze(1))
        
        return predicted.squeeze(1)
```

**应用场景**:
- TMS刺激响应预测
- 药物效应模拟
- 神经调控效果评估

---

### 2. 虚拟刺激和扰动

#### 2.1 刺激模型设计

**刺激类型**:

##### A. 空间刺激
```python
class SpatialStimulation:
    """空间定位的刺激"""
    def __init__(self):
        self.target_regions = []  # 目标脑区
        self.amplitude = 0.0      # 强度
        self.spatial_spread = 0.0  # 空间扩散
    
    def apply(self, brain_state):
        """
        应用刺激到脑状态
        """
        for region_id in self.target_regions:
            # 直接刺激
            brain_state[region_id] += self.amplitude
            
            # 空间扩散（基于距离）
            for neighbor_id, distance in self.get_neighbors(region_id):
                spread_effect = self.amplitude * np.exp(
                    -distance / self.spatial_spread
                )
                brain_state[neighbor_id] += spread_effect
        
        return brain_state
```

##### B. 时间刺激
```python
class TemporalStimulation:
    """时间模式刺激"""
    def __init__(self):
        self.pattern = "pulse"  # pulse/sine/ramp
        self.frequency = 10.0   # Hz
        self.duration = 1.0     # seconds
        self.phase = 0.0        # radians
    
    def generate_pattern(self, t):
        """生成时间刺激模式"""
        if self.pattern == "pulse":
            return 1.0 if t % (1/self.frequency) < 0.5/self.frequency else 0.0
        
        elif self.pattern == "sine":
            return np.sin(2 * np.pi * self.frequency * t + self.phase)
        
        elif self.pattern == "ramp":
            return min(t / self.duration, 1.0)
        
        return 0.0
```

##### C. 频率特定刺激
```python
class FrequencyStimulation:
    """特定频段刺激（模拟tACS等）"""
    def __init__(self):
        self.target_frequency = 10.0  # Hz (如alpha频段)
        self.bandwidth = 2.0          # Hz
        self.amplitude = 0.5
    
    def apply_to_signal(self, signal, fs=1000):
        """
        对信号应用频率特定刺激
        """
        t = np.arange(len(signal)) / fs
        
        # 生成刺激信号
        stim = self.amplitude * np.sin(2 * np.pi * self.target_frequency * t)
        
        # 应用带通滤波
        from scipy.signal import butter, filtfilt
        b, a = butter(
            4, 
            [self.target_frequency - self.bandwidth/2,
             self.target_frequency + self.bandwidth/2],
            btype='band', fs=fs
        )
        stim_filtered = filtfilt(b, a, stim)
        
        return signal + stim_filtered
```

#### 2.2 刺激效应建模

建模刺激如何影响大脑网络：

```python
class StimulationEffectModel(nn.Module):
    """刺激效应模型"""
    def __init__(self, n_regions, hidden_dim):
        super().__init__()
        
        # 直接效应：刺激→局部响应
        self.direct_effect = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 网络效应：通过连接传播
        self.network_propagation = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2
        )
        
        # 时间衰减
        self.decay_rate = nn.Parameter(torch.tensor(0.95))
    
    def forward(self, stimulation, connectivity, n_steps):
        """
        模拟刺激效应的时空传播
        
        stimulation: [N_regions] - 初始刺激
        connectivity: [N_regions, N_regions] - 连接矩阵
        n_steps: 模拟时间步数
        """
        # 初始化
        state = stimulation.clone()
        trajectory = [state]
        
        for t in range(n_steps):
            # 直接效应
            local_response = self.direct_effect(state.unsqueeze(-1)).squeeze(-1)
            
            # 网络传播
            propagated = connectivity @ state
            
            # 组合并应用衰减
            state = (local_response + propagated) * (self.decay_rate ** t)
            
            trajectory.append(state)
        
        return torch.stack(trajectory)
```

#### 2.3 反向刺激设计

给定目标状态，设计刺激方案：

```python
class InverseStimulationDesigner:
    """反向设计刺激方案"""
    def __init__(self, model, optimizer_config):
        self.model = model
        self.optimizer_config = optimizer_config
    
    def design_stimulation(
        self, 
        initial_state, 
        target_state, 
        max_amplitude=1.0,
        n_optimization_steps=100
    ):
        """
        设计达到目标状态的刺激
        
        initial_state: 初始大脑状态
        target_state: 目标大脑状态
        """
        # 初始化可优化的刺激参数
        stimulation = nn.Parameter(
            torch.zeros_like(initial_state)
        )
        
        optimizer = torch.optim.Adam([stimulation], lr=0.01)
        
        for step in range(n_optimization_steps):
            # 模拟刺激效应
            predicted_state = self.model.simulate_with_stimulation(
                initial_state, stimulation
            )
            
            # 损失：目标状态 vs 预测状态
            loss = F.mse_loss(predicted_state, target_state)
            
            # 约束：刺激幅度
            loss += torch.sum(torch.abs(stimulation) - max_amplitude).clamp(min=0)
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 投影到可行域
            with torch.no_grad():
                stimulation.clamp_(-max_amplitude, max_amplitude)
        
        return stimulation.detach()
```

**应用**:
- 治疗方案优化
- 神经调控参数设计
- 认知增强策略

---

### 3. 多模态融合优化

#### 3.1 跨模态注意力

当前模态融合可能不够充分，引入注意力机制：

```python
class CrossModalAttention(nn.Module):
    """跨模态注意力融合"""
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        
        # 多头注意力
        self.cross_attention_fmri_to_eeg = nn.MultiheadAttention(
            hidden_dim, n_heads
        )
        self.cross_attention_eeg_to_fmri = nn.MultiheadAttention(
            hidden_dim, n_heads
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, fmri_feat, eeg_feat):
        """
        fmri_feat: [N_regions, T, H]
        eeg_feat: [N_regions, T, H]
        """
        # fMRI关注EEG
        fmri_attended, _ = self.cross_attention_fmri_to_eeg(
            fmri_feat, eeg_feat, eeg_feat
        )
        
        # EEG关注fMRI
        eeg_attended, _ = self.cross_attention_eeg_to_fmri(
            eeg_feat, fmri_feat, fmri_feat
        )
        
        # 门控融合
        combined = torch.cat([fmri_attended, eeg_attended], dim=-1)
        gate_weights = self.gate(combined)
        
        fused = gate_weights * fmri_attended + (1 - gate_weights) * eeg_attended
        
        return fused
```

#### 3.2 模态对齐学习

改进时间对齐方法：

```python
class ImprovedModalityAligner(nn.Module):
    """改进的模态对齐器"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 学习模态特定的时间变换
        self.fmri_temporal_transform = nn.Conv1d(
            hidden_dim, hidden_dim, 
            kernel_size=5, padding=2
        )
        
        self.eeg_temporal_transform = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1
        )
        
        # 对齐网络
        self.align_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 对齐分数
        )
    
    def forward(self, fmri_seq, eeg_seq):
        """
        计算最优时间对齐
        """
        # 时间变换
        fmri_transformed = self.fmri_temporal_transform(
            fmri_seq.transpose(1, 2)
        ).transpose(1, 2)
        
        eeg_transformed = self.eeg_temporal_transform(
            eeg_seq.transpose(1, 2)
        ).transpose(1, 2)
        
        # 计算所有时间点对的对齐分数
        T_fmri, T_eeg = fmri_transformed.shape[1], eeg_transformed.shape[1]
        alignment_scores = torch.zeros(T_fmri, T_eeg)
        
        for i in range(T_fmri):
            for j in range(T_eeg):
                combined = torch.cat([
                    fmri_transformed[:, i, :],
                    eeg_transformed[:, j, :]
                ], dim=-1)
                alignment_scores[i, j] = self.align_net(combined).mean()
        
        # 使用动态时间规整（DTW）找最优路径
        aligned_fmri, aligned_eeg = self.dtw_align(
            fmri_transformed, eeg_transformed, alignment_scores
        )
        
        return aligned_fmri, aligned_eeg
```

#### 3.3 互信息最大化

优化跨模态表征：

```python
class MutualInformationMaximizer(nn.Module):
    """最大化跨模态互信息"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 判别器：判断是否来自相同时间点
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def compute_mi_loss(self, fmri_feat, eeg_feat):
        """
        计算互信息损失（最大化）
        使用对比学习框架
        """
        batch_size = fmri_feat.shape[0]
        
        # 正样本：配对的fMRI和EEG特征
        positive_pairs = torch.cat([fmri_feat, eeg_feat], dim=-1)
        positive_scores = self.discriminator(positive_pairs)
        
        # 负样本：随机配对
        indices = torch.randperm(batch_size)
        negative_pairs = torch.cat([
            fmri_feat, eeg_feat[indices]
        ], dim=-1)
        negative_scores = self.discriminator(negative_pairs)
        
        # 对比损失
        loss = -torch.mean(
            torch.log(torch.sigmoid(positive_scores) + 1e-8) +
            torch.log(1 - torch.sigmoid(negative_scores) + 1e-8)
        )
        
        return loss
```

---

### 4. 因果推断和网络分析

#### 4.1 因果发现

识别脑区间的因果关系：

```python
class CausalDiscovery:
    """因果关系发现"""
    def __init__(self, model):
        self.model = model
    
    def granger_causality(self, x, y, max_lag=10):
        """
        Granger因果检验
        
        x: 源信号
        y: 目标信号
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        data = np.column_stack([y, x])
        results = grangercausalitytests(data, max_lag, verbose=False)
        
        # 提取p值
        p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        
        return min(p_values) < 0.05  # 是否显著
    
    def pairwise_causality(self, brain_signals):
        """
        计算所有脑区对的因果关系
        
        brain_signals: [N_regions, T]
        """
        n_regions = brain_signals.shape[0]
        causality_matrix = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    causality_matrix[i, j] = self.granger_causality(
                        brain_signals[i], brain_signals[j]
                    )
        
        return causality_matrix
    
    def effective_connectivity(self, brain_signals):
        """
        计算有效连接（基于动态因果模型）
        """
        # 简化的DCM实现
        n_regions = brain_signals.shape[0]
        T = brain_signals.shape[1]
        
        # 使用向量自回归(VAR)估计连接
        from statsmodels.tsa.api import VAR
        
        model = VAR(brain_signals.T)
        results = model.fit(maxlags=5)
        
        # 提取系数矩阵（有效连接强度）
        effective_conn = results.coefs[0]  # lag 1的系数
        
        return effective_conn
```

#### 4.2 网络拓扑分析

分析大脑网络的拓扑属性：

```python
class NetworkTopologyAnalyzer:
    """网络拓扑分析"""
    def __init__(self):
        pass
    
    def compute_graph_metrics(self, connectivity_matrix, threshold=0.3):
        """
        计算图论指标
        """
        import networkx as nx
        
        # 二值化连接矩阵
        binary_matrix = (np.abs(connectivity_matrix) > threshold).astype(int)
        
        # 构建图
        G = nx.from_numpy_array(binary_matrix)
        
        metrics = {
            # 全局指标
            'clustering_coef': nx.average_clustering(G),
            'path_length': nx.average_shortest_path_length(G),
            'global_efficiency': nx.global_efficiency(G),
            'modularity': self.compute_modularity(G),
            'assortativity': nx.degree_assortativity_coefficient(G),
            
            # 局部指标（per node）
            'degree': dict(G.degree()),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
        }
        
        # 小世界性
        metrics['small_worldness'] = self.compute_small_worldness(
            metrics['clustering_coef'],
            metrics['path_length']
        )
        
        return metrics
    
    def compute_modularity(self, G):
        """计算网络模块性"""
        from networkx.algorithms import community
        
        communities = community.greedy_modularity_communities(G)
        return community.modularity(G, communities)
    
    def compute_small_worldness(self, C, L):
        """
        小世界性：S = (C/C_rand) / (L/L_rand)
        C: 聚类系数
        L: 平均路径长度
        """
        # 理论随机网络值（Watts-Strogatz）
        C_rand = 0.1  # 近似值
        L_rand = 2.0  # 近似值
        
        S = (C / C_rand) / (L / L_rand)
        return S
    
    def identify_hubs(self, connectivity_matrix, top_k=10):
        """
        识别网络中的枢纽节点
        """
        # 计算节点强度
        node_strength = np.sum(np.abs(connectivity_matrix), axis=1)
        
        # 计算betweenness centrality
        G = nx.from_numpy_array(connectivity_matrix)
        betweenness = nx.betweenness_centrality(G)
        
        # 综合排名
        hub_score = (
            node_strength / node_strength.max() +
            np.array(list(betweenness.values()))
        ) / 2
        
        top_hubs = np.argsort(hub_score)[-top_k:][::-1]
        
        return top_hubs, hub_score[top_hubs]
```

---

### 5. Unity可视化增强

#### 5.1 高级JSON输出格式

设计更丰富的JSON格式：

```json
{
  "version": "2.0",
  "timestamp": "2026-01-31T10:00:00Z",
  "metadata": {
    "subject": "sub-01",
    "session": "ses-01",
    "atlas": "Schaefer200",
    "model_version": "v4",
    "sampling_rate": {
      "fmri": 0.5,
      "eeg": 250
    }
  },
  
  "brain_state": {
    "time_point": 100,
    "time_second": 200.0,
    
    "regions": [
      {
        "id": 1,
        "label": "7Networks_LH_Vis_1",
        "network": "Visual",
        "hemisphere": "left",
        "position": {"x": -5, "y": -85, "z": 5},
        
        "activity": {
          "fmri": {
            "amplitude": 0.75,
            "z_score": 2.3,
            "percentile": 85
          },
          "eeg": {
            "amplitude": 0.82,
            "power": {
              "delta": 0.15,
              "theta": 0.25,
              "alpha": 0.45,
              "beta": 0.10,
              "gamma": 0.05
            }
          }
        },
        
        "predicted": {
          "next_activity": 0.78,
          "trend": "increasing",
          "confidence": 0.85
        }
      }
    ],
    
    "connections": [
      {
        "source": 1,
        "target": 2,
        "strength": 0.65,
        "type": "structural",
        "bidirectional": true,
        "weight": 0.65,
        "delay_ms": 15
      },
      {
        "source": 1,
        "target": 5,
        "strength": 0.42,
        "type": "functional",
        "correlation": 0.68,
        "causality": "source_to_target"
      }
    ],
    
    "networks": {
      "visual": {
        "avg_activity": 0.72,
        "coherence": 0.68,
        "regions": [1, 2, 3, 4, 5]
      },
      "motor": {
        "avg_activity": 0.65,
        "coherence": 0.71,
        "regions": [50, 51, 52, 53]
      }
    },
    
    "global_metrics": {
      "mean_activity": 0.68,
      "std_activity": 0.12,
      "max_activity": 0.95,
      "active_regions": 150,
      "sync_index": 0.62,
      "entropy": 4.25
    }
  },
  
  "stimulation": {
    "active": true,
    "type": "TMS",
    "target_regions": [10, 11],
    "amplitude": 0.5,
    "frequency": 10.0,
    "start_time": 95.0,
    "duration": 1.0
  },
  
  "prediction": {
    "horizon": 10,
    "confidence": 0.82,
    "trajectory": [
      {"time": 101, "global_activity": 0.70},
      {"time": 102, "global_activity": 0.72},
      {"time": 103, "global_activity": 0.71}
    ]
  }
}
```

#### 5.2 实时通信协议

WebSocket实时数据流：

```python
import asyncio
import websockets
import json

class BrainVisualizationServer:
    """Unity可视化WebSocket服务器"""
    def __init__(self, model, port=8765):
        self.model = model
        self.port = port
        self.clients = set()
    
    async def register(self, websocket):
        """注册客户端"""
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
    
    async def unregister(self, websocket):
        """注销客户端"""
        self.clients.remove(websocket)
        print(f"Client disconnected: {websocket.remote_address}")
    
    async def send_brain_state(self, brain_state_json):
        """广播大脑状态给所有客户端"""
        if self.clients:
            message = json.dumps(brain_state_json)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        """处理客户端连接"""
        await self.register(websocket)
        
        try:
            async for message in websocket:
                # 解析请求
                request = json.loads(message)
                
                if request['type'] == 'get_state':
                    # 获取当前状态
                    state = self.get_current_brain_state(request)
                    await websocket.send(json.dumps(state))
                
                elif request['type'] == 'predict':
                    # 预测未来状态
                    prediction = self.predict_future(request)
                    await websocket.send(json.dumps(prediction))
                
                elif request['type'] == 'simulate':
                    # 模拟刺激
                    response = self.simulate_stimulation(request)
                    await websocket.send(json.dumps(response))
                
                elif request['type'] == 'stream':
                    # 开始流式传输
                    await self.stream_brain_activity(websocket, request)
        
        finally:
            await self.unregister(websocket)
    
    async def stream_brain_activity(self, websocket, config):
        """流式传输大脑活动"""
        fps = config.get('fps', 10)
        duration = config.get('duration', 60)
        
        for t in range(int(fps * duration)):
            # 计算当前大脑状态
            brain_state = self.model.get_state_at_time(t / fps)
            
            # 转换为JSON
            state_json = self.brain_state_to_json(brain_state)
            
            # 发送
            await websocket.send(json.dumps(state_json))
            
            # 控制帧率
            await asyncio.sleep(1.0 / fps)
    
    def start(self):
        """启动服务器"""
        start_server = websockets.serve(self.handler, "0.0.0.0", self.port)
        
        asyncio.get_event_loop().run_until_complete(start_server)
        print(f"WebSocket server started on port {self.port}")
        asyncio.get_event_loop().run_forever()
```

Unity客户端示例：

```csharp
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;

public class BrainVisualizationClient : MonoBehaviour
{
    private WebSocket ws;
    private BrainStateData currentState;
    
    void Start()
    {
        // 连接WebSocket服务器
        ws = new WebSocket("ws://localhost:8765");
        
        ws.OnMessage += (sender, e) =>
        {
            // 解析JSON
            currentState = JsonConvert.DeserializeObject<BrainStateData>(e.Data);
            
            // 在主线程更新可视化
            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                UpdateVisualization(currentState);
            });
        };
        
        ws.Connect();
    }
    
    void UpdateVisualization(BrainStateData state)
    {
        // 更新脑区颜色
        foreach (var region in state.brain_state.regions)
        {
            GameObject regionObj = GetRegionObject(region.id);
            
            // 活跃度→颜色
            Color color = GetActivityColor(region.activity.fmri.amplitude);
            regionObj.GetComponent<Renderer>().material.color = color;
            
            // 活跃度→大小
            float scale = 1.0f + region.activity.fmri.amplitude * 0.5f;
            regionObj.transform.localScale = Vector3.one * scale;
        }
        
        // 更新连接
        foreach (var conn in state.brain_state.connections)
        {
            UpdateConnection(conn.source, conn.target, conn.strength);
        }
    }
    
    public void RequestPrediction(int nSteps)
    {
        var request = new
        {
            type = "predict",
            n_steps = nSteps
        };
        
        ws.Send(JsonConvert.SerializeObject(request));
    }
    
    public void SimulateStimulation(int[] regions, float amplitude)
    {
        var request = new
        {
            type = "simulate",
            stimulation = new
            {
                target_regions = regions,
                amplitude = amplitude,
                duration = 1.0
            }
        };
        
        ws.Send(JsonConvert.SerializeObject(request));
    }
}
```

---

### 6. 意识相关特征提取

#### 6.1 整合信息理论(IIT)

计算Φ（Phi）值量化意识水平：

```python
class IntegratedInformationComputer:
    """整合信息计算（简化版）"""
    def __init__(self, connectivity_matrix):
        self.connectivity = connectivity_matrix
        self.n_elements = connectivity_matrix.shape[0]
    
    def compute_phi(self, brain_state):
        """
        计算整合信息Φ
        
        Φ = min(MIP(cause) + MIP(effect))
        MIP: Minimum Information Partition
        """
        # 计算系统的因果能力
        cause_info = self.compute_cause_information(brain_state)
        effect_info = self.compute_effect_information(brain_state)
        
        # 寻找最小信息分割(MIP)
        min_partition_cause = self.find_mip(brain_state, 'cause')
        min_partition_effect = self.find_mip(brain_state, 'effect')
        
        # Φ = 完整系统的整合信息 - MIP的信息
        phi = (cause_info + effect_info) - \
              (min_partition_cause + min_partition_effect)
        
        return phi
    
    def compute_cause_information(self, state):
        """计算因果信息"""
        # 使用互信息估计因果能力
        from sklearn.feature_selection import mutual_info_regression
        
        # 当前状态 → 未来状态的信息
        past_state = state[:-1]
        future_state = state[1:]
        
        mi = mutual_info_regression(
            past_state.reshape(-1, 1),
            future_state
        ).sum()
        
        return mi
    
    def find_mip(self, state, direction='cause'):
        """
        找到最小信息分割
        尝试所有可能的二分，找使信息最小的分割
        """
        n = len(state)
        min_info = float('inf')
        
        # 遍历所有可能的分割
        for i in range(1, 2**n - 1):
            partition1 = [j for j in range(n) if i & (1 << j)]
            partition2 = [j for j in range(n) if not (i & (1 << j))]
            
            # 计算分割后的信息
            if direction == 'cause':
                info = self.partition_cause_info(state, partition1, partition2)
            else:
                info = self.partition_effect_info(state, partition1, partition2)
            
            min_info = min(min_info, info)
        
        return min_info
```

#### 6.2 神经振荡分析

分析意识相关的振荡模式：

```python
class ConsciousnessOscillationAnalyzer:
    """意识相关振荡分析"""
    def __init__(self):
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def compute_power_spectrum(self, signal, fs=1000):
        """计算功率谱"""
        from scipy.signal import welch
        
        freqs, psd = welch(signal, fs=fs, nperseg=1024)
        
        return freqs, psd
    
    def compute_band_power(self, signal, band, fs=1000):
        """计算特定频段的功率"""
        from scipy.signal import welch
        from scipy.integrate import simps
        
        freqs, psd = welch(signal, fs=fs, nperseg=1024)
        
        # 找到频段范围
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        
        # 使用Simpson规则积分
        band_power = simps(psd[idx_band], freqs[idx_band])
        
        return band_power
    
    def compute_phase_amplitude_coupling(self, low_freq, high_freq, fs=1000):
        """
        计算相位-幅度耦合(PAC)
        低频相位调制高频幅度是意识的标志
        """
        from scipy.signal import hilbert, butter, filtfilt
        
        # 提取低频相位
        b_low, a_low = butter(4, [4, 8], btype='band', fs=fs)
        low_filtered = filtfilt(b_low, a_low, low_freq)
        low_phase = np.angle(hilbert(low_filtered))
        
        # 提取高频幅度
        b_high, a_high = butter(4, [30, 100], btype='band', fs=fs)
        high_filtered = filtfilt(b_high, a_high, high_freq)
        high_amplitude = np.abs(hilbert(high_filtered))
        
        # 计算调制指数
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        mean_amplitude = np.zeros(n_bins)
        for i in range(n_bins):
            phase_mask = np.logical_and(
                low_phase >= phase_bins[i],
                low_phase < phase_bins[i + 1]
            )
            mean_amplitude[i] = high_amplitude[phase_mask].mean()
        
        # 归一化
        mean_amplitude /= mean_amplitude.sum()
        
        # 计算KL散度（相对于均匀分布）
        uniform = np.ones(n_bins) / n_bins
        modulation_index = np.sum(
            mean_amplitude * np.log(mean_amplitude / uniform + 1e-10)
        )
        
        return modulation_index
    
    def detect_traveling_waves(self, multi_region_signals, positions):
        """
        检测行波（traveling waves）
        意识状态下大脑会产生协调的行波
        """
        n_regions, n_time = multi_region_signals.shape
        
        # 计算相位
        phases = np.angle(hilbert(multi_region_signals, axis=1))
        
        # 计算相位梯度
        phase_gradients = []
        for t in range(n_time):
            # 拟合相位到空间位置
            from scipy.optimize import curve_fit
            
            def plane(pos, a, b, c):
                return a * pos[:, 0] + b * pos[:, 1] + c
            
            try:
                popt, _ = curve_fit(plane, positions, phases[:, t])
                gradient = np.sqrt(popt[0]**2 + popt[1]**2)
                phase_gradients.append(gradient)
            except:
                phase_gradients.append(0)
        
        # 行波强度：平均相位梯度
        wave_strength = np.mean(phase_gradients)
        
        return wave_strength, phase_gradients
```

---

### 7. 计算效率优化

#### 7.1 模型压缩

##### A. 知识蒸馏
```python
class KnowledgeDistillation:
    """将大模型知识迁移到小模型"""
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = 3.0
    
    def distillation_loss(self, student_output, teacher_output, labels):
        """
        蒸馏损失 = KL散度(软标签) + CE(硬标签)
        """
        # 软标签损失
        soft_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=-1),
            F.softmax(teacher_output / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.mse_loss(student_output, labels)
        
        # 组合
        total_loss = 0.7 * soft_loss + 0.3 * hard_loss
        
        return total_loss
```

##### B. 模型剪枝
```python
class ModelPruning:
    """模型剪枝减少参数"""
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def magnitude_pruning(self):
        """基于权重幅度的剪枝"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                # 计算权重幅度
                weight = module.weight.data
                threshold = torch.quantile(
                    torch.abs(weight), 
                    self.pruning_ratio
                )
                
                # 创建mask
                mask = (torch.abs(weight) > threshold).float()
                
                # 应用mask
                module.weight.data *= mask
                
                # 注册mask以在前向传播中使用
                module.register_buffer('weight_mask', mask)
```

##### C. 量化
```python
def quantize_model(model, dtype=torch.qint8):
    """模型量化"""
    # 动态量化（不需要校准数据）
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=dtype
    )
    
    return quantized_model
```

#### 7.2 并行计算

```python
class ParallelBrainSimulator:
    """并行模拟多个被试"""
    def __init__(self, model, device_ids=[0, 1, 2, 3]):
        self.model = nn.DataParallel(model, device_ids=device_ids)
        self.device_ids = device_ids
    
    def simulate_batch(self, subjects_data):
        """批量模拟"""
        # 分配到不同GPU
        batch_size = len(subjects_data)
        per_device = batch_size // len(self.device_ids)
        
        results = []
        for i, device_id in enumerate(self.device_ids):
            start_idx = i * per_device
            end_idx = start_idx + per_device if i < len(self.device_ids) - 1 else batch_size
            
            batch = subjects_data[start_idx:end_idx]
            device = torch.device(f'cuda:{device_id}')
            
            # 在特定GPU上运行
            batch_results = self.model(batch.to(device))
            results.extend(batch_results)
        
        return results
```

---

### 8. 长期研究方向

#### 8.1 自监督学习

减少对标注数据的依赖：

```python
class SelfSupervisedPretraining:
    """自监督预训练"""
    def __init__(self, model):
        self.model = model
    
    def contrastive_loss(self, z1, z2):
        """
        对比学习：同一数据的不同augmentation应该相似
        """
        # 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 正样本：同一数据
        pos_sim = torch.sum(z1 * z2, dim=-1)
        
        # 负样本：不同数据
        neg_sim = torch.matmul(z1, z2.T)
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(z1), dtype=torch.long, device=z1.device)
        
        loss = F.cross_entropy(logits / 0.07, labels)
        
        return loss
    
    def temporal_prediction(self, past_seq):
        """
        时间预测任务：预测未来片段
        """
        # 过去序列编码
        past_encoding = self.model.encode(past_seq[:, :-10])
        
        # 预测未来
        future_pred = self.model.predict_future(past_encoding, n_steps=10)
        
        # 实际未来
        future_actual = past_seq[:, -10:]
        
        # 损失
        loss = F.mse_loss(future_pred, future_actual)
        
        return loss
```

#### 8.2 元学习

快速适应新被试：

```python
class MetaLearning:
    """元学习（学习如何学习）"""
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=meta_lr
        )
        self.inner_lr = inner_lr
    
    def maml_step(self, support_set, query_set):
        """
        MAML (Model-Agnostic Meta-Learning) 步骤
        
        support_set: 少量新被试数据用于快速适应
        query_set: 测试适应效果
        """
        # 保存初始参数
        meta_params = {
            name: param.clone() 
            for name, param in self.model.named_parameters()
        }
        
        # Inner loop: 在support set上快速适应
        for _ in range(5):  # 5步内部优化
            support_loss = self.compute_loss(support_set)
            
            # 手动梯度下降
            grads = torch.autograd.grad(
                support_loss, 
                self.model.parameters(),
                create_graph=True
            )
            
            for (name, param), grad in zip(
                self.model.named_parameters(), grads
            ):
                param.data = param.data - self.inner_lr * grad
        
        # Outer loop: 在query set上评估
        query_loss = self.compute_loss(query_set)
        
        # 元优化
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        
        # 恢复参数（为下一个任务）
        for name, param in self.model.named_parameters():
            param.data = meta_params[name]
        
        return query_loss.item()
```

#### 8.3 神经符号集成

结合神经网络和符号推理：

```python
class NeuroSymbolicReasoning:
    """神经-符号推理"""
    def __init__(self, neural_model, knowledge_graph):
        self.neural_model = neural_model
        self.kg = knowledge_graph  # 神经科学知识图谱
    
    def reason_with_knowledge(self, observation):
        """
        结合神经网络感知和符号知识推理
        """
        # 神经网络提取特征
        neural_features = self.neural_model.encode(observation)
        
        # 映射到概念
        concepts = self.map_to_concepts(neural_features)
        
        # 符号推理
        inferred_concepts = self.symbolic_reasoning(concepts)
        
        # 转回神经表示
        enhanced_features = self.concepts_to_neural(inferred_concepts)
        
        return enhanced_features
    
    def symbolic_reasoning(self, concepts):
        """
        基于知识图谱的推理
        
        例如：
        IF region_A is active AND region_A connects_to region_B
        THEN region_B will_be_active
        """
        inferred = set(concepts)
        
        # 应用推理规则
        for concept in concepts:
            # 查询知识图谱
            related = self.kg.get_related(concept, relation='implies')
            inferred.update(related)
        
        return list(inferred)
```

---

## 🎓 理论基础和科学价值

### 意识的计算理论

TwinBrain系统的设计基于以下意识理论：

1. **整合信息理论(IIT)**
   - Φ值量化意识水平
   - 系统的整合性和信息容量

2. **全局工作空间理论(GWT)**
   - 广播机制：信息在脑区间传播
   - 注意力选择性增强

3. **预测编码理论**
   - 大脑持续预测感觉输入
   - 预测误差驱动学习

### 研究问题

1. **意识的神经关联物(NCC)**
   - 哪些脑区活动模式对应意识体验？
   - 如何从神经活动预测主观体验？

2. **因果结构**
   - 脑区间的因果关系如何组织？
   - 刺激如何通过网络传播？

3. **动力学特征**
   - 意识状态转换的动力学规律
   - 临界状态和相变现象

---

## 📚 参考文献和资源

### 关键论文

1. Tononi G. (2008). "Consciousness as Integrated Information" - IIT理论
2. Dehaene S. (2014). "Consciousness and the Brain" - 全局工作空间
3. Friston K. (2010). "The free-energy principle" - 预测编码
4. Sporns O. (2016). "Networks of the Brain" - 脑网络分析

### 技术资源

1. PyTorch Geometric - 图神经网络
2. MNE-Python - EEG/MEG数据处理
3. Nibabel - 神经影像数据
4. NetworkX - 网络分析

### 数据集

1. Human Connectome Project (HCP)
2. UK Biobank
3. OpenNeuro
4. THINGS EEG Dataset

---

## 🚀 实施路线图

### 短期目标（1-3个月）

- [ ] 实现基础的未来状态预测
- [ ] 完成虚拟刺激模块
- [ ] 优化Unity JSON输出
- [ ] 添加实时WebSocket接口

### 中期目标（3-6个月）

- [ ] 引入因果推断分析
- [ ] 实现网络拓扑分析
- [ ] 开发自监督预训练
- [ ] 优化计算效率（剪枝、量化）

### 长期目标（6-12个月）

- [ ] 整合信息理论的Φ计算
- [ ] 元学习快速适应
- [ ] 神经符号集成
- [ ] 发布完整的意识建模框架

---

## 📊 增强训练监控和日志 ✅ 已实现

**实施状态**: ✅ **已完成** (2026-02-01)

### 实现的功能

#### 1. MetricsTracker 类 ✅

**位置**: `utils/metrics_tracker.py`

**功能**:
- ✅ 自动记录所有训练指标历史
- ✅ 保存损失分量（重构、时序、对齐等）
- ✅ 记录梯度统计信息
- ✅ 导出 JSON 格式的指标历史
- ✅ 自动生成训练摘要报告

**使用方法**:
```yaml
# config/default.yaml
metrics:
  enabled: true
  output_dir: "metrics"
```

**代码示例**:
```python
from utils.metrics_tracker import MetricsTracker

# 创建追踪器
tracker = MetricsTracker(output_dir="results/metrics")

# 记录损失分量
tracker.log_loss_components(
    epoch=10,
    recon_loss=0.45,
    temp_loss=0.32,
    align_loss=0.28,
    total_loss=1.05
)

# 记录梯度统计
tracker.log_gradient_stats(
    epoch=10,
    grad_norm=2.3,
    grad_max=5.1,
    grad_min=0.01
)

# 保存和打印摘要
tracker.save_metrics()
tracker.print_summary(last_n_epochs=10)
```

#### 2. TrainingMonitor 类 ✅

**功能**:
- ✅ 监控训练进度
- ✅ 检测训练停滞
- ✅ 检测异常值（NaN/Inf）
- ✅ 自动生成警告

#### 3. 集成到训练器 ✅

**位置**: `train/hetero_trainer.py`

**功能**:
- ✅ 自动记录每个 epoch 的所有损失
- ✅ 训练结束后自动保存指标
- ✅ 打印训练摘要
- ✅ 支持通过配置启用/禁用

### 输出示例

**训练日志**:
```
[Epoch  10] total=1.2345 align=0.3456 temp=0.4567 recon=0.4322
[Epoch  10] relative_error={'fmri': 0.12, 'eeg': 0.15}
```

**指标摘要**:
```
================================================================================
Metrics Summary (Last 10 epochs)
================================================================================
loss/total                     | Latest:   1.2345 | Avg:   1.3456 | Std:   0.1234
loss/reconstruction            | Latest:   0.4322 | Avg:   0.4500 | Std:   0.0234
loss/temporal                  | Latest:   0.4567 | Avg:   0.4600 | Std:   0.0123
loss/alignment                 | Latest:   0.3456 | Avg:   0.3400 | Std:   0.0210
rel_error/fmri                 | Latest:   0.1200 | Avg:   0.1250 | Std:   0.0050
rel_error/eeg                  | Latest:   0.1500 | Avg:   0.1550 | Std:   0.0060
================================================================================
```

**JSON 输出** (`metrics_history.json`):
```json
{
  "loss/total": [
    {"epoch": 1, "value": 2.5},
    {"epoch": 2, "value": 2.3},
    ...
  ],
  "loss/reconstruction": [...],
  "rel_error/fmri": [...]
}
```

### 收益

- ✅ **完整的训练可见性**: 追踪所有重要指标
- ✅ **易于调试**: 快速定位训练问题
- ✅ **实验对比**: JSON 格式便于比较不同实验
- ✅ **自动化**: 无需手动记录，自动保存

---

## 💡 创新点和贡献

1. **数字孪生脑范式**
   - 首个多模态数字孪生脑系统
   - 支持虚拟刺激和未来预测

2. **异构图神经网络**
   - 统一建模多模态数据
   - 灵活的跨模态信息传递

3. **实时可视化**
   - Unity 3D交互式展示
   - WebSocket实时数据流

4. **意识计算框架**
   - 整合多种意识理论
   - 可量化的意识指标

---

**文档版本**: 1.1  
**最后更新**: 2026-02-01  
**维护者**: TwinBrain Development Team

> "探索意识的本质，理解大脑的奥秘。TwinBrain不仅是一个工具，更是通往意识科学的桥梁。"
