import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super(GraphSAGE, self).__init__()
        # +1 是给 observed bit
        self.conv1 = SAGEConv(input_dim + 1, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim or input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 拼接 observed bit
        if hasattr(data, "node_mask"):
            observed = data.node_mask.float().unsqueeze(1)
        else:
            observed = torch.ones((x.size(0), 1), device=x.device)
        x = torch.cat([x, observed], dim=1)

        # GNN 编码
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 输出层
        out = self.lin(x)
        return out

    def get_node_embeddings(self, data):
        """返回中间层的节点嵌入（不经过输出层），用于 Twin Brain 可视化"""
        x, edge_index = data.x, data.edge_index

        # 拼接 observed bit
        if hasattr(data, "node_mask"):
            observed = data.node_mask.float().unsqueeze(1)
        else:
            observed = torch.ones((x.size(0), 1), device=x.device)
        x = torch.cat([x, observed], dim=1)

        # 提取到 conv2 为止的表征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class GNNTrainer:
    def __init__(self, input_dim: int, hidden_dim: int, num_epochs: int = 200, lr: float = 0.005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphSAGE(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def train(self, data_or_loader):
        self.model.train()

        # 单个 Data
        if hasattr(data_or_loader, "x"):
            data_or_loader = [data_or_loader]

        for epoch in range(self.num_epochs):
            total_loss = 0
            for data in data_or_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)

                # 使用 node_mask 过滤缺失节点
                if hasattr(data, "node_mask"):
                    mask = data.node_mask.bool().squeeze()
                    loss = F.mse_loss(out[mask], data.x[mask])
                else:
                    loss = F.mse_loss(out, data.x)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(data_or_loader):.4f}")

    def predict(self, data, return_embeddings=False):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            if return_embeddings:
                out = self.model.get_node_embeddings(data)
            else:
                out = self.model(data)
        return out.cpu().numpy()

    def save_model(self, path="models/gnn_trainer.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="models/gnn_trainer.pth"):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print(f"Model loaded from {path}")
