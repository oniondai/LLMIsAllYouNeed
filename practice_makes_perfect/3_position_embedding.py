
import torch
import torch.nn as nn
import math

# 正弦位置编码 (Sinusoidal Positional Encoding)
# 正弦位置编码（Sinusoidal，原始 Transformer 使用）
# 可学习位置编码（Learned，BERT 等模型使用）：

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, type='sinusoidal'):
        """
        位置编码模块
        
        参数:
            d_model: 模型维度
            max_len: 最大序列长度
            type: 编码类型 ('sinusoidal' 或 'learned')
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.type = type
        
        if type == 'sinusoidal':
            # 创建正弦位置编码 (不可学习)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
            
        elif type == 'learned':
            # 可学习的位置嵌入
            self.pe = nn.Embedding(max_len, d_model)
            self.register_buffer('position_ids', torch.arange(max_len).expand(1, -1))
        else:
            raise ValueError(f"未知的位置编码类型: {type}")

    def forward(self, x):
        """
        添加位置编码到输入
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        返回:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")
        
        if self.type == 'sinusoidal':
            # 获取前 seq_len 个位置编码
            position_embedding = self.pe[:, :seq_len]
        else:  # learned
            position_ids = self.position_ids[:, :seq_len]
            position_embedding = self.pe(position_ids)
        
        # 添加位置编码到输入
        return x + position_embedding

    def visualize(self, n_dims=64, title=None):
        """
        可视化位置编码 (仅支持正弦编码)
        """
        if self.type != 'sinusoidal':
            print("警告: 可视化仅支持正弦位置编码")
            return
        
        import matplotlib.pyplot as plt
        
        pe = self.pe.squeeze(0).detach().cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pe[:, :n_dims].T, cmap='viridis', aspect='auto')
        plt.xlabel('位置')
        plt.ylabel('维度')
        plt.colorbar(label='编码值')
        plt.title(title or f'位置编码 (前 {n_dims} 维)')
        plt.show()

if __name__ == "__main__":

    # 创建位置编码模块
    d_model = 512
    max_len = 100
    pos_embed = PositionalEmbedding(d_model, max_len, type='sinusoidal')

    # 创建输入 (模拟一批序列)
    batch_size = 2
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)

    # 添加位置编码
    x_with_pos = pos_embed(x)

    print("输入形状:", x.shape)
    print("输出形状:", x_with_pos.shape)
    print("位置编码范数:", torch.norm(x_with_pos - x).item())

    # 可视化正弦位置编码
    pos_embed.visualize(n_dims=64, title="正弦位置编码")

    # 创建并可视化可学习位置编码
    learned_embed = PositionalEmbedding(d_model, max_len, type='learned')
    learned_embed.visualize()  # 会显示警告，但可以尝试可视化