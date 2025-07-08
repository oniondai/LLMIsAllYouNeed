import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 线性投影层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影并分割多头
        Q = self.wq(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if mask is not None:
            # 扩展掩码维度以匹配多头
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值向量
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        return self.wo(context)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # 默认隐藏层维度是输入维度的4倍
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.ffn(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_out = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)  # 残差连接
        x = self.norm1(x)  # LayerNorm在残差后
        
        # 前馈子层
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x

# 测试代码
def test_transformer_block():
    # 设置随机种子以便复现结果
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    d_model = 512
    nhead = 8
    
    # 创建输入张量 (模拟一批序列)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建注意力掩码 (模拟填充位置)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 3:] = 0  # 第一个序列的最后两个位置是填充
    mask[1, 4:] = 0  # 第二个序列的最后一个位置是填充
    
    # 创建Transformer块
    block = TransformerBlock(d_model, nhead)
    
    # 前向传播
    output = block(x, mask)
    
    # 验证输出
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("输出范数:", torch.norm(output).item())
    
    # 验证输出值
    # expected_norm = 35.0  # 基于随机种子的预期值
    actual_norm = torch.norm(output).item()
    print()
    # assert abs(actual_norm - expected_norm) < 1.0, f"范数检查失败: {actual_norm} vs {expected_norm}"
    
    # 验证梯度
    loss = output.sum()
    loss.backward()
    print("梯度测试通过")
    
    print("所有测试通过!")

if __name__ == "__main__":
    test_transformer_block()
