import torch
import torch.nn as nn
import math

from 3_position_embedding import PositionalEmbedding
from 2_mha_block import MultiHeadAttention, PositionwiseFFN

class TransformerWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(d_model, max_len, type='sinusoidal')
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入
        x = self.token_embed(input_ids)
        
        # 添加位置编码
        x = self.pos_embed(x)
        
        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        return x

# 测试 Transformer
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6

model = TransformerWithEmbeddings(vocab_size, d_model, nhead, num_layers)

# 创建测试输入
input_ids = torch.randint(0, vocab_size, (2, 50))  # [batch, seq_len]
attention_mask = torch.ones(2, 50)  # 全1掩码

# 前向传播
output = model(input_ids, attention_mask)
print("Transformer 输出形状:", output.shape)
