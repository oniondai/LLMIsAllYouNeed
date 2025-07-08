import torch
import torch.nn as nn


# https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003
# 旋转位置编码 (RoPE)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 预计算旋转矩阵
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        position = torch.arange(max_len).float()
        sinusoid = torch.einsum('i,j->ij', position, inv_freq)
        
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        
        # 缓存旋转矩阵
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)
    
    def rotate_half(self, x):
        """将张量的后半部分旋转"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x, seq_dim=1):
        """
        应用旋转位置编码
        
        参数:
            x: 输入张量 [..., seq_len, dim]
            seq_dim: 序列维度
        
        返回:
            旋转后的张量
        """
        seq_len = x.size(seq_dim)
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")
        
        sin = self.sin[:seq_len]
        cos = self.cos[:seq_len]
        
        # 调整维度以匹配输入
        for _ in range(x.dim() - 2):
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        
        # 应用旋转
        return (x * cos) + (self.rotate_half(x) * sin)
