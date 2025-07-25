# https://blog.csdn.net/weixin_54338498/article/details/135269411
import torch
import torch.nn as nn

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps # eps 防止取倒数之后分母为0
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # weight 是末尾乘的可训练参数, 即g_i
        return (self.weight * hidden_states).to(input_dtype)
