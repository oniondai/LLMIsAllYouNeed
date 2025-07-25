# import torch
# import torch.nn as nn

# class MHA(nn.Module):
#     def __init__(self, num_heads, hidden_size, dropout_rate=0.0):
#         super(MHA, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size

#         assert hidden_size % num_heads == 0

#         self.head_dim = hidden_size // num_heads
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.out_projection = nn.Linear(hidden_size, hidden_size)

#     def forward(self, hidden_state, attention_mask=None):
#         batch_size, seq_len, _ = hidden_state.size()

#         # 通过线性层得到QKV
#         query = self.query(hidden_state)
#         key = self.key(hidden_state)
#         value = self.value(hidden_state) # [batch_size, seq_len, hidden_size]

#         # 分头
#         query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [batch_size, num_heads, seq_len, head_dim]
#         key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)    # [batch_size, nums_heads,seq_len, head_dim]
#         value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

#         # 计算attention, q * k^T / sqrt(d_k)
#         attention_weight = torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5) # [batch_size, nums_heads,seq_len, seq_len]
#         if attention_mask is not None:
#             attention_weight = attention_weight.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))

#         attention_weight = torch.softmax(attention_weight, dim=-1) # [batch_size, nums_heads,seq_len, seq_len]
#         attention_weight  =self.dropout(attention_weight)

#         # 计算上下文
#         # # [bs, num_heads, seq_len, seq_len] * [bs, num_heads, seq_len, head_dim] = [bs, num_heads, seq_len, head_dim]
#         context = torch.matmul(attention_weight, value) 

#         # 合并多头
#         # [bs, num_heads, seq_len, head_dim]-> [bs, seq_len, num_heads, head_dim]->[bs, seq_len, num_heads * head_dim]
#         context = context.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

#         output = self.out_projection(context) # [batch_size, seq_len, hidden_size]
#         return output


# if __name__ == '__main__':
#     batch_size = 2
#     hidden_size = 256
#     seq_len = 10
#     num_heads = 8

#     mha = MHA(num_heads, hidden_size)
#     hidden_state = torch.randn(batch_size, seq_len, hidden_size)
#     attention_mask =  torch.ones(batch_size, seq_len)
#     attention_mask[:, 5:] = 0

#     output = mha(hidden_state, attention_mask)
#     print(output.shape)

# import torch
# import torch.nn as nn 

# class MHA(nn.Module):
#     def __init__(self, hidden_size, num_heads, dropout_rate=0.0):
#         super().__init__()
#         assert hidden_size % num_heads == 0

#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads

#         # qkv矩阵
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)

#         self.dropout = nn.Dropout(dropout_rate)
#         self.out_projection = nn.Linear(hidden_size, hidden_size)

#     def forward(self, hidden_state, attention_masks=None):
#         batch_size, seq_len, _ = hidden_state.size()

#         # 输入*QKV矩阵
#         query = self.query(hidden_state) # [batch_size, seq_len, hidden_size]
#         key = self.key(hidden_state)
#         value = self.value(hidden_state)
#         # 分头

#         query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [batch_size, num_heads, seq_len, head_dim]
#         key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
#         value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

#         # 缩放点积
#         attention_weights = torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5) # [batch_size, num_heads, seq_len, seq_len]
#         if attention_masks is not None:
#             attention_weights = attention_weights.masked_fill(attention_masks[:, None, None, :] == 0, float('-inf'))

#         attention_weights = torch.softmax(attention_weights, dim=-1) # [batch_size, num_heads, seq_len, seq_len]
#         attention_weights = self.dropout(attention_weights) 
#         attention_weights = torch.matmul(attention_weights, value) # [batch_size, num_heads, seq_len, head_dim]

#         # 合并头
#         context = attention_weights.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size) # [batch_size, seq_len, num_heads * seq_len]

#         # 线性层输出
#         out = self.out_projection(context)

#         return out


# if __name__ == '__main__':
#     batch_size = 2
#     seq_len = 10
#     hidden_size = 512
#     num_heads = 8 
#     mha = MHA(hidden_size, num_heads)
#     hidden_state = torch.randn(batch_size, seq_len, hidden_size)
#     attention_masks = torch.ones(batch_size, seq_len)
#     attention_masks[:, 5:] = 0 
#     out = mha(hidden_state, attention_masks)
#     print(out.shape)
#     nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=True)
