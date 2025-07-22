import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import re
import pandas as pd

train_df = pd.read_csv("/home/sankuai/dolphinfs_daicong/learn_transformer/data/wmt14_translate_de-en_train.csv", lineterminator='\n')

train_df = train_df.sample(frac=0.2, random_state=42)
print(train_df.head())



def clean_text(text):
    text = text.lower()
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['de'] = train_df['de'].apply(clean_text)
train_df['en'] = train_df['en'].apply(clean_text)

src_tokens_list = [s.split() for s in train_df['en'].tolist()]
tgt_tokens_list = [s.split() for s in train_df['de'].tolist()]

from collections import Counter

src_counter = Counter(tok for sent in src_tokens_list for tok in sent)
tgt_counter = Counter(tok for sent in tgt_tokens_list for tok in sent)

src_vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
for i, (tok, _) in enumerate(src_counter.most_common(), start=4):
    src_vocab[tok] = i

tgt_vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
for i, (tok, _) in enumerate(tgt_counter.most_common(), start=4):
    tgt_vocab[tok] = i

inv_src_vocab = {i: w for w, i in src_vocab.items()}
inv_tgt_vocab = {i: w for w, i in tgt_vocab.items()}

max_len = 16 

def encode_and_pad(tokens, vocab, max_len):

    ids = [vocab.get(tok, vocab['<unk>']) for tok in tokens]

    ids = [vocab['<bos>']] + ids + [vocab['<eos>']]

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [vocab['<pad>']] * (max_len - len(ids))
    return ids

src_seqs = [encode_and_pad(tok_list, src_vocab, max_len) for tok_list in src_tokens_list]
tgt_seqs = [encode_and_pad(tok_list, tgt_vocab, max_len) for tok_list in tgt_tokens_list]


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_tensor = torch.tensor(src_seqs, dtype=torch.long)  
tgt_tensor = torch.tensor(tgt_seqs, dtype=torch.long)

d_model = 16
nhead = 4
dim_feedforward = 64
max_len = 16

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # CPU'da oluştur
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)                     # çift indeks: sin
        pe[:, 1::2] = torch.cos(position * div_term)                     # tek indeks: cos
        self.register_buffer('pe', pe.unsqueeze(0))                      # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]  
    

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(src, src, src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)
        src2 = self.norm1(src + self.dropout(attn_output))                 # Residual + Norm
        ff_output = self.linear2(F.relu(self.linear1(src2)))               # FFN
        output = self.norm2(src2 + self.dropout(ff_output))                # Residual + Norm
        return output, attn_weights


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        attn_output1, attn_weights1 = self.self_attn(tgt, tgt, tgt,
                                                     attn_mask=tgt_mask,
                                                     key_padding_mask=tgt_key_padding_mask)
        tgt2 = self.norm1(tgt + self.dropout(attn_output1))               # Maskeli Self-Attn
        attn_output2, attn_weights2 = self.multihead_attn(tgt2, memory, memory,
                                                          attn_mask=memory_mask,
                                                          key_padding_mask=memory_key_padding_mask)
        tgt3 = self.norm2(tgt2 + self.dropout(attn_output2))              # Cross-Attn
        ff_output = self.linear2(F.relu(self.linear1(tgt3)))              # FFN
        output = self.norm3(tgt3 + self.dropout(ff_output))               # Residual + Norm
        return output, attn_weights1, attn_weights2


class Frenzy(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, dim_feedforward, max_len):
        super().__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_vocab['<pad>'])
        
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_vocab['<pad>'])

        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)

        self.encoder_layer = Encoder(d_model, nhead, dim_feedforward)
        self.decoder_layer = Decoder(d_model, nhead, dim_feedforward)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        # Embedding layer'ın weight'ini kullanarak device'ı al
        device = self.src_embedding.weight.device
        return mask.to(device)

    def forward(self, src_ids, tgt_ids):

        batch_size, src_len = src_ids.size()
        _, tgt_len = tgt_ids.size()

        # 1) Encoder
        src_emb = self.src_embedding(src_ids)             # (batch, src_len, d_model)
        src_emb = self.pos_encoder(src_emb)
        memory, _ = self.encoder_layer(src_emb)

        # 2) Decoder
        tgt_emb = self.tgt_embedding(tgt_ids)             # (batch, tgt_len, d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len)  # (tgt_len, tgt_len)
        output, _, _ = self.decoder_layer(tgt_emb, memory, tgt_mask=tgt_mask)

        logits = self.fc_out(output)                      # (batch, tgt_len, tgt_vocab_size)
        return logits


src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

model = Frenzy(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    max_len=max_len,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.device_count() > 1:
    print("Birden fazla GPU bulundu. DataParallel aktif ediliyor.")
    model = nn.DataParallel(model)  

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>']) 


from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(src_tensor, tgt_tensor)

train_loader = DataLoader(
    dataset,
    batch_size=1024,  # İki GPU için daha büyük batch (her GPU'da 16)
    shuffle=True,
    num_workers=4,  # GPU sayısına eşit
    pin_memory=True,
    persistent_workers=True,
    drop_last=True  # Son batch'i at (GPU'lar arası eşit dağıtım için)
)

from torch.cuda.amp import autocast, GradScaler

scaler = torch.amp.GradScaler('cuda')

num_epochs = 10
print("\n=== Training with FP16 (Mixed Precision) ===")

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        current_batch = batch_idx + 1

        src_batch = src_batch.to(device, non_blocking=True)
        tgt_batch = tgt_batch.to(device, non_blocking=True)

        decoder_input = tgt_batch[:, :-1]
        target = tgt_batch[:, 1:]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'): 
            logits = model(src_batch, decoder_input)
            logits = logits.reshape(-1, logits.size(-1))
            loss = criterion(logits, target.reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1

        print(f"\rEpoch {epoch:02d} [{current_batch:04d}/{len(train_loader):04d}]  "
              f"Batch Loss: {loss.item():.4f}", end='')

    avg_loss = total_loss / batch_count
    print(f"\nEpoch {epoch:02d} Complete  |  Avg Loss: {avg_loss:.4f}")


def translate_sentence(model, sentence, src_vocab, tgt_vocab, inv_tgt_vocab, max_len=15):
    model.eval()
    with torch.no_grad():
        # Preprocess
        tokens = sentence.lower().split()
        src_ids = [src_vocab.get(tok, src_vocab['<unk>']) for tok in tokens]
        src_ids = [src_vocab['<bos>']] + src_ids + [src_vocab['<eos>']]
        src_ids = src_ids[:max_len]
        if len(src_ids) < max_len:
            src_ids += [src_vocab['<pad>']] * (max_len - len(src_ids))
        
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        # Greedy decoding
        tgt_ids = [tgt_vocab['<bos>']]
        for _ in range(max_len-1):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            
            with torch.amp.autocast('cuda'):
                output = model(src_tensor, tgt_tensor)
            
            next_token = output[0, -1, :].argmax().item()
            if next_token == tgt_vocab['<eos>']:
                break
            tgt_ids.append(next_token)
        
        # Decode
        result = [inv_tgt_vocab.get(id, '<unk>') for id in tgt_ids[1:]]  # Skip <bos>
        return ' '.join(result)

# Test
test_sentence = "hello world"
translation = translate_sentence(model, test_sentence, src_vocab, tgt_vocab, inv_tgt_vocab)
print(f"Input: {test_sentence}")
print(f"Translation: {translation}")
