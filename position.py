import math
import torch
import torch.nn as nn


class LearnedEncoderPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 max_ver_indices=13, max_hor_indices=161,
                 layer_norm_eps=1e-12):
        super(LearnedEncoderPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ver_emb = nn.Embedding(max_ver_indices, d_model, padding_idx=0)
        self.hor_emb = nn.Embedding(max_hor_indices, d_model, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, ver_indices, hor_indices):
        pos_emb = self.ver_emb(ver_indices) + self.hor_emb(hor_indices)
        x = x + pos_emb
        x = self.dropout(self.LayerNorm(x))
        return x


class DecoderPositionalEncoding(nn.Module):
    def __init__(self, args, dropout=0.1, max_len=30):
        super(DecoderPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, args.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.d_model, 2).float() * \
                             (-math.log(10000.0) / args.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('d_pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.d_pe[:x.size(0), :]
        return self.dropout(x)
