import torch.nn as nn

from attention import MultiHeadAttention


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )
        self.d_model = args.d_model

    def forward(self, inputs):
        """
        :param inputs: [B, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [B, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args=args)
        self.pos_ffn = PoswiseFeedForwardNet(args=args)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [B, max_contexts_len, d_model]
        :param enc_self_attn_mask: [B, max_contexts_len, max_contexts_len]
        """
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_input to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [B, src_len, d_model]
        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.dec_enc_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [B, target_len, d_model]
        :param enc_outputs: [B, src_len, d_model]
        :param dec_self_attn_mask: [B, target_len, target_len]
        :param dec_enc_attn_mask: [B, target_len, src_len]
        """
        # dec_outputs: [B, target_len, d_model], dec_self_attn: [B, n_heads, target_len, target_len]
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [B, target_len, d_model], dec_enc_attn: [B, n_heads, target_len, src_len]
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs
