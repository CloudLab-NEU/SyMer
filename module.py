from common import Common
from layer import DecoderLayer, EncoderLayer
from position import LearnedEncoderPositionalEncoding, DecoderPositionalEncoding

import torch.nn as nn
import torch
import numpy as np


class Encoder(nn.Module):
    def __init__(self, args, node_dict, terminal_dict):
        super(Encoder, self).__init__()
        self.d_model = args.d_model
        self.node_dict = node_dict
        self.terminal_dict = terminal_dict

        self.node_embedding = nn.Embedding(len(node_dict), self.d_model, padding_idx=node_dict[Common.PAD])
        self.terminal_embedding = nn.Embedding(len(terminal_dict), self.d_model, padding_idx=terminal_dict[Common.PAD])

        self.pos_emb = LearnedEncoderPositionalEncoding(d_model=self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.encoder_n_layers)])

    def forward(self, input_node, input_term, ver_indices, hor_indices):
        """
        :param input_node: [B, max_contexts_len, max_path_len]
        :param input_term: [B, max_contexts_len, max_term_subtoken_len]
        :param ver_indices: [B, m_c_l, m_p_l]
        :param hor_indices: [B, m_c_l, m_p_l]
        """
        batch_size, m_c_l, m_p_l = input_node.size()
        _, _, m_t_s_l = input_term.size()
        # [B, m_c_l, m_p_l, d_model]
        embed_node = self.node_embedding(input_node)

        # [B, m_c_l]: Get nodes' length Tensor
        node_len = (m_p_l + 1) - input_node.eq(self.node_dict[Common.PAD]).sum(dim=2)
        # [B, m_c_l, m_c_l]: Get contexts'(path) mask Tensor
        enc_self_attn_mask = self.get_attn_pad_mask(node_len)
        node_len = node_len.unsqueeze(2)

        # Get term's sub token length
        _term_len = m_t_s_l - input_term.data.eq(self.terminal_dict[Common.PAD]).sum(dim=-1)
        # [B, m_c_l]: Avoid 0 being divided
        _term_len = _term_len + _term_len.eq(0)
        # [B, m_c_l, 1]
        _term_len = _term_len.unsqueeze(2)

        # [B, m_c_l, m_t_s_l, d_model]
        embed_term = self.terminal_embedding(input_term)
        # [B, m_c_l, 1, d_model]:  Scaled sum of sub token
        embed_term = embed_term.sum(dim=2).div(_term_len).unsqueeze(2)

        # [B, m_c_l, m_p_l, d_model]
        embed_node = self.pos_emb(embed_node, ver_indices, hor_indices)
        embed_node[:, :, -1:, :] = embed_node[:, :, -1:, :] + embed_term

        # [B, m_c_l, m_p_l * d_model]
        embed_path = torch.cumsum(embed_node, dim=2).view(batch_size, m_c_l, -1)
        embed_path = torch.div(embed_path, node_len).view(batch_size, m_c_l, m_p_l, -1)

        # [B, m_c_l, d_model]
        enc_outputs = embed_path[:, :, -1, :]

        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)

        return enc_outputs

    @staticmethod
    def get_attn_pad_mask(node_len):
        # [B, m_c_l]
        batch_size, m_c_l = node_len.size()
        enc_self_attn_mask = node_len.eq(1).unsqueeze(1)  # [B, 1, m_c_l]
        return enc_self_attn_mask.expand(batch_size, m_c_l, m_c_l)  # [B, m_c_l, m_c_l]


class Decoder(nn.Module):
    def __init__(self, args, target_dict, node_dict, terminal_dict, max_target_len):
        super(Decoder, self).__init__()
        self.target_emb = nn.Embedding(len(target_dict), args.d_model,
                                       padding_idx=target_dict[Common.PAD])
        self.pos_emb = DecoderPositionalEncoding(args, max_len=max_target_len)

        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.decoder_n_layers)])

        self.node_pad_idx = node_dict[Common.PAD]
        self.terminal_pad_idx = terminal_dict[Common.PAD]
        self.target_pad_idx = target_dict[Common.PAD]

    def forward(self, dec_inputs, enc_inputs_node, enc_outputs):
        """
        :param dec_inputs: [B, tgt_len(src+2)]
        :param enc_inputs_node: [B, max_contexts_len, max_path_len], For getting mask Tensor
        :param enc_outputs: [B, m_c_l, d_model]
        """
        dec_outputs = self.target_emb(dec_inputs)
        # [B, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()
        dec_self_attn_pad_mask = self.get_attn_pad_mask(dec_inputs).cuda()  # [B, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = self.get_attn_subsequence_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda()

        dec_enc_attn_mask = self.get_dec_enc_mask(dec_inputs, enc_inputs_node)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

        return dec_outputs

    def get_attn_pad_mask(self, dec_inputs):
        batch_size, tgt_len = dec_inputs.size()
        # [B, 1, tgt_len], True is masked
        pad_attn_mask = dec_inputs.data.eq(self.target_pad_idx).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, tgt_len, tgt_len)

    @staticmethod
    def get_attn_subsequence_mask(dec_inputs):
        """
        :param dec_inputs: [B, tgt_len]
        """
        seq_shape = [dec_inputs.size(0), dec_inputs.size(1), dec_inputs.size(1)]
        subsequence_mask = np.triu(np.ones(seq_shape), k=1)
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask

    def get_dec_enc_mask(self, dec_inputs, enc_inputs_node):
        """
        :param dec_inputs: [B, tgt_len]
        :param enc_inputs_node: [B, max_contexts_len, max_path_len]
        :return: [B, tgt_len, m_c_l]
        """
        batch_size, tgt_len = dec_inputs.size()
        _, max_contexts_len, max_path_len = enc_inputs_node.size()
        node_len = (max_path_len + 1) - enc_inputs_node.eq(self.node_pad_idx).sum(dim=2)

        dec_enc_mask = node_len.eq(1).unsqueeze(1)  # [B, 1, m_c_l]
        return dec_enc_mask.expand(batch_size, tgt_len, max_contexts_len)
