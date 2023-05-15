from module import Encoder, Decoder
import torch.nn as nn


class HierarchicalTransformer(nn.Module):
    def __init__(self, args, node_dict, terminal_dict, target_dict, max_target_len):
        super(HierarchicalTransformer, self).__init__()
        self.encoder = Encoder(args=args, node_dict=node_dict, terminal_dict=terminal_dict).cuda()
        self.decoder = Decoder(args=args, target_dict=target_dict, node_dict=node_dict, terminal_dict=terminal_dict,
                               max_target_len=max_target_len).cuda()
        self.projection = nn.Linear(args.d_model, len(target_dict), bias=False).cuda()

    def forward(self, enc_inputs_node, enc_inputs_terminal, ver_indices, hor_indices, dec_inputs):
        """
        :param enc_inputs_node: [B, max_contexts_len, max_path_len]
        :param enc_inputs_terminal: [B, max_contexts_len, max_terminal_subtoken_length]
        :param ver_indices: [B, max_contexts_len, max_path_len]
        :param hor_indices: [B, max_contexts_len, max_path_len]
        :param dec_inputs: [B, tgt_len(src+2)]
        """
        enc_outputs = self.encoder(enc_inputs_node, enc_inputs_terminal, ver_indices, hor_indices)
        dec_outputs = self.decoder(dec_inputs, enc_inputs_node, enc_outputs)

        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1))
