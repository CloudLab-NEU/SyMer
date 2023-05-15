import torch

from common import Common


def print_predict(target_dict, outputs, target, do_print=True):
    tgt_idx2str = {}

    for k, v in target_dict.items():
        tgt_idx2str[v] = k

    rets = list()

    for pre, tgt in zip(outputs, target):
        single_tgt = []
        single_pre = []

        for word in tgt[1:]:
            if word == target_dict[Common.EOS]:
                break
            single_tgt.append(tgt_idx2str[word.item()])

        for word in pre[1:]:
            if word == target_dict[Common.EOS]:
                break
            single_pre.append(tgt_idx2str[word.item()])
        ret = "{}|{}".format(" ".join(single_pre), " ".join(single_tgt))
        if do_print:
            print(ret)
        rets.append(ret)
    return rets


def greedy_decoder(args, model, enc_input, start_symbol, max_target_length=None):
    [input_node, input_term, ver_indices, hor_indices] = enc_input
    enc_outputs = model.encoder(input_node, input_term, ver_indices, hor_indices)
    return _greedy_decoder(args, model, start_symbol, input_node, enc_outputs, max_target_length)


def _greedy_decoder(args, model, start_symbol, input_node,
                    enc_outputs, tgt_len=None):
    if tgt_len is None:
        tgt_len = args.max_target_length + 2
    dec_inputs = torch.zeros(args.batch_size, tgt_len).type_as(input_node)
    next_symbol = start_symbol

    for i in range(0, tgt_len):
        dec_inputs[:, i] = next_symbol

        dec_outputs = model.decoder(dec_inputs, input_node, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.max(dim=-1, keepdim=False)[1]
        next_symbol = prob.data[:, i]
    return dec_inputs
