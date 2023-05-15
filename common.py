import pickle

import torch


class Common:
    TARGET_SYMBOL = '|'
    TERMINAL_SYMBOL = '|'
    PATH_SYMBOL = '﹣'
    EMBEDDING_SYMBOL = '°'
    DEEPCOM_PATH_SYMBOL = '`'
    DEEPCOM_EDBEDDING_SYMBOL = '#'
    DEEPCOM_TARGET_SYMBOL = ' '
    BOS = '<S>'
    EOS = '</S>'
    PAD = '<PAD>'
    UNK = '<UNK>'
    COM = '<COM>'


def get_dict(path):
    with open(path, "rb") as file:
        terminal_counter = pickle.load(file)
        node_counter = pickle.load(file)
        target_counter = pickle.load(file)

    terminal_dict = {w: i for i, w in enumerate(
        sorted([w for w, c in terminal_counter.items()]))}
    terminal_dict[Common.UNK] = len(terminal_dict)
    terminal_dict[Common.PAD] = len(terminal_dict)
    # terminal_dict[Common.COM] = len(terminal_dict)

    node_dict = {w: i for i, w in enumerate(sorted(node_counter.keys()))}
    node_dict[Common.UNK] = len(node_dict)
    node_dict[Common.PAD] = len(node_dict)

    target_dict = {w: i for i, w in enumerate(
        sorted([w for w, c in target_counter.items()]))}
    target_dict[Common.UNK] = len(target_dict)
    target_dict[Common.BOS] = len(target_dict)
    target_dict[Common.PAD] = len(target_dict)
    target_dict[Common.EOS] = len(target_dict)

    return node_dict, terminal_dict, target_dict


def target2outputs(target, target_dict):
    batch_size, tgt_len = target.size()

    dec_outputs = target[:, 1:tgt_len]
    pad_outputs = torch.full((batch_size, 1), target_dict[Common.PAD], dtype=torch.long).cuda()
    dec_outputs = torch.cat((dec_outputs, pad_outputs), dim=1)
    return dec_outputs


