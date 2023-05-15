import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from args import get_args
from common import Common, get_dict


def nlcode2tensor4deepcom(code, nl, node_dict, terminal_dict, target_dict,
                          max_context_length, max_path_length, max_target_length,
                          max_terminal_subtoken_length):
    contexts = code.split(" ")[1:]

    contexts_path = []
    contexts_path_ver_indices = []
    contexts_path_hor_indices = []
    contexts_path_terminal_node = []
    target = []

    for path in contexts[:int(max_context_length)]:
        if len(path) <= 0:
            break

        path_type = []
        path_ver_indices = []
        path_hor_indices = []
        path_terminal_node = []

        node = path.split(Common.DEEPCOM_PATH_SYMBOL)
        for n in node[:-1][:max_path_length]:
            n_type, ver_indices, hor_indices = n.split(Common.DEEPCOM_EDBEDDING_SYMBOL)

            path_type.append(node_dict[n_type] if n_type in node_dict
                             else node_dict[Common.UNK])
            path_ver_indices.append(int(ver_indices))
            path_hor_indices.append(int(hor_indices))

        _path_len = len(path_type)
        path_ver_indices += [0] * (max_path_length - _path_len)
        path_hor_indices += [0] * (max_path_length - _path_len)
        path_type += [node_dict[Common.PAD]] * (max_path_length - _path_len)

        contexts_path.append(path_type)
        contexts_path_ver_indices.append(path_ver_indices)
        contexts_path_hor_indices.append(path_hor_indices)

        # Terminals node process
        for t in node[-1].split(Common.TERMINAL_SYMBOL)[:max_terminal_subtoken_length]:
            path_terminal_node.append(terminal_dict[t] if t in terminal_dict
                                      else terminal_dict[Common.UNK])
        path_terminal_node += [terminal_dict[Common.PAD]] * (
                max_terminal_subtoken_length - len(path_terminal_node))

        contexts_path_terminal_node.append(path_terminal_node)

    # PAD Contexts
    _contexts_len = len(contexts_path)
    contexts_path += [[node_dict[Common.PAD]] * max_path_length] * (
            max_context_length - _contexts_len)
    contexts_path_ver_indices += [[0] * max_path_length] * (max_context_length - _contexts_len)
    contexts_path_hor_indices += [[0] * max_path_length] * (max_context_length - _contexts_len)

    contexts_path_terminal_node += [[terminal_dict[Common.PAD]] * max_terminal_subtoken_length] * (
            max_context_length - _contexts_len)

    # Target Process
    target.append(target_dict[Common.BOS])
    # print(nl)
    for word in nl.split(Common.DEEPCOM_TARGET_SYMBOL)[:max_target_length]:
        target.append(target_dict[word] if word in target_dict
                      else target_dict[Common.UNK])
    target.append(target_dict[Common.EOS])
    target += [target_dict[Common.PAD]] * (
            max_target_length - len(target) + 2)

    return torch.tensor(contexts_path, dtype=torch.long).cuda(), \
           torch.tensor(contexts_path_ver_indices, dtype=torch.long).cuda(), \
           torch.tensor(contexts_path_hor_indices, dtype=torch.long).cuda(), \
           torch.tensor(target, dtype=torch.long).cuda(), \
           torch.tensor(contexts_path_terminal_node, dtype=torch.long).cuda()


class DeepcomDataset(Dataset):
    def __init__(self, args, file, data_size, node_dict, terminal_dict, target_dict):
        super(DeepcomDataset, self).__init__()
        self.file = file
        self.data_size = data_size
        self.node_dict = node_dict
        self.terminal_dict = terminal_dict
        self.target_dict = target_dict

        self.max_context_length = args.max_context_length
        self.max_target_length = args.max_deepcom_target_length
        self.max_path_length = args.max_path_length
        self.max_terminal_subtoken_length = args.max_terminal_subtoken_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        code = self.file[str(index)]["line"][()].strip("\n")
        nl = self.file[str(index)]["line"][()].strip("\n")

        contexts = code.split(" ")[1:]

        contexts_path = []
        contexts_path_ver_indices = []
        contexts_path_hor_indices = []
        contexts_path_terminal_node = []
        target = []

        for path in contexts[:int(self.max_context_length)]:
            if len(path) <= 0:
                break

            path_type = []
            path_ver_indices = []
            path_hor_indices = []
            path_terminal_node = []

            node = path.split(Common.DEEPCOM_PATH_SYMBOL)
            for n in node[:-1][:self.max_path_length]:
                n_type, ver_indices, hor_indices = n.split(Common.DEEPCOM_EDBEDDING_SYMBOL)

                path_type.append(node_dict[n_type] if n_type in node_dict
                                 else node_dict[Common.UNK])
                path_ver_indices.append(int(ver_indices))
                path_hor_indices.append(int(hor_indices))

            _path_len = len(path_type)
            path_ver_indices += [0] * (self.max_path_length - _path_len)
            path_hor_indices += [0] * (self.max_path_length - _path_len)
            path_type += [node_dict[Common.PAD]] * (self.max_path_length - _path_len)

            contexts_path.append(path_type)
            contexts_path_ver_indices.append(path_ver_indices)
            contexts_path_hor_indices.append(path_hor_indices)

            # Terminals node process
            for t in node[-1].split(Common.TERMINAL_SYMBOL)[:self.max_terminal_subtoken_length]:
                path_terminal_node.append(terminal_dict[t] if t in terminal_dict
                                          else terminal_dict[Common.UNK])
            path_terminal_node += [terminal_dict[Common.PAD]] * (
                    self.max_terminal_subtoken_length - len(path_terminal_node))

            contexts_path_terminal_node.append(path_terminal_node)

        # PAD Contexts
        _contexts_len = len(contexts_path)
        contexts_path += [[node_dict[Common.PAD]] * self.max_path_length] * (
                self.max_context_length - _contexts_len)
        contexts_path_ver_indices += [[0] * self.max_path_length] * (self.max_context_length - _contexts_len)
        contexts_path_hor_indices += [[0] * self.max_path_length] * (self.max_context_length - _contexts_len)

        contexts_path_terminal_node += [[terminal_dict[Common.PAD]] * self.max_terminal_subtoken_length] * (
                self.max_context_length - _contexts_len)

        # Target Process
        target.append(target_dict[Common.BOS])
        # print(nl)
        for word in nl.split(Common.DEEPCOM_TARGET_SYMBOL)[:self.max_target_length]:
            target.append(target_dict[word] if word in target_dict
                          else target_dict[Common.UNK])
        target.append(target_dict[Common.EOS])
        target += [target_dict[Common.PAD]] * (
                self.max_target_length - len(target) + 2)

        return torch.tensor(contexts_path, dtype=torch.long).cuda(), \
               torch.tensor(contexts_path_ver_indices, dtype=torch.long).cuda(), \
               torch.tensor(contexts_path_hor_indices, dtype=torch.long).cuda(), \
               torch.tensor(target, dtype=torch.long).cuda(), \
               torch.tensor(contexts_path_terminal_node, dtype=torch.long).cuda()


if __name__ == '__main__':
    args = get_args()
    train_h5 = h5py.File(args.train_path, "r")

    node_dict, terminal_dict, target_dict = get_dict(args.data_path)

    dd = DeepcomDataset(args=args, file=train_h5,
                        data_size=args.train_num, node_dict=node_dict,
                        terminal_dict=terminal_dict, target_dict=target_dict)

    print(dd.__getitem__(1))
