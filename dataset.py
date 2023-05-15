import torch
from torch.utils.data import Dataset, DataLoader

from args import get_args
from common import Common, get_dict


class SymerDataset(Dataset):

    def __init__(self, args, file, data_size, node_dict, terminal_dict, target_dict):
        super(SymerDataset, self).__init__()
        with open(file, 'r') as f:
            files = []
            for line in f:
                line = line.strip('\n')
                line = line.rstrip()
                files.append(line)
        self.files = files
        self.data_size = data_size
        self.node_dict = node_dict
        self.target_dict = target_dict
        self.terminal_dict = terminal_dict

        self.max_context_length = args.max_context_length
        self.max_target_length = args.max_deepcom_target_length
        self.max_path_length = args.max_path_length
        self.max_terminal_subtoken_length = args.max_terminal_subtoken_length

        self.device = args.device

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        file = self.files[index]

        nl = file.split(" ")[0]
        contexts = file.split(" ")[1:]

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

            nodes = path.split(Common.DEEPCOM_PATH_SYMBOL)
            for n in nodes[:-1][:self.max_path_length]:
                n_type, ver, hor = n.split(Common.DEEPCOM_EDBEDDING_SYMBOL)
                path_type.append(self.node_dict[n_type] if n_type in self.node_dict
                                 else self.node_dict[Common.UNK])
                path_ver_indices.append(int(ver))
                path_hor_indices.append(int(hor))

            _path_len = len(path_type)
            path_ver_indices += [0] * (self.max_path_length - _path_len)
            path_hor_indices += [0] * (self.max_path_length - _path_len)
            path_type += [self.node_dict[Common.PAD]] * (self.max_path_length - _path_len)

            contexts_path.append(path_type)
            contexts_path_ver_indices.append(path_ver_indices)
            contexts_path_hor_indices.append(path_hor_indices)

            for t in nodes[-1].split(Common.TERMINAL_SYMBOL)[:self.max_terminal_subtoken_length]:
                path_terminal_node.append(self.terminal_dict[t] if t in self.terminal_dict
                                          else self.terminal_dict[Common.UNK])
            path_terminal_node += [self.terminal_dict[Common.PAD]] * (
                self.max_terminal_subtoken_length - len(path_terminal_node)
            )

            contexts_path_terminal_node.append(path_terminal_node)

        # Padding Contexts
        _contexts_len = len(contexts_path)
        contexts_path += [[self.node_dict[Common.PAD]] * self.max_path_length] * (
                self.max_context_length - _contexts_len)
        contexts_path_ver_indices += [[0] * self.max_path_length] * (self.max_context_length - _contexts_len)
        contexts_path_hor_indices += [[0] * self.max_path_length] * (self.max_context_length - _contexts_len)

        contexts_path_terminal_node += [[self.terminal_dict[Common.PAD]] * self.max_terminal_subtoken_length] * (
                self.max_context_length - _contexts_len)

        # Target Processing
        target.append(self.target_dict[Common.BOS])

        for word in nl.split(Common.DEEPCOM_TARGET_SYMBOL)[:self.max_target_length]:
            target.append(self.target_dict[word] if word in self.target_dict
                          else self.target_dict[Common.UNK])
        target.append(self.target_dict[Common.EOS])
        # adding 2 for `BOS` and `EOS` token
        target += [self.target_dict[Common.PAD]] * (2 + self.max_target_length - len(target))

        return torch.tensor(contexts_path, dtype=torch.long).to(self.device), \
               torch.tensor(contexts_path_ver_indices, dtype=torch.long).to(self.device), \
               torch.tensor(contexts_path_hor_indices, dtype=torch.long).to(self.device), \
               torch.tensor(target, dtype=torch.long).to(self.device), \
               torch.tensor(contexts_path_terminal_node, dtype=torch.long).to(self.device)


if __name__ == '__main__':
    args = get_args()
    node_dict, terminal_dict, target_dict = get_dict(args.data_path)

    dd = SymerDataset(args=args, file=args.train_path,
                      data_size=args.train_num, node_dict=node_dict,
                      terminal_dict=terminal_dict, target_dict=target_dict)

    print(dd.__getitem__(1))
