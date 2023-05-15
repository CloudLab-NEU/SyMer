import os
import torch

from torch.utils.data import DataLoader
from dataset import SymerDataset
from args import get_args
from common import get_dict
from metric import evaluate
from model import SyMer
from train import run_eval_epoch

if __name__ == '__main__':
    args = get_args()
    assert args.load_model is not None, "`load_model` param does not exist."

    node_dict, terminal_dict, target_dict = get_dict(args.data_path)
    
    valid_loader = DataLoader(
        SymerDataset(args=args, file=args.valid_path,
                     data_size=args.valid_num, node_dict=node_dict,
                     terminal_dict=terminal_dict, target_dict=target_dict),
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=False
    )
    
    symer = SyMer(args, node_dict, terminal_dict, target_dict)
    symer.load_state_dict(torch.load(args.load_model))

    symer.eval()
    print("-------------------------")
    print("Validation Dataset:")
    print("sample number:{}".format(args.val_num))
    pre_score, rec_score, f1_score = run_eval_epoch(args=args, data_loader=valid_loader,
                                                    model=symer, target_dict=target_dict,
                                                    save_name=f"{args.load_model}_valid")
    rouge_score, ba_score, meteor_score = evaluate(f"{args.load_model}_valid", True)
