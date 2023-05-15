import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concat", action="store_false")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save", action="store_false")
    parser.add_argument("--batch_size", "-b", type=int,
                        default=128)

    parser.add_argument("--device", default=torch.device("cuda"))

    parser.add_argument("--lr_scale", required=False, default=1, type=float)

    parser.add_argument("--log_interval", "-i", type=int, default=2500)
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--max_context_length", "-c", type=int,
                        default=120)
    parser.add_argument("--max_path_length", type=int,
                        default=9)
    parser.add_argument("--max_terminal_subtoken_length", type=int,
                        default=6,
                        help="分词时最长分词数")

    parser.add_argument("--max_deepcom_target_length",
                        default=30)
    parser.add_argument("--data_path",
                        default="./data/deepcom.dict.psc")
    parser.add_argument("--train_path",
                        default="./data/deepcom.train")
    parser.add_argument("--test_path",
                        default="./data/deepcom.test")
    parser.add_argument("--valid_path",
                        default="./data/deepcom.valid")
    parser.add_argument("--train_num", type=int, default=445763)
    parser.add_argument("--test_num", type=int, default=19999)
    parser.add_argument("--valid_num", type=int, default=19999)

    parser.add_argument("--save_name", required=False, default="n")
    parser.add_argument("--num_workers", type=int,
                        default=0)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8, help="多头头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈神经网络隐藏层维度")
    parser.add_argument("--encoder_n_layers", type=int, default=6, help="encoder layer叠加个数")
    parser.add_argument("--decoder_n_layers", type=int, default=6)

    parser.add_argument("--load_model", required=False)

    assert parser.parse_args().d_model / parser.parse_args().n_heads == parser.parse_args().d_k, "n heads error."

    return parser.parse_args()
