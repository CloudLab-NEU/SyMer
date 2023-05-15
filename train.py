import gc
import os
import re
from metric import evaluate
import torch
import h5py
import time
import torch.nn.init as init

from torch.utils.data import DataLoader
from dataset import DeepcomDataset, SymerDataset
from args import get_args
from common import get_dict, Common
from model import HierarchicalTransformer
from opt import LossCompute, NoamOpt, LabelSmoothing
from utils.eval import greedy_decoder, print_predict


def save_model(save_name, epoch, model, optimizer,
               step, warmup):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'opt_warmup': warmup
    }
    torch.save(state, save_name)
    print("Model saved to:", save_name)


def run_train_epoch(data_loader, model, loss_compute, epoch, target_dict, log_interval=2500, amp=False):
    total_loss = 0
    total_step = 0
    time_start = time.time()
    epoch_start = time.time()
    for i, batch in enumerate(data_loader):
        [input_node, ver_indices, hor_indices, target, input_terminal] = batch
        outputs = model.forward(input_node, input_terminal,
                                ver_indices, hor_indices, target)
        n_tokens = (target != target_dict[Common.PAD]).data.sum()

        loss = loss_compute(outputs, target, n_tokens)
        total_loss = total_loss + loss
        total_step += 1
        if i % log_interval == 1:
            time_end = time.time()
            print('Epoch:', '%03d' % epoch, ', Iternum:', '%05d' % i, ', loss=',
                  '{:.6f}'.format(loss), ', lr={:.6f}'.format(loss_compute.opt.optimizer.param_groups[0]['lr']),
                  ", Runs: {:.0f}s".format(time_end - time_start))
            time_start = time_end
    epoch_end = time.time()
    print("Epoch runs {:.0f}s:".format(epoch_end - epoch_start))
    return total_loss / total_step


def run_eval_epoch_deepcom(args, data_loader, model, target_dict, save_name):
    with open("output/{}.txt".format(save_name), "w") as file:
        for i, batch in enumerate(data_loader):
            [input_node, ver_indices, hor_indices, target, input_terminal] = batch
            with torch.no_grad():
                outputs = greedy_decoder(args=args, model=model,
                                         enc_input=[input_node, input_terminal, ver_indices, hor_indices],
                                         start_symbol=target_dict[Common.BOS],
                                         max_target_length=args.max_deepcom_target_length)

            print_line = print_predict(target_dict, outputs, target, False)
            file.writelines("\n".join(print_line) + "\n")
    print("eval file saved as 'output/{}.txt'".format(save_name))


if __name__ == '__main__':
    args = get_args()

    gc.collect()
    torch.cuda.empty_cache()

    node_dict, terminal_dict, target_dict = get_dict(args.data_path)

    trainLoader = DataLoader(
        SymerDataset(args=args, file=args.train_path,
                     data_size=args.train_num, node_dict=node_dict,
                     terminal_dict=terminal_dict, target_dict=target_dict),
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True
    )

    testLoader = DataLoader(
        SymerDataset(args=args, file=args.test_path,
                     data_size=args.test_num,
                     node_dict=node_dict,
                     terminal_dict=terminal_dict, target_dict=target_dict),
        batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        drop_last=True
    )

    hierarchical_transformer = HierarchicalTransformer(args, node_dict, terminal_dict,
                                                       target_dict, args.max_deepcom_target_length + 2).cuda()

    criterion = LabelSmoothing(size=len(target_dict), padding_idx=target_dict[Common.PAD],
                               smoothing=0.1)
    criterion.cuda()

    epoch_start = 0
    print(f"Params: \n{args}")
    if args.load_model:
        _epoch_idx = re.search("\d", args.load_model).start()
        checkpoint = torch.load(args.load_model, map_location='cpu')
        hierarchical_transformer.load_state_dict(checkpoint['model'])
        epoch_start = checkpoint['epoch'] + 1
        model_opt = NoamOpt(d_model=args.d_model, warmup=checkpoint['opt_warmup'],
                            optimizer=torch.optim.Adam(
                                hierarchical_transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                            start_step=checkpoint['step'],
                            scale=args.lr_scale)
        model_opt.load_optimizer(checkpoint['optimizer'])

        args.save_name = args.load_model[6:_epoch_idx] if (args.save_name == "n") else args.save_name

    else:
        model_opt = NoamOpt(d_model=args.d_model, warmup=4000,
                            optimizer=torch.optim.Adam(
                                hierarchical_transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                            start_step=0,
                            scale=args.lr_scale)
        if args.init:
            for p in hierarchical_transformer.parameters():
                if p.dim() > 1:
                    init.xavier_uniform_(p)

    print("-------{}-------".format(time.asctime(time.localtime(time.time()))))

    for epoch in range(epoch_start, epoch_start + args.epoch):
        hierarchical_transformer.train()
        loss = run_train_epoch(data_loader=trainLoader, model=hierarchical_transformer,
                               loss_compute=LossCompute(target_dict, criterion, model_opt, amp=args.amp), epoch=epoch,
                               target_dict=target_dict, amp=args.amp, log_interval=args.log_interval)

        hierarchical_transformer.eval()
        run_eval_epoch_deepcom(args, testLoader, hierarchical_transformer, target_dict,
                               args.save_name + str(epoch))
        rouge_score, ba_score, meteor_score = evaluate(args.save_name + str(epoch), True)

        save_name = "model/{}{}_Ba{}_Me{}_Rl{}.pt".format(args.save_name, str(epoch),
                                                          ba_score, meteor_score, rouge_score)
        save_model(save_name, epoch, hierarchical_transformer,
                   model_opt.optimizer,
                   model_opt.get_step(), model_opt.warmup)
