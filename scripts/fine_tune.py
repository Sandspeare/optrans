from unicodedata import name
from transformers import AutoTokenizer, BertForMaskedLM, BertModel
import torch.multiprocessing
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time
from datasets import load_from_disk
import data
import pickle
from data import OpTransTrainDataset
from microlm import AsmEncoder
from utils import recall_mrr, cross_entropy_loss
WANDB = True
device = torch.device("cuda")

def train_dp(model, args, train_dataloader, valid_dataloader):

    class Triplet_COS_Loss(nn.Module):
        def __init__(self,margin):
            super(Triplet_COS_Loss, self).__init__()
            self.margin=margin

        def forward(self, repr, good_code_repr, bad_code_repr):
            good_sim=F.cosine_similarity(repr, good_code_repr)
            bad_sim=F.cosine_similarity(repr, bad_code_repr)
            loss=(self.margin-(good_sim-bad_sim)).clamp(min=1e-6).mean()
            return loss

    if WANDB:
        wandb.init(project=f'microlm-finetune', name="microlm_disasm_bcsd")
        wandb.config.update(args)

    
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    model = nn.DataParallel(model)
    global_steps = 0

    triplet_loss = Triplet_COS_Loss(margin=0.5)
    criterion = nn.CrossEntropyLoss()

    mrr, recall1 = finetune_eval(model, valid_dataloader)
    print(mrr, recall1)
    if WANDB:
        wandb.log({'mrr': mrr, 'recall1': recall1})

    model.train()

    for epoch in range(args.epoch):

        train_iterator = tqdm(train_dataloader)

        for i, (seq1, seq2, seq3) in enumerate(train_iterator):
            input_ids1 = seq1.to(device)
            input_ids2 = seq2.to(device)
            # input_ids3 = seq3.to(device)

            arc = model(**input_ids1)
            pos = model(**input_ids2)
            # neg = model(**input_ids3)

            # loss = triplet_loss(arc, pos, neg)

            loss = cross_entropy_loss(arc, pos)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (i + 1) % args.log_every == 0:
                global_steps += 1
                train_iterator.set_description(f"[*] epoch: [{epoch}/{args.epoch+1}], steps: [{i}/{len(train_iterator)}], loss={loss}")
                if WANDB:
                    wandb.log({'triplet loss':loss, 'global_step':global_steps})

            if (i + 1) % 150 == 0:
                mrr, recall1 = finetune_eval(model, valid_dataloader)
                print(mrr, recall1)
                if WANDB:
                    wandb.log({'mrr': mrr, 'recall1': recall1})

            if (i + 1) % 150 == 0:
                model.module.save_pretrained(os.path.join(args.output_path, f"finetune_epoch_{i+1}"))





def finetune_eval(model, data_loader):
    model.eval()
    with torch.no_grad():
        eval_iterator = tqdm(data_loader)
        arc_list = []
        pos_list = []
        for index, (seq1, seq2, _) in enumerate(eval_iterator):
            
            if index == 20:
                break

            input_ids1 = seq1.to(device)
            input_ids2 = seq2.to(device)

            arc = model(**input_ids1).cpu()
            pos = model(**input_ids2).cpu()

            arc_list.append(arc)
            pos_list.append(pos)

        arc = torch.cat(arc_list)
        pos = torch.cat(pos_list)

        recall, mrr = recall_mrr(arc, pos, 10000)
    
    model.train() 
    return mrr, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune")
    parser.add_argument("--model_path", type=str, default='/mnt/data/xxx/models/microlm',  help='the path of pretrain model')
    parser.add_argument("--tokenizer_path", type=str, default='/mnt/data/xxx/models/microlm',  help='the path of pretrain model')
    parser.add_argument("--epoch", type=int, default=10, help='number of training epochs')
    parser.add_argument("--lr", type=float, default=1e-6, help='learning rate')
    parser.add_argument("--batch_size", type=int, default = 192, help='training batch size')
    parser.add_argument("--eval_batch_size", type=int, default = 512, help='evaluation batch size')
    parser.add_argument("--finetune_cnt", type=int, default=2, help='number of layers to freeze')
    parser.add_argument("--weight_decay", type=int, default = 1e-4, help='regularization weight decay')
    parser.add_argument("--eval_every", type=int, default=1, help="evaluate the model every x epochs")
    parser.add_argument("--save_every", type=int, default=1, help="save the model every x epochs")
    parser.add_argument("--log_every", type=int, default =1, help='logging frequency')

    parser.add_argument("--output_path", type=str, default='/mnt/data/xxx/models/optrans/inline', help='the path where the finetune model be saved')
    parser.add_argument("--train_path", type=str, default="/mnt/data/xxx/share/datasets/optrans/BinaryCorp/hf/inline/small_test", help='the path of training data')
    parser.add_argument("--eval_path", type=str, default="/mnt/data/xxx/share/datasets/optrans/BinaryCorp/hf/inline/small_test", help='the path of evaluation data')
    args = parser.parse_args()

    model = AsmEncoder.from_pretrained(args.model_path)
    
    finetune_layer_count = args.finetune_cnt
    for param in model.roformer.embeddings.parameters():
        param.requires_grad = False

    if finetune_layer_count != 0:
        for layer in model.roformer.encoder.layer[:-finetune_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.unk_token

    fp_train_data = OpTransTrainDataset(args.train_path, tokenizer, ["O0", "O1", "O2", "O3", "Os"])
    train_dataloader = DataLoader(fp_train_data, batch_size=args.batch_size, num_workers=24, shuffle=True, collate_fn=fp_train_data.collate_gat, prefetch_factor=4)

    fp_eval_data = OpTransTrainDataset(args.eval_path, tokenizer, ["O0", "O1", "O2", "O3", "Os"])
    eval_dataloader = DataLoader(fp_eval_data, batch_size=args.eval_batch_size, num_workers=24, shuffle=False, collate_fn=fp_eval_data.collate_gat, prefetch_factor=4)

    train_dp(model, args, train_dataloader, eval_dataloader)

