from unicodedata import name
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch.multiprocessing
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import *
from transformers import AdamW
import torch.nn.functional as F
import argparse
from microlm import TextEncoder, tokenize_function, AsmEncoder
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import random
import pandas as pd
from datasets import Dataset
import pickle
from utils import cross_entropy_loss, recall_mrr
device = torch.device("cuda")

def eval_bcsd(options, path, poolsize, negpool):

    fp_data = BCSDatasetDBG(options, path)
    eval_loader = DataLoader(fp_data, batch_size=poolsize, num_workers=24, shuffle=False, collate_fn=fp_data.collate_gat) # 

    arc_list = []
    pos_list = []

    with torch.no_grad():
        for arc, pos, x_len, y_len, x_code, y_code, name in tqdm(eval_loader):
            arc = arc.to(device)
            pos = pos.to(device)

            for i in tqdm(range(len(arc))):  # check every vector of (vA,vB)

                vA = arc[i : i + 1]  #pos[i]
                posA = pos[i : i + 1]
                vB = torch.cat((pos[ : i], pos[i + 1 : ]))            
                sim_pos = torch.mm(vA, posA.T).squeeze()
                sim_neg = torch.mm(vA, vB.T).squeeze()

            arc_list.append(arc)
            pos_list.append(pos)

        arc = torch.cat(arc_list)
        pos = torch.cat(pos_list)

        recall, mrr = recall_mrr(arc, pos, poolsize, device=torch.device("cuda"))

    print('Recall@1: ', recall)
    print('mrr: ', mrr)
    return int(recall.cpu())


def save_datasets(data_list, output_path):
    print(len(data_list))
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="eval_bcsd")
    parser.add_argument("--data_path", type=str, default='/home/xxx/workspace/subject/optrans/data/no_finetune_for_disasm', help='the path of microlm file')
    parser.add_argument("--encoder", type=str, default='/mnt/data/xxx/models/microlm/encoder', help='the path of the encoder')
    parser.add_argument("--tokenizer", type=str, default='/mnt/data/xxx/models/microlm/tokenizer', help='the path of the tokenizer')
    parser.add_argument("--poolsize", type=int, default=10000, help='eval poolsize')
    args = parser.parse_args()

    results = {}
    for options in [["O0", "O3"], ["O1", "O3"], ["O2", "O3"], ["O0", "Os"], ["O1", "Os"], ["O2", "Os"]]:
        results[options[0] + "vs" + options[1]] = eval_bcsd(options, args.data_path, args.poolsize, negpool)
    print(args.data_path)
    print(results)
