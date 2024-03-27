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
from optrans import TextEncoder, tokenize_function, AsmEncoder
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import random
import pandas as pd
from datasets import Dataset
import pickle
from utils import cross_entropy_loss, recall_mrr
device = torch.device("cuda")

def eval_bcsd(options, path, poolsize):

    fp_data = BCSDataset(options, path)
    eval_loader = DataLoader(fp_data, batch_size=poolsize, num_workers=24, shuffle=False, collate_fn=fp_data.collate_gat)

    arc_list = []
    pos_list = []

    with torch.no_grad():
        for arc, pos in tqdm(eval_loader):
            arc = arc.to(device)
            pos = pos.to(device)

            arc_list.append(arc)
            pos_list.append(pos)

        arc = torch.cat(arc_list)
        pos = torch.cat(pos_list)

        recall, mrr = recall_mrr(arc, pos, poolsize, device=torch.device("cuda"))

    print('Recall@1: ', recall)
    print('mrr: ', mrr)
    return int(recall.cpu())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="eval_bcsd")
    parser.add_argument("--data_path", type=str, default='/path/to/dataset', help='the path of embedding file')
    parser.add_argument("--encoder", type=str, default='/path/to/optrans', help='the path of the encoder')
    parser.add_argument("--tokenizer", type=str, default='/path/to/optrans', help='the path of the tokenizer')
    parser.add_argument("--poolsize", type=int, default=10000, help='eval poolsize')
    args = parser.parse_args()

    results = {}
    for options in [["O0", "O3"], ["O1", "O3"], ["O2", "O3"], ["O0", "Os"], ["O1", "Os"], ["O2", "Os"]]:
        results[options[0] + "vs" + options[1]] = eval_bcsd(options, args.data_path, args.poolsize)
    print(results)
