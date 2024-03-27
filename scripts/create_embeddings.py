from unicodedata import name
from transformers import BertTokenizer, BertForMaskedLM, BertModel
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
from microlm import TextEncoder, tokenize_function, AsmEncoder
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import pandas as pd
import json
from data import EncodeMicroDataset
from datasets import Dataset
# from clap import *
device = torch.device("cuda")

def create_micro_embeddings(path):

    data_list = []
    fp_data = EncodeMicroDataset(path, micro_tokenizer)
    eval_loader = DataLoader(fp_data, batch_size=800, num_workers=24, shuffle=False, prefetch_factor=4, collate_fn=fp_data.collate_gat)    #prefetch_factor=4, 

    encoder = torch.nn.DataParallel(micro_encoder)
    encoder.eval()

    asm_list = []
    len_list = []
    code_list = []
    with torch.no_grad():
        for asm, length, microcode in tqdm(eval_loader):
            asm = asm.to(device)
            asm = encoder(**asm).squeeze(0).cpu().numpy()
            for data, len, code in zip(asm, length, microcode):
                asm_list.append(data)
                len_list.append(len)
                code_list.append(code)

    index = 0
    for data in tqdm(load_from_disk(path)):
        data_dict = {}
        data_dict["name"] = data["name"]
        for opt in ["O0", "O1", "O2", "O3", "Os"]:
            if data[opt] == None:
                data_dict[opt] = None
                continue
            data_dict[opt] = asm_list[index]
            data_dict[opt + "_len"] = len_list[index]
            data_dict[opt + "_code"] = code_list[index]
            index += 1
        data_list.append(data_dict)

    return data_list

def save_datasets(data_list, output_path):
    print(len(data_list))
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="eval_bcsd")
    parser.add_argument("--data_path", type=str, default='/mnt/data/szh/share/datasets/optrans/BinaryCorp/hf/inline/small_test', help='the path of microlm file')
    parser.add_argument("--encoder", type=str, default='/mnt/data/szh/models/microlm', help='the path of the encoder')
    parser.add_argument("--tokenizer", type=str, default='/mnt/data/szh/models/microlm/', help='the path of the tokenizer')
    parser.add_argument("--output_path", type=str, default='/home/szh/workspace/subject/optrans/data/no_finetune_for_inline', help='the path of the disasm data')
    args = parser.parse_args()
    # /mnt/data/szh/models/optrans/disasm/finetune_epoch_2

    micro_encoder = AsmEncoder.from_pretrained(args.encoder).to(device)
    micro_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    micro_tokenizer.pad_token=micro_tokenizer.unk_token

    data_list = create_micro_embeddings(args.data_path)
    save_datasets(data_list, args.output_path)