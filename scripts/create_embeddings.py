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
from optrans import TextEncoder, tokenize_function, AsmEncoder
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import pandas as pd
import json
from data import EncodeMicroDataset
from datasets import Dataset
device = torch.device("cuda")

def create_embeddings(path):

    data_list = []
    fp_data = EncodeMicroDataset(path, micro_tokenizer)
    eval_loader = DataLoader(fp_data, batch_size=800, num_workers=24, shuffle=False, prefetch_factor=4, collate_fn=fp_data.collate_gat)

    encoder = torch.nn.DataParallel(micro_encoder)
    encoder.eval()

    asm_list = []
    with torch.no_grad():
        for asm in tqdm(eval_loader):
            asm = asm.to(device)
            asm = encoder(**asm).squeeze(0).cpu().numpy()
            for data in asm:
                asm_list.append(data)

    index = 0
    for data in tqdm(load_from_disk(path)):
        data_dict = {}
        data_dict["name"] = data["name"]
        for opt in ["O0", "O1", "O2", "O3", "Os"]:
            if data[opt] == None:
                data_dict[opt] = None
                continue
            data_dict[opt] = asm_list[index]
            index += 1
        data_list.append(data_dict)

    return data_list

def save_datasets(data_list, output_path):
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="eval_bcsd")
    parser.add_argument("--data_path", type=str, default='/path/to/testdataset', help='the path of test file')
    parser.add_argument("--encoder", type=str, default='/path/to/optrans', help='the path of the encoder')
    parser.add_argument("--tokenizer", type=str, default='/path/to/optrans', help='the path of the tokenizer')
    parser.add_argument("--output_path", type=str, default='/path/to/output', help='the path of the disasm data')
    args = parser.parse_args()

    micro_encoder = AsmEncoder.from_pretrained(args.encoder).to(device)
    micro_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    micro_tokenizer.pad_token=micro_tokenizer.unk_token

    data_list = create_embeddings(args.data_path)
    save_datasets(data_list, args.output_path)