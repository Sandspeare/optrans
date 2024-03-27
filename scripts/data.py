import sys
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import json
import re
import pickle
import numpy as np
import torch
import random
import time
from datasets import load_from_disk
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from torch.utils.data import DataLoader
from microlm import tokenize_function
from transformers import AutoTokenizer, AutoModel

class EncodeMicroDataset(torch.utils.data.Dataset):

    def __init__(self, path, tokenizer):
        data_list = []
        dataset = load_from_disk(path)
        for data in dataset:
            for opt in ["O0", "O1", "O2", "O3", "Os"]:
                asm = data[opt]
                if asm == None:
                    continue
                data_list.append(asm)
        
        self.asmset = data_list        
        self.tokenizer = tokenizer

    def collate_gat(self, batch):
        """
        collate a batch of graph
        """
        x_input = [item[0] for item in batch]
        length = [item[1] for item in batch]
        asm = [item[2] for item in batch]

        x_input = self.tokenizer.pad(x_input, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        return x_input, length, asm

    def __getitem__(self, idx):
        asm = self.asmset[idx]
        x = json.loads(asm)
        x_input = tokenize_function(self.tokenizer, x, model_max_length=1024)  
        return x_input, len(x_input.input_ids), asm
    
    def __len__(self):
        return len(self.asmset)

class OpTransTrainDataset(torch.utils.data.Dataset):

    def __init__(self, path, tokenizer, options): 
        dataset = load_from_disk(path)
        self.options = options
        self.dataset = dataset.filter(self.filter_columns)
        self.tokenizer = tokenizer
        self.nameset = set()
        self.index = 0
        self.random = random.Random()
        self.random.seed(42)

    def filter_columns(self, sample):  
        count = 0
        for opt in sample:
            if opt == 'name':
                continue
            count += 1
        return False if count < 2 else True

    def collate_gat(self, batch):
        """
        collate a batch of graph
        """
        arc_tokens = [item[0] for item in batch]
        pos_tokens = [item[1] for item in batch]
        neg_tokens = [item[2] for item in batch]

        arc_tokens = self.tokenizer.pad(arc_tokens, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        pos_tokens = self.tokenizer.pad(pos_tokens, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        neg_tokens = self.tokenizer.pad(neg_tokens, padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)

        self.nameset = set()
        return arc_tokens, pos_tokens, neg_tokens

    def __getitem__(self, idx): 

        data = self.dataset[idx]

        opt = [item for item in list(data.keys())[1:] if data[item] is not None]
        self.random.shuffle(opt)

        arc_tokens = data[opt[0]]
        pos_tokens = data[opt[1]]

        self.index += 1

        name = data["name"].split("@")[1]
        self.nameset.add(name)

        neg = random.randint(0, len(self.dataset)-1)
        while self.dataset[neg]["name"] in self.nameset:
            neg = random.randint(0, len(self.dataset)-1)
        
        data = self.dataset[neg]
        opt = [item for item in list(data.keys())[1:] if data[item] is not None]
        neg_tokens = data[random.choice(opt)]

        arc_tokens = tokenize_function(self.tokenizer, json.loads(arc_tokens), model_max_length=1024)
        pos_tokens = tokenize_function(self.tokenizer, json.loads(pos_tokens), model_max_length=1024)  
        neg_tokens = tokenize_function(self.tokenizer, json.loads(neg_tokens), model_max_length=1024) 

        return arc_tokens, pos_tokens, neg_tokens

    def __len__(self):
        return len(self.dataset)

class BCSDataset(torch.utils.data.Dataset):

    def __init__(self, options, path):
        self.opt1 = options[0]
        self.opt2 = options[1]
        dataset = load_from_disk(path).shuffle(seed=42)
        self.dataset = dataset.filter(self.filter_columns)

    def filter_columns(self, sample):       
        if sample[self.opt1] == None or sample[self.opt2] == None:
            return False
        else:
            return True

    def collate_gat(self, batch):
        """
        collate a batch of graph
        """
        x_input = [item[0] for item in batch]
        y_input = [item[1] for item in batch]       
        return torch.tensor(x_input), torch.tensor(y_input)

    def __getitem__(self, idx):  

        data = self.dataset[idx]
        x_input = data[self.opt1]
        y_input = data[self.opt2]
        return x_input, y_input
    
    def __len__(self):
        return len(self.dataset)
