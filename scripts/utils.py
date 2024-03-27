import torch.nn.functional as F
import torch
from torch.distributed.nn import all_gather

from typing import Dict, Union, Iterable
from pathlib import Path
import yaml
import json



def cross_entropy_loss(anchor, positive, negative=None, T: float = 0.07):
    l_pos = torch.einsum("ad,pd->ap", [anchor, positive])  # [batch_size, batch_size]

    if negative is not None:
        negative = F.normalize(negative, p=2, dim=-1).to(anchor)
        l_neg = torch.einsum(
            "ad,nd->an", [anchor, negative]
        )  # [batch_size, negative_size]
        logits = torch.cat([l_pos, l_neg], dim=-1)
    else:
        logits = l_pos
    logits /= T
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)


def load_default_config(ctx, param, filename):
    if filename is None:
        return
    cfg = yaml.safe_load(open(filename))
    ctx.default_map = cfg


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .to(token_embeddings)
    )
    result = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return result


def convert_int_keys(data):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            try:
                int_key = int(key)
                new_data[int_key] = convert_int_keys(value)
            except ValueError:
                new_data[key] = convert_int_keys(value)
        return new_data
    elif isinstance(data, list):
        return [convert_int_keys(item) for item in data]
    else:
        return data


def load_json_with_int_keys(file_path: Union[str, Path]):
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return convert_int_keys(data)


def recall_mrr(anchor, positive, poolsize, device=torch.device("cpu")):
    # anchor: [batch_size, emb_dim]

    anchor = anchor.to(device)
    positive = positive.to(device)

    # Normalize the embeddings
    # anchor = F.normalize(anchor, p=2, dim=-1)
    # positive = F.normalize(positive, p=2, dim=-1)

    # Initialize a random permutation
    perm = torch.randperm(anchor.shape[0]).to(anchor.device)
    anchor = torch.index_select(anchor, 0, perm)
    positive = torch.index_select(positive, 0, perm)

    # Take first (poolsize - 1) elements as negative
    negative_cnt = poolsize - 1
    negative = anchor[:negative_cnt]
    anchor = anchor[negative_cnt:]
    positive = positive[negative_cnt:]

    # Calculate l_pos and l_neg
    l_pos = torch.einsum("nc,nc->n", [anchor, positive]).unsqueeze(-1)
    l_neg = torch.einsum("ic,jc->ij", [anchor, negative])
    logits = torch.cat([l_pos, l_neg], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

    # Sort the logits in descending order
    # logits = logits.cpu()
    _, indices = torch.sort(logits, dim=-1, descending=True)
    indices = indices.to(device)
    ranks = torch.nonzero(indices == labels.unsqueeze(-1), as_tuple=False)[:, -1]
    mrr = torch.reciprocal(ranks.float() + 1).mean()
    recall = (ranks < 1).float().mean()
    return recall, mrr


class TensorQueue:
    def __init__(self, length: int, valid: bool = True):
        self.valid = valid
        self.val = None
        self.length = length

    @torch.no_grad()
    def enqueue(self, val: torch.Tensor):
        if not self.valid:
            return
        if self.val is None:
            self.val = val.detach().clone()
        else:
            val = val.to(self.val.device)
            self.val = torch.cat([self.val, val.detach().clone()], dim=0)
        self.val = self.val[-self.length :].detach().clone()

    def __len__(self):
        if self.val is None:
            return 0
        else:
            return self.val.shape[0]


def create_lr_group(model, lr, weight_decay, name_filter=lambda x: True):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if name_filter(n) and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if name_filter(n) and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return optimizer_grouped_parameters


def gather_with_grad_single(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.cat(all_gather(tensor), dim=0)
    else:
        return tensor

def gather_with_grad(*args):
    if len(args) == 1:
        return gather_with_grad_single(args[0])
    else:
        return [gather_with_grad_single(arg) for arg in args]


def encoder_infer(encoder, input_ids, attention_mask):
    model_output = encoder(input_ids, attention_mask)
    if hasattr(model_output, "last_hidden_state"):
        embedding = mean_pooling(model_output.last_hidden_state, attention_mask)
    elif hasattr(model_output, "pooler_output"):
        embedding = model_output.pooler_output
    else:
        raise NotImplementedError
    return embedding
