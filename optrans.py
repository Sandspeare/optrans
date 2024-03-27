import logging
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers import BatchEncoding
from transformers import MPNetTokenizerFast


from transformers.models.roformer.modeling_roformer import (
    RoFormerEmbeddings,
    RoFormerModel,
    RoFormerEncoder,
    RoFormerLayer,
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerSelfAttention,
    RoFormerPreTrainedModel
)

from transformers.models.mpnet.modeling_mpnet import MPNetModel



class JRoFormerEmbeddings(RoFormerEmbeddings):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = self.word_embeddings


class JRoFormerSelfAttention(RoFormerSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )


class JRoFormerAttention(RoFormerAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = JRoFormerSelfAttention(config)


class JRoFormerLayer(RoFormerLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = JRoFormerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = RoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)


class JRoFormerEncoder(RoFormerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [JRoFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )


class JRoFormerModel(RoFormerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = JRoFormerEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )

        self.encoder = JRoFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

class AsmEncoder(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roformer = JRoFormerModel(config)
        self.projection = nn.Linear(config.hidden_size, config.bla_dim)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        asm_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        asm_embedding = self.projection(asm_embedding)
        asm_embedding = F.normalize(asm_embedding, p=2, dim=1)

        return asm_embedding

def tokenize_function(tokenizer, function, model_max_length=1024):
    total_len = 0
    tokenized_functions = {"instr": [], "input_ids": [], "attention_mask": []}
    for key, value in function.items():

        coprus_inst = ""
        for i in range(len(value)):
            string = str(value[i])
            coprus_inst += string
        tokens = tokenizer(coprus_inst, add_special_tokens=False)
        # tokens = tokenizer(coprus_inst, max_length=32, truncation=True, add_special_tokens=False)
        # print(tokenizer.batch_decode(tokens.input_ids))
        instr_index = key
        instructions = [instr_index] * len(tokens.input_ids)

        tokenized_functions["instr"].extend(instructions)
        tokenized_functions["input_ids"].extend(tokens.input_ids)
        tokenized_functions["attention_mask"].extend(tokens.attention_mask)
        
        total_len += len(tokens.input_ids)
        if total_len > model_max_length:
            tokenized_functions['instr'] = tokenized_functions['instr'][:model_max_length]
            tokenized_functions['input_ids'] = tokenized_functions['input_ids'][:model_max_length]
            tokenized_functions['attention_mask'] = tokenized_functions['attention_mask'][:model_max_length] 
            break

    return BatchEncoding({
            "input_ids": tokenized_functions['input_ids'],
            "attention_mask": tokenized_functions['attention_mask'],
            "token_type_ids": tokenizer.convert_tokens_to_ids(tokenized_functions["instr"]),
        })
