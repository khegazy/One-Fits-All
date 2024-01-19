import numpy as np
from typing import NamedTuple
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class ModelPrediction(NamedTuple):
    """a docstring"""
    values: torch.tensor
    tokens: torch.tensor
    logits: torch.tensor

class reduce_layer(nn.Module):

    def __init__(self, vector_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((vector_size, 1)))

        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    
    def forward(self, input):
        return torch.squeeze(torch.matmul(input, self.weight), -1)

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device, tokenizer=None):
        super(GPT4TS, self).__init__()
        self.config = configs
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if self.config.is_gpt:
            if self.config.pretrain:
                self.gpt2_enc = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                if self.config.recursive:
                    self.gpt2_dec = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True, add_cross_attention=self.config.recursive)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2_enc = GPT2Model(GPT2Config())
                if self.config.recursive:
                    self.gpt2_dec = GPT2Model(GPT2Config(add_cross_attention=self.config.recursive))
            self.gpt2_enc.h = self.gpt2_enc.h[:self.config.gpt_layers]
            print("gpt2 = {}".format(self.gpt2_enc))

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.in_layer = reduce_layer(self.config.patch_size)
        else: 
            self.in_layer = nn.Linear(self.config.patch_size, self.config.d_model)
        if self.config.recursive:
            if self.config.predict_tokens:
                self.out_layer = nn.Linear(self.config.d_model, self.tokenizer.n_tokens)
            else:
                self.out_layer = reduce_layer(self.config.d_model)
        else:
            self.out_layer = nn.Linear(self.config.d_model * self.patch_num, self.config.pred_len)
        
        if self.config.freeze and self.config.pretrain:
            for i, (name, param) in enumerate(self.gpt2_enc.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            if self.config.recursive:
                for i, (name, param) in enumerate(self.gpt2_dec.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        for layer in (self.gpt2_enc, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        if self.config.recursive:
            self.gpt2_dec.to(device=device)
            self.gpt2_dec.train()
        
        # Setup tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            print(self.tokenizer.n_tokens, self.config.d_model)
            self.embedding = torch.nn.Embedding(self.tokenizer.n_tokens, self.config.d_model, device=device)
            self.start_token = torch.nn.Parameter(torch.empty((1, 1, self.config.d_model)))
            torch.nn.init.kaiming_uniform_(self.start_token, a=np.sqrt(5))
            self.rearrange1 = 'b l m d -> (b m) d l'
            self.rearrange2 = 'bm d n p -> bm n d p'
            self.rearrange3 = '(b m) l d -> b l m d'
        else:
            self.rearrange1 = 'b l m -> b m l'
            self.rearrange2 = 'b m n p -> (b m) n p'
            self.rearrange3 = '(b m) l -> b l m'
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        if self.config.normalize_input:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev

        if self.tokenizer is not None:
            #print("IN TOKENIZER", x.shape)
            x = self.tokenizer(x)
            x = self.embedding(x)
            #print("OUT TOKENIZER", x.shape)
        x = rearrange(x, self.rearrange1)
        #print("SHAPE", "1", x.shape)

        x = self.padding_patch_layer(x)
        #print("SHAPE", "1", x.shape)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        #print("SHAPE", "2", x.shape)
        x = rearrange(x, self.rearrange2)
        #print("SHAPE", "3", x.shape)

        outputs = self.in_layer(x)
        #print("SHAPE", "4", outputs.shape)

        # Run encoder
        if self.is_gpt:
            enc_outputs = self.gpt2_enc(inputs_embeds=outputs)
        
        # Run decoder
        if self.config.recursive:
            is_training = self.gpt2_dec.training
            if is_training:
                self.gpt2_dec.eval()
            with torch.no_grad():
                dec_outputs = enc_outputs.last_hidden_state[:,-1,:].unsqueeze(1) #torch.tile(self.start_token, (x.shape[0], 1, 1))
                for idx in range(self.config.pred_len-1):
                    new_outputs = self.gpt2_dec(
                        inputs_embeds=dec_outputs,
                        encoder_hidden_states=enc_outputs.last_hidden_state,
                    )
                    dec_outputs = torch.cat((dec_outputs, new_outputs.last_hidden_state[:,-1,:].unsqueeze(1)), dim=1)
            if is_training:
                self.gpt2_dec.train()
            new_outputs = self.gpt2_dec(
                inputs_embeds=dec_outputs,
                encoder_hidden_states=enc_outputs.last_hidden_state,
            )
            pred_outputs = torch.cat((dec_outputs, new_outputs.last_hidden_state[:,-1,:].unsqueeze(1)), dim=1)
            print("DEC OUT", pred_outputs.shape)
            #pred_tokens = dec_outputs
            pred_outputs = self.out_layer(pred_outputs)
        else:
            pred_outputs = enc_outputs.last_hidden_state
            pred_outputs = pred_outputs.reshape(B*M, -1)
        
        # Transform decoder output to values and get tokens
        if self.tokenizer is None:
            pred_logits = None
            pred_tokens = None
            pred_values = self.out_layer(pred_outputs)
            pred_values = rearrange(pred_values, '(b m) l -> b l m', b=B)
        elif self.config.predict_tokens:
            pred_logits = pred_outputs
            pred_logits = rearrange(pred_logits, '(b m) l d -> b l m d', b=B)
            pred_tokens = torch.argmax(self.out_layer(pred_logits), dim=-1)
            pred_tokens = rearrange(pred_tokens, '(b m) l -> b l m', b=B)
            pred_values = self.tokenizer.invert(pred_tokens)
            pred_values = rearrange(pred_values, '(b m) l -> b l m', b=B)
        else:
            #IDEA: Do x_entropy on probability distribution of tokens (add error bars to results)
            pred_logits = None
            pred_values = self.out_layer(pred_outputs)
            pred_values = rearrange(pred_values, '(b m) l -> b l m', b=B)
            pred_tokens = self.tokenizer(pred_values)
        
        #print("gpt out", outputs.shape)
        # Unnormalize values
        if self.config.normalize_input:
            pred_values = pred_values * stdev
            pred_values = pred_values + means
        
        if self.tokenizer is not None and pred_tokens is not None:
            pred_logits = pred_outputs
            pred_tokens = torch.argmax(self.out_layer(pred_logits), dim=-1)

        #TODO: if predict tokens must have tokenizer, if predict tokens must not normalize
        return ModelPrediction(pred_values, pred_tokens, pred_logits)
