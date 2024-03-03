import sys
import time
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
    value_values: torch.tensor
    token_values: torch.tensor
    token_logits: torch.tensor
    attn_mask_idxs: torch.tensor

class reduce_layer(nn.Module):
    def __init__(self, vector_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((vector_size, 1)))

        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    
    def forward(self, input):
        return torch.squeeze(torch.matmul(input, self.weight), -1)

class output_mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            n_layers,
            reshape=None,
            squeeze_result=False,
            tanh_activation=False,
            name=""):
        super().__init__()
        self.n_layers = n_layers
        self.squeeze_result = squeeze_result
        self.reshape = reshape
        if name != "":
            name = "_" + name
        self.name_prefix = f"output_mlp{name}"

        if n_layers == 1:
            self.layers = nn.ModuleDict(
                {f"{self.name_prefix}_{0}" : nn.Linear(input_size, output_size)}
            )
            self.activations = None
            return

        if tanh_activation:
            self.activation = nn.Tanh()
        else:        
            self.activation = nn.ReLU()
        
        hidden_size = min(input_size, output_size)*2
        out_sizes = [hidden_size]*(n_layers - 1) + [output_size]
        in_sizes = [input_size] + out_sizes[:-1]
        self.layers = nn.ModuleDict([
            [f"{self.name_prefix}_{idx}", nn.Linear(i_size, o_size)]
            for idx, (i_size, o_size) in enumerate(zip(in_sizes, out_sizes))
        ])
    
    def forward(self, input):
        x = self.layers[f"{self.name_prefix}_0"](input)
        for idx in range(1, self.n_layers):
            x = self.activation(x)
            x = self.layers[f"{self.name_prefix}_{idx}"](x)

        if self.squeeze_result:
            return torch.squeeze(x, -1)
        elif self.reshape is not None:
            return torch.reshape(x, self.reshape)
        else:       
            return x


class GPT4TS(nn.Module):
    
    def __init__(self, configs, device, tokenizer=None):
        super(GPT4TS, self).__init__()
        self.device = device
        self.config = configs
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_len = self.pred_len + self.label_len
        self.from_pretrain_model = configs.from_pretrain_model
        self.do_pretrain_embeddings = configs.do_pretrain_embeddings
        self.pretrain_embeddings = configs.pretrain_embeddings
        self.pretrain_mask_ratio = configs.pretrain_mask_ratio

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        #self.patch_num += 1
        
        if self.config.is_gpt:
            if self.config.from_pretrain_model:
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
        out_layer_depth = 1
        if self.config.predict_tokens and self.config.predict_values:
            out_layer_depth = 3
        if self.do_pretrain_embeddings:
            self.pretrain_logits_layer = output_mlp(
                self.config.d_model,
                self.tokenizer.n_tokens,
                3,
                name="pretrain_recursive_tokens"
            )
        if self.config.recursive:
            if self.config.predict_tokens:
                if self.do_pretrain_embeddings:
                    self.logits_layer = self.pretrain_logits_layer
                else:
                    #self.out_layer = nn.Linear(self.config.d_model, self.tokenizer.n_tokens)
                    self.logits_layer = output_mlp(
                        self.config.d_model,
                        self.tokenizer.n_tokens,
                        3,
                        name="recursive_tokens"
                    )
            if self.config.predict_values:
                #self.out_layer = reduce_layer(self.config.d_model)
                self.values_layer = output_mlp(
                    self.config.d_model,
                    1,
                    squeeze_result=True,
                    name="recursive_values"
                )
        else:
            if self.config.predict_values:
                self.values_layer = output_mlp(
                    self.config.d_model*self.patch_num,
                    self.output_len,
                    out_layer_depth,
                    squeeze_result=True,
                    name="linear_values"
                )
            if self.config.predict_tokens:
                print("SHOULD NOT BE HERE", self.config.d_model*self.patch_num, self.config.pred_len*tokenizer.n_tokens)
                self.logits_layer = output_mlp(
                    self.config.d_model*self.patch_num,
                    self.config.pred_len*tokenizer.n_tokens,
                    out_layer_depth,
                    reshape=(-1, self.config.pred_len, tokenizer.n_tokens),
                    name="linear_tokens"
                )
            #self.out_layer = nn.Linear(self.config.d_model * self.patch_num, self.config.pred_len)
        
        if self.config.freeze_pretrain_model and self.config.from_pretrain_model and not self.pretrain_embeddings:
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
        elif self.pretrain_embeddings:
            for i, (name, param) in enumerate(self.gpt2_enc.named_parameters()):
                param.requires_grad = False
            if self.config.recursive:
                for i, (name, param) in enumerate(self.gpt2_dec.named_parameters()):
                    param.requires_grad = False
 
        for layer in (self.gpt2_enc, self.in_layer):
            layer.to(device=device)
            layer.train()
        if self.tokenizer is not None:
            self.tokenizer.train()
        self.out_layers = []
        if self.config.predict_values:
            self.values_layer.to(device=device)
            self.values_layer.train()
            self.out_layers.append(self.values_layer)
        if self.config.predict_tokens:
            self.logits_layer.to(device=device)
            self.logits_layer.train()
            self.out_layers.append(self.logits_layer)
        if self.config.recursive:
            self.gpt2_dec.to(device=device)
            self.gpt2_dec.train()
            self.out_layers.append(self.gpt2_dec)
        if self.pretrain_embeddings:
            self.pretrain_logits_layer.to(device=device)
            self.pretrain_logits_layer.train()
            self.out_layers.append(self.pretrain_logits_layer)
        
        # Setup tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            print(self.tokenizer.n_tokens, self.config.d_model)
            self.embedding = torch.nn.Embedding(self.tokenizer.n_tokens, self.config.d_model, device=device)
            self.start_token = torch.nn.Parameter(torch.empty((1, 1, self.config.d_model)))
            torch.nn.init.kaiming_uniform_(self.start_token, a=np.sqrt(5))
            self.rearrange1 = 'b l m d -> (b m) d l'
            self.rearrange2 = 'b m n p d -> (b m) n d p'
            self.rearrange3 = '(b m) l d -> b l m d'
        else:
            self.rearrange1 = 'b l m -> b m l'
            self.rearrange2 = 'b m n p -> (b m) n p'
            self.rearrange3 = '(b m) l -> b l m'
        self.cnt = 0


    def forward(self, input):
        B, L, M = input.shape

        x = input
        if self.config.normalize_input:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev

        #print("SHAPE", "0", x.shape)
        x = rearrange(x, 'b l m -> b m l')
        input = rearrange(input, 'b l m -> b m l')
        
        #print("SHAPE", "1", x.shape)
        #TODO: Investigate effects of having padding at the end (duplicating last patch_size/2)
        #x = self.padding_patch_layer(x)
        #input = self.padding_patch_layer(input)
        #print("SHAPE", "2", x.shape)
            
        if self.tokenizer is None or not self.tokenizer.is_patched:
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            input = input.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        #print("SHAPE", "2", x.shape)
        
        if self.tokenizer is not None:
            #print("IN TOKENIZER", x.shape)
            x = self.tokenizer(input)
            x = self.embedding(x)
            #print("OUT TOKENIZER", x.shape)

        if self.tokenizer is None or self.tokenizer.merge_patch:
            #print("BEFORE REARRANGE 2", x.shape)
            x = rearrange(x, self.rearrange2)
            #print("SHAPE", "3", x.shape)

            #print("WTF SHAPE", x.shape)
            outputs = self.in_layer(x)
            #print("SHAPE", "4", outputs.shape)
        else:
            outputs = rearrange(x, 'b m p d -> (b m) p d')

        
        #print("INPUT INTO GPT", outputs.shape)
        # Run encoder
        if self.pretrain_embeddings:
            attn_mask = torch.ones((B*M, outputs.shape[-2])).to(self.device)
            attn_mask_idxs = torch.rand(attn_mask.shape) <= self.pretrain_mask_ratio
            attn_mask[attn_mask_idxs] = 0
        else:
            attn_mask = None
            attn_mask_idxs = None
        
        if self.is_gpt:
            enc_outputs = self.gpt2_enc(inputs_embeds=outputs, attention_mask = attn_mask)
        
        #print("OUTPUT SIZE", len(enc_outputs.hidden_states), enc_outputs.hidden_states[0].shape, enc_outputs.last_hidden_state.shape)
        if self.pretrain_embeddings:
            pred_outputs = enc_outputs.last_hidden_state
        # Run decoder
        elif self.config.recursive:
            is_training = self.gpt2_dec.training
            if is_training:
                self.gpt2_dec.eval()
            with torch.no_grad():
                dec_outputs = enc_outputs.last_hidden_state[:,-1,:].unsqueeze(1) #torch.tile(self.start_token, (x.shape[0], 1, 1))
                for idx in range(self.pred_len-1):
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
            #print("DEC OUT", pred_outputs.shape)
            #pred_tokens = dec_outputs
            pred_outputs = self.out_layer(pred_outputs)
        else:
            pred_outputs = enc_outputs.last_hidden_state
            pred_outputs = pred_outputs.reshape(B*M, -1)
        
        # Transform decoder output to values and get tokens
        #print("SHAPE", pred_outputs.shape, "DOES THIS MATCH d_model FOR INPUT TO MLP OUT")
        #print("SHAPE", pred_outputs.shape)
        pred_value_values = None
        pred_token_logits = None
        pred_token_values = None
        if self.pretrain_embeddings:
            #print("PRED OUTPUTS", pred_outputs.shape)
            pred_token_logits = self.pretrain_logits_layer(pred_outputs)
            #print("PRETRAIN LOGITS", pred_token_logits.shape)
            pred_token_values = self.tokenizer.invert_logits(pred_token_logits, self.embedding)
            pred_token_logits = rearrange(pred_token_logits, '(b m) l p -> b l m p', b=B)
            pred_token_values = rearrange(pred_token_values, '(b m) l -> b l m', b=B)
        else:
            if self.config.predict_values and not self.pretrain_embeddings:
                pred_value_values = self.values_layer(pred_outputs)
                pred_value_values = rearrange(pred_value_values, '(b m) l -> b l m', b=B)
                if False and self.tokenizer is not None:
                    pred_value_tokens = self.tokenizer(pred_value_values)
                else:
                    pred_value_tokens = None
            if self.config.predict_tokens or self.pretrain_embeddings:
                pred_token_logits = self.logits_layer(pred_outputs)
                pred_token_values = self.tokenizer.invert_logits(pred_token_logits, self.embedding)
                print("TKN SHAPE", pred_token_logits.shape, pred_token_values.shape)
                #pred_tokens_logits = rearrange(pred_tokens_logits, '(b m) l d -> b l m d', b=B)
                #pred_tokens_logits = rearrange(pred_tokens_logits, '(b m) l d -> b l m d', b=B)


        """
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
        """
        
        #print("gpt out", outputs.shape)
        # Unnormalize values
        if self.config.normalize_input:
            if self.config.predict_values and not self.pretrain_embeddings:
                pred_value_values = pred_value_values*stdev
                pred_value_values = pred_value_values + means
            if self.config.predict_tokens or self.pretrain_embeddings:
                pred_token_values = pred_token_values*stdev
                pred_token_values = pred_token_values + means
        
        """
        if self.tokenizer is not None and pred_tokens is not None:
            pred_logits = pred_outputs
            pred_tokens = torch.argmax(self.out_layer(pred_logits), dim=-1)
        """

        #print("FINAL OUTPUT", pred_token_values.shape, pred_token_logits.shape)
        #TODO: if predict tokens must have tokenizer, if predict tokens must not normalize
        return ModelPrediction(pred_value_values, pred_token_values, pred_token_logits, attn_mask_idxs)
