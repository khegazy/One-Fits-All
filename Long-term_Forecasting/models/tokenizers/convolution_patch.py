import json
import torch
import numpy as np
from einops import rearrange

from utils.hashes import get_hash

class ConvolutionPatches(torch.nn.Module):
    def __init__(self, config, train_data, device, hashes_only=False) -> None:
        super().__init__()
        self.n_tokens = config['n_tokens']
        self.embed_size = config['embed_size']
        self.n_channels = 1
        self.kernel = config['kernel']
        self.patch_size = config['patch_size']
        self.n_layers = config['n_layers']
        self.n_filters = config['n_filters']
        self.stride = config['stride']
        self.is_patched = True
        self.merge_patch = False
        self.legendre_init = config['legendre_init']
        self.hashes_only = hashes_only
        if self.hashes_only:
            return
        if self.n_layers == 1:
            self.n_filters = 1
            self.kernel = self.patch_size
        if train_data is None:
            return None
        
        self.activation = torch.nn.Tanh()
        
        # Model to calculate tokens from data
        self.fwd_layers = torch.nn.ModuleList()
        if self.n_layers > 1:
            self.fwd_layers.append(
                torch.nn.Conv1d(
                    self.n_channels,
                    self.n_filters,
                    self.kernel,
                    padding='same',
                    device=device
                )
            )
            for idx in range(max(0, self.n_layers-2)):
                self.fwd_layers.append(
                    torch.nn.Conv1d(
                        self.n_filters,
                        self.n_filters,
                        self.kernel,
                        padding='same',
                        device=device
                    ) 
                )
        self.fwd_layers.append(
            torch.nn.Conv1d(
                self.n_filters,
                self.n_tokens,
                self.patch_size,
                stride=self.stride,
                padding='valid',
                device=device
            )
        )

        # Model to calculate time series from tokens
        if self.stride % 2 != 0:
            raise ValueError("Currently requires stride to be a multiple of 2")
        n_upconv = self.stride//2 - 1
        self.inv_layers = torch.nn.ModuleList()
        for idx in range(n_upconv - 1):
            self.inv_layers.append(
                torch.nn.ConvTranspose1d(
                    self.embed_size//2**idx,
                    self.embed_size//2**(idx+1),
                    3,
                    stride=2,
                    device=device
                )
            )
        self.inv_layers.append(
            torch.nn.ConvTranspose1d(
                self.embed_size//2**(n_upconv-1),
                self.embed_size//2**n_upconv,
                4,
                stride=2,
                device=device
            )
        )

        if self.legendre_init:
            x = torch.tile(torch.linspace(0,1,self.kernel), (self.n_filters, 1))
            n = torch.tile(torch.arange(self.n_filters), (self.kernel, 1)).transpose(0,1)
            leg_init = torch.special.legendre_polynomial_p(x, n)
            leg_init  = leg_init/torch.sum(leg_init, dim=-1, keepdim=True)

            sd = self.state_dict()
            sd['fwd_layers.0.weight'] = torch.tile(leg_init.unsqueeze(1), (1, self.n_channels, 1))
            self.load_state_dict(sd)

    def get_hash(self):
        tokenizer_args = {
            'name' : 'convPatch',
            'kernel' : self.kernel,
            'n_tokens' : self.n_tokens,
            'patch_size' : self.patch_size,
            'n_layers' : self.n_layers,
            'n_filters' : self.n_filters,
            'legendre_init' : self.legendre_init
        }
        self.hash = get_hash(json.dumps(tokenizer_args, sort_keys=True))
        return f"convPatch{self.hash}"
    
    def forward(self, input, return_logits=False):
        if self.hashes_only:
            return ValueError("Cannot run tokenizer with in hashes_only mode")
        #print("INP SHAPE", input.shape)
        B, M, L = input.shape
        input = rearrange(input, 'b m l -> (b m) l')
        input = torch.unsqueeze(input, dim=-2)
        #print("INPUT", input.shape)
        x = self.fwd_layers[0](input)
        if self.n_layers > 1:
            x = self.activation(x)
            for lyr in self.fwd_layers[1:-1]:
                #print("XXXXXXXX", x.shape)
                #print("RES SHAPES", x.shape, lyr(x).shape)
                x = x + lyr(x)
                x = self.activation(x)
            #print("XXXXXXXX", x.shape)
            x = self.fwd_layers[-1](x)
        if return_logits:
            return rearrange(x, '(b m) c l -> b m l c', b=B)
            #return rearrange(x, '(b m) c l -> b c m l', b=B)
        else: 
            x = torch.argmax(x, dim=-2)
            #print("OUT", x.shape)
            return rearrange(x, '(b m) l -> b m l', b=B)
    
    def invert_logits(self, logits, embeddings):
        #print("INPUT LOGITS", logits.shape)
        return self.invert_tokens(torch.argmax(logits, -1), embeddings)
    
    def invert_tokens(self, tokens, embeddings):
        #print("INPUT TOKENS", tokens.shape)
        x = embeddings(tokens).transpose(1,2)
        for idx, lyr in enumerate(self.inv_layers[:-1]):
            #print("XXXXXXXX", idx, x.shape)
            #print("RES SHAPES", x.shape, lyr(x).shape)
            x = lyr(x)
            x = self.activation(x)
        x = self.inv_layers[-1](x)
        return torch.sum(x, -2)
