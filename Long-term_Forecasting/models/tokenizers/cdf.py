import json
import torch
import numpy as np

from utils.hashes import get_hash


class CDFbinning(torch.nn.Module):
    def __init__(self, config, train_data, device):
        super().__init__()
        self.n_tokens = config["n_tokens"]
        self.is_patched = False
        self.merge_patch = True
        if train_data is None:
            return None
        
        histogram, bins = np.histogram(train_data, bins=100000)
        cdf = np.cumsum(histogram)
        cdf = cdf/cdf[-1]
        self.cdf_delta = 1./self.n_tokens
        self.bin_delta = bins[2] - bins[1]
        
        self.token_values = torch.empty(self.n_tokens).to(device)
        idx_token = 0
        for idx_cdf, val in enumerate(cdf):
            if idx_token == self.n_tokens:
                break
            if val >= idx_token*self.cdf_delta:
                delta_cdf = val - cdf[idx_cdf-1]
                ratio = (val - idx_token*self.cdf_delta)/delta_cdf
                self.token_values[idx_token] = bins[idx_cdf+1]\
                    - ratio*self.bin_delta
                idx_token = idx_token + 1

        self.min_val = self.token_values[0].cpu().item()
        self.max_val = self.token_values[-1].cpu().item()
        self.delta = torch.mean(
            torch.abs(self.token_values[:-1] - self.token_values[1:])
        ).cpu().item() 
        self.token_dtype = torch.int32 if "dtype" not in config\
            else get_torch_dtype(config["dtype"]) 
        print(f"token limits | delta: {self.min_val} / {self.max_val} | {self.delta}") 
    
    def __len__(self):
        return self.n_tokens
    
    def get_hash(self):    
        tokenizer_args = {
            "name" : "cdfbinning",
            "n_tokens" : self.n_tokens,
        }
        self.hash = get_hash(json.dumps(tokenizer_args, sort_keys=True))
        return f"cdf{self.hash}"
    
    def invert_logits(self, logits, embeddings=None):
        return self.invert_tokens(torch.argmax(logits, -1))
    
    def invert_tokens(self, tokens, emeddings=None):
        return self.token_values[tokens]

    def forward(self, input):
        token_idxs = torch.searchsorted(self.token_values, input)
        token_idxs = token_idxs.to(self.token_dtype)
        token_idxs[token_idxs>=self.n_tokens] = self.n_tokens - 1
        tokens = token_idxs - torch.argmin(
            torch.abs(torch.concatenate(
                [
                    (self.token_values[token_idxs] - input).unsqueeze(-1),
                    (input - self.token_values[token_idxs-1]).unsqueeze(-1),
                ],
                dim=-1)),
            dim=-1
        )

        return tokens.to(self.token_dtype)

