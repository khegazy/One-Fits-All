import json
import torch
import numpy as np
from utils.hashes import get_hash


class NaiveBinning(torch.nn.Module):
    def __init__(self, config, train_data, device, hashes_only=False):
        super().__init__()
        #self.min_val = np.amin(train_data)
        #self.max_val = np.amax(train_data)
        self.is_patched = False
        self.merge_patch = True
        self.n_tokens = config["n_tokens"]
        self.cdf_cut = config['cdf_cut']
        self.token_dtype = torch.int64
        self.hashes_only = hashes_only
        if hashes_only:
            return
        
        histogram, bins = np.histogram(train_data, bins=100000)
        cdf = np.cumsum(histogram)
        cdf = cdf/cdf[-1]
        self.min_val = bins[np.argmin(np.abs(cdf - self.cdf_cut))]
        self.max_val = bins[np.argmin(np.abs(cdf - (1 - self.cdf_cut)))]
        self.delta = (self.max_val - self.min_val)/(self.n_tokens - 1)
        #self.token_dtype = torch.int32 if "dtype" not in config\
        #    else get_torch_dtype(config["dtype"]) 
        print(f"token limits | delta: {self.min_val} / {self.max_val} | {self.delta}") 

    def get_hash(self):    
        tokenizer_args = {
            'name' : 'naivebinning',
            'cdf_cut' : self.cdf_cut,
            'n_tokens' : self.n_tokens,
        }
        self.hash = get_hash(json.dumps(tokenizer_args, sort_keys=True))
        return f"naive{self.hash}"
    
    def invert_logits(self, logits, embeddings=None):
        return self.invert_tokens(torch.argmax(logits, -r))

    def invert_tokens(self, tokens):
        return self.min_val + tokens*self.delta

    def __len__(self):
        return self.n_tokens
    
    def forward(self, input, return_logits=False):
        if self.hashes_only:
            return ValueError("Cannot run tokenizer with in hashes_only mode")
        input = input - self.min_val
        input = input/self.delta
        input = input.to(torch.int64)
        input[input < 0] = 0
        input[input >= self.n_tokens] = self.n_tokens - 1
        input = input.to(self.token_dtype)

        if return_logits:
            logits = torch.zeros(
                (*input.shape, self.n_tokens),
                dtype=torch.float
            )
            logits[:,:,:,input] = 1e5
            return logits

        return input.to(self.token_dtype)