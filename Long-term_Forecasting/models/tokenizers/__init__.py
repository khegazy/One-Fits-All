from .cdf import CDFbinning
from .convolution_patch import ConvolutionPatches
from .naive import NaiveBinning

tokenizer_dict = {
    "cdf" : CDFbinning,
    "convpatch" : ConvolutionPatches,
    "naive" : NaiveBinning
}

def get_tokenizer(config, train_data, device, embed_size=None):
    if config.tokenizer is None:
        return None
    
    config.tokenizer = config.tokenizer.lower()
    if config.tokenizer not in tokenizer_dict:
        raise ValueError(f"get_tokenizer cannot handle tokenizer {config.tokenizer}, either add it to tokenizer_dict or use {list(tokenizer_dict.keys())}") 

    if train_data is None:
        return tokenizer_dict[config.tokenizer](
            config.tokenizer_config, None, device
        )
    else:
        return tokenizer_dict[config.tokenizer](
            config.tokenizer_config, train_data.data_x, device
        )