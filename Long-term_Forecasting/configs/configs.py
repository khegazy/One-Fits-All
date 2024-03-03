import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class RunConfig:
    name : str
    tag : str
    model_id : str
    seq_len : int
    label_len : int
    pred_len : int
    d_model : int
    n_heads : int
    e_layers : int
    gpt_layers : int
    d_ff : int
    embded : int
    stride : int
    patch_size : int
    batch_size : int
    iter : int
    is_gpt : bool = True
    normalize_input : bool = True
    tokenizer : str = None
    tokenizer_config : Dict = None
    predict_values : bool = True
    value_loss_config : Dict = None
    predict_tokens : bool = False
    token_loss_config : Dict = None
    pretrain_embeddings : bool = False
    pretrain_mask_ratio : float = None
    embedding_value_loss_config : Dict = None
    embedding_token_loss_config : Dict = None
    do_pretrain_embeddings : bool = False
    from_pretrain_model : bool = False
    freeze_pretrain_model : bool = False
    recursive : bool = False

def import_yaml(address, is_expected=False):
    if os.path.exists(address):
        with open(address, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
        return loaded_yaml
    elif is_expected:
        raise ImportError(f"Cannot find required file {address}")
    else:
        ImportWarning(f"Cannot find file {address}, still running")
        return {}

def import_config(config_name, name, args, iter, tag="", dir="./configs/", is_expected=True, is_training=False):
    if is_training:
        if iter is None or iter >= args.n_itr:
            raise ValueError("Flag itr must not be None when training and must be < n_itr.")
    filename = config_name
    filename += f"_{tag}" if tag != "" else ""
    filename += ".yaml"
    yaml_config = import_yaml(os.path.join(dir, filename), is_expected)
    
    for k in yaml_config.keys():
        if yaml_config[k] == "None" or yaml_config[k] == "none":
            yaml_config[k] = None
    config = RunConfig(
        name=name,
        tag=tag, 
        model_id = args.model_id,
        seq_len = args.seq_len,
        label_len = args.label_len,
        pred_len = args.pred_len,
        d_model = args.d_model,
        n_heads = args.n_heads,
        e_layers = args.e_layers,
        patch_size = args.patch_size,
        stride = args.stride,
        gpt_layers = args.gpt_layers,
        d_ff = args.d_ff,
        embded = args.embed,
        batch_size = args.batch_size,
        iter = iter,
        pretrain_embeddings = args.pretrain_embeddings,
        pretrain_mask_ratio = args.pretrain_mask_ratio,
        **yaml_config
    )

    if config.tokenizer_config is not None:
        config.tokenizer_config['patch_size'] = args.patch_size
        config.tokenizer_config['stride'] = args.stride
        config.tokenizer_config['embed_size'] = args.d_model
    
    return config


    

