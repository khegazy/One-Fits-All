import os
import json
import numpy as np
from matplotlib import pyplot as plt

import utils
from configs import configs
from models import tokenizers
from data_provider.data_factory import data_provider

args = utils.build_default_arg_parser().parse_args()
setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}'.format(args.model_id, args.model_config, args.seq_len, args.label_len, args.pred_len,
                                                                args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                args.d_ff, args.embed)
if args.debug:
    setting = "debug_" + setting

# Get Configuration
config = configs.import_config(args.model_config, setting, args, 0)

# Get tokenizer
#train_data, train_loader = data_provider(args, 'train')
tokenizer = tokenizers.get_tokenizer(config, None, 'cpu')

# Get noise curve
tk_label = "None" if tokenizer is None else tokenizer.get_hash()
ls_label = f"{config.loss_config['type']}{utils.get_hash(json.dumps(config.loss_config, sort_keys=True))}"
model_folder = f"tk-{tk_label}_ls-{ls_label}_rc-{config.recursive}_predTkns-{config.predict_tokens}_preTrn-{config.pretrain}"
checkpoint_model_dir = os.path.join(args.checkpoints, setting, model_folder)
with open(os.path.join(checkpoint_model_dir, "noise_curve.json"), "r") as file:
    noise_results = json.load(file)

# Plot
plt_dir = os.path.join("plots", setting, model_folder)
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
scales = noise_results['std_scale']
order_idx = np.argsort(scales)
for key, val in noise_results.items():
    if "std" in key:
        continue
    stds = noise_results[key+'_std']
    if np.isnan(stds[0]):
        stds = np.zeros_like(stds)
    fig, ax = plt.subplots()
    ax.errorbar(scales, val, yerr=stds)
    ax.set_ylabel(key)
    ax.set_xlabel("noise standard deviation scale")
    plt.tight_layout()
    fig.savefig(os.path.join(plt_dir, key+'.png'))
