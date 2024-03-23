import os
import json
import numpy as np
from matplotlib import pyplot as plt
from bisect import bisect

import utils
from configs import configs
from models import tokenizers
from data_provider.data_factory import data_provider

args = utils.build_default_arg_parser().parse_args()

plot_types, figs, axes = None, None, None
noise_label = f"{args.noise}_{args.noise_var}"
plot_configs = ['OFA', 'cdf_tokens_2500', 'cdf_tokens_5000', 'cdf_tokens_10000', 'convPatch_tokens']
plot_configs = ['OFA', 'naive_tokens', 'cdf_tokens_2500', 'convPatch', 'pretrain_convPatch']
plot_model_folders = {
    'convPatch' : [
        'tk-convPatchc8a9d86e_ls-mse4ae52c71_rc-False_predTkns-False_preTrn-True',
        'tk-convPatche63be0b6_ls-mse4ae52c71_rc-False_predTkns-False_preTrn-True',
        'tk-convPatchf40758b1_ls-mse4ae52c71_rc-False_predTkns-False_preTrn-True'
    ],
    'pretrain_convPatch' : [
        'tk-convPatchc8a9d86e_ls-mse4ae52c71_rc-False_predTkns-False_preTrn-True',
        'tk-convPatche63be0b6_ls-mse4ae52c71_rc-False_predTkns-False_preTrn-True'
    ]
}
plot_labels = {
    'convPatch' : [
        'convPatch: BERT = False | K = 5 | Lg = True',
        'convPatch: BERT = False | K = 5 | Lg = False',
        'convPatch: BERT = False | K = 3 | Lg = True',
    ],
    'pretrain_convPatch' : [
        'convPatch: BERT = True | K = 5 | Lg = True',
        'convPatch: BERT = True | K = 5 | Lg = False',
    ]
}
gridspec = {
    'height_ratios' : [3, 1],
    'hspace' : 0,
    'left' : 0.17,
    'top' : 0.98,
    'right' : 0.96
}
colors = ['k', 'b', 'r', 'green', 'orange', 'pink', 'navy', 'teal']
idx_color = 0
for idx_config, pConfig in enumerate(plot_configs):
    args.model_config = pConfig
    if pConfig in plot_model_folders:
        n_models = len(plot_model_folders[pConfig])
    else:
        n_models = 1
    for idx_model in range(n_models):
        setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}'.format(args.model_id, args.model_config, args.seq_len, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                        args.d_ff, args.embed)
        if args.debug:
            setting = "debug_" + setting

        # Get Configuration
        config = configs.import_config(args.model_config, setting, args, 0)

        # Get tokenizer
        #train_data, train_loader = data_provider(args, 'train')
        tokenizer = tokenizers.get_tokenizer(config, None, 'cpu', hashes_only=True)

        # Get noise curve
        model_folder, checkpoint_model_dir, checkpoint_dir = utils.get_folder_names(
            args, setting, config, tokenizer, has_itr=False
        )
        if args.model_config in plot_model_folders:
            model_folder = plot_model_folders[pConfig][idx_model]
            label = plot_labels[pConfig][idx_model]
        else:
            label = pConfig

        with open(os.path.join(checkpoint_dir, f"noise_{args.noise}_{args.noise_var}_results.json"), "r") as file:
            noise_results = json.load(file)
        if plot_types is None:
            plot_types, figs, axes = [], [], []
            for key in noise_results.keys():
                if 'std' not in key:
                    plot_types.append(key)
                    fig, ax = plt.subplots(2, 1, gridspec_kw=gridspec)
                    figs.append(fig)
                    axes.append(ax)

        # Plot
        plt_dir = os.path.join("plots", setting, model_folder, noise_label)
        if not os.path.exists(plt_dir):
            os.makedirs(plt_dir)
        scales = np.array(noise_results['std_scales'])
        all_order_idx = np.argsort(scales)
        if scales[all_order_idx[0]] == 0:
            order_idx = all_order_idx[1:]
        else:
            order_dix = all_order_idx
        if args.plot_max < np.amax(scales):
            order_idx = order_idx[:bisect(scales[order_idx], args.plot_max)]
        for idx, key in enumerate(plot_types):
            stds = noise_results[key+'_std']
            if np.isnan(stds[0]):
                stds = np.zeros_like(stds)
            plot_scales = np.array(scales)[order_idx] 
            plot_ratios = np.array(noise_results[key])[order_idx]/noise_results[key][all_order_idx[0]] 
            plot_ratios_std = stds[order_idx]/noise_results[key][all_order_idx[0]]
            
            fig, ax = plt.subplots(2, 1, gridspec_kw=gridspec)
            ax[0].errorbar(plot_scales, np.array(noise_results[key])[order_idx])
            ax[1].errorbar(plot_scales, plot_ratios, yerr=plot_ratios_std)
            ax[0].set_ylabel(key)
            ax[1].set_ylabel(f"{key} Ratio")
            ax[1].set_xlabel("noise standard deviation scale")
            #plt.tight_layout()
            fig.savefig(os.path.join(plt_dir, key+'.png'))
            
            axes[idx][0].errorbar(
                plot_scales,
                np.array(noise_results[key])[order_idx],
                yerr=plot_ratios_std,
                label=label,
                color=colors[idx_color])
            axes[idx][1].errorbar(
                plot_scales,
                plot_ratios,
                yerr=plot_ratios_std,
                label=label,
                color=colors[idx_color])
            axes[idx][0].set_xlim(plot_scales[0], min(plot_scales[-1], args.plot_max))
            axes[idx][1].set_xlim(plot_scales[0], min(plot_scales[-1], args.plot_max))
        idx_color += 1

plot_dir = os.path.join("plots", noise_label)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
for idx, ax in enumerate(axes):
    ax[0].set_ylabel(plot_types[idx])
    ax[1].set_ylabel(f"{plot_types[idx]} Ratios")
    ax[0].xaxis.set_visible(False)
    ax[1].set_xlabel("noise standard deviation scale")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    #ax[0].legend(loc='lower right', fontsize=10, framealpha=1.)
    #plt.tight_layout()
    figs[idx].savefig(os.path.join(plot_dir, f"{plot_types[idx]}.png"))