import argparse

def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='GPT4TS')

    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model_config', type=str, required=True, default='OFA')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--is_slurm', default=False, action='store_true')
    parser.add_argument('--keep_checkpoints', default=False, action='store_true')
    parser.add_argument('--restart_latest', default=False, action='store_true')
    parser.add_argument('--require_restart', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', default=False, action='store_true')

    parser.add_argument('--noise', type=str, default=None)
    parser.add_argument('--noise_drift', type=str, default="identity")
    parser.add_argument('--noise_var', type=str, default="identity")
    parser.add_argument('--noise_scale', type=float, default=1.)
    parser.add_argument('--noise_curve', default=False, action='store_true')

    parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
    parser.add_argument('--data_path', type=str, default='traffic.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=100)

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--pretrain_embeddings', default=False, action='store_true')
    parser.add_argument('--pretrain_mask_ratio', type=float, default=0.15)

    parser.add_argument('--loss_func', type=str, default='mse')
    #parser.add_argument('--pretrain', type=bool, default=True)
    #parser.add_argument('--freeze', type=bool, default=True)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)

    parser.add_argument('--n_itr', type=int, default=3)
    parser.add_argument('--itr', type=int, default=None)
    parser.add_argument('--cos', type=int, default=0)

    parser.add_argument('--plot_pred', default=False, action='store_true')
    parser.add_argument('--plot_seed', type=int, default=101)
    parser.add_argument('--plot_max', type=float, default=1e100)

    return parser

    
