from data_provider.data_factory import data_provider
import utils
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from configs import configs
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models import tokenizers

import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import wandb

import os
import sys
import time
import json

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')


args = utils.build_default_arg_parser().parse_args()

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)





# Get configuration 
setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}'.format(args.model_id, args.model_config, args.seq_len, args.label_len, args.pred_len,
                                                                args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                args.d_ff, args.embed)
if args.debug:
    setting = "debug_" + setting
device = torch.device('cuda:0')

# Get Data
if args.freq == 0:
    args.freq = 'h'

print("getting data")
train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')
print("got data")

if args.freq != 'h':
    args.freq = utils.SEASONALITY_MAP[test_data.freq]
    print("freq = {}".format(args.freq))


#########################
#####  Train Model  #####
#########################

if args.train:
    print("TRAINING")
    # Get Configuration
    config = configs.import_config(args.model_config, setting, args, args.itr, is_training=True)
    # Get tokenizer
    print("getting tokenizer")
    tokenizer = tokenizers.get_tokenizer(config, train_data, device)

    # Get model
    print("getting model")
    if args.model == 'PatchTST':
        model = PatchTST(config, device, tokenizer=tokenizer)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(config, device, tokenizer=tokenizer)
        model.to(device)
    else:
        model = GPT4TS(config, device, tokenizer=tokenizer)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)


    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    
    # Logging and checkpointing
    tk_label = "None" if tokenizer is None else tokenizer.get_hash()
    ls_label = f"{config.loss_config['type']}{utils.get_hash(json.dumps(config.loss_config, sort_keys=True))}"
    model_folder = f"tk-{tk_label}_ls-{ls_label}_rc-{config.recursive}_predTkns-{config.predict_tokens}_preTrn-{config.pretrain}"
    checkpoint_dir = os.path.join(args.checkpoints, setting, model_folder, f"itr_{args.itr}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_handler = utils.CheckpointHandler(
        checkpoint_dir, "state", args.keep_checkpoints
    )
    with open(os.path.join(checkpoint_dir, "loss_config.json"), "w") as f:
        json.dump(config.loss_config, f)
    if config.tokenizer_config is not None:
        with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config.tokenizer_config, f)
    print("checkpoint dir:", checkpoint_dir)
    
    start_epoch = 0
    min_valid_loss = np.inf
    if args.restart_latest:
        prev_train_values = checkpoint_handler.load_latest(
            state=utils.CheckpointState(model, model_optim, None),
            device=device,
            is_optimal=False# not args.train,
        )
        if prev_train_values is not None:
            start_epoch = prev_train_values["epochs"]
            prev_train_steps = prev_train_values["train_steps"]
            min_valid_loss = prev_train_values["min_valid_loss"]
            #ema_train_metrics.load_metrics(prev_train_values["metrics"])
            print("Model is reloaded from previous checkpoint")
        elif args.require_restart:
            raise ValueError("Cannot load required checkpoint.")
        else:
            print("Model is starting from scratch")
        
        if start_epoch >= args.train_epochs:
            sys.exit(0) 
    is_wandb = True
    run = utils.setup_wandb(config, checkpoint_dir, setting, args.debug, itr=args.itr)

    time_now = time.time()
    train_steps = len(train_loader)
    wandb_iter = start_epoch
    for epoch in range(start_epoch, args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), disable=args.is_slurm):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            predictions = model(batch_x, args.itr)

            values = predictions.values[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(values, batch_y)
            train_loss.append(loss.item())
            if i % 500 == 0 and is_wandb:
                metrics = {
                    "train_loss" : np.mean(np.array(train_loss)[-10:])
                }
                wandb.log(metrics, step=wandb_iter)

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                """
                checkpoint_handler.save(
                    state=utils.CheckpointState(model, model_optim, None),
                    epochs=epoch,
                    train_steps=epoch,
                    min_valid_loss=min_valid_loss.item(),
                    #metrics=ema_train_metrics.get_metrics(),
                    is_optimal=False,
                )
                """
                if args.debug:
                    break
                time_now = time.time()

            loss.backward()
            model_optim.step()
            wandb_iter = wandb_iter + 1

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        print("should be saving")
        checkpoint_handler.save(
            state=utils.CheckpointState(model, model_optim, None),
            epochs=epoch+1,
            train_steps=epoch,
            min_valid_loss=min_valid_loss,
            #metrics=ema_train_metrics.get_metrics(),
            is_optimal=False,
        )
        

        train_loss = np.average(train_loss)
        valid_metrics = utils.evaluate_dataset(
            model, vali_loader, criterion, args, device, args.itr, is_training=True
        )
        #valid_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, valid_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, valid_metrics['loss']))
        if is_wandb:
            metrics = {
                "train_loss" : train_loss,
                "valid_loss" : valid_metrics['loss'],
                "epoch" : epoch+1,    
            }
            wandb.log(metrics, step=wandb_iter)
        
        if valid_metrics['loss'] < min_valid_loss:
            min_valid_loss = valid_metrics['loss']
            checkpoint_handler.save(
                state=utils.CheckpointState(model, model_optim, None),
                epochs=epoch+1,
                train_steps=epoch,
                min_valid_loss=min_valid_loss,
                is_optimal=True
            )

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(valid_metrics['loss'], model, checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

######################
#####  Evaluate  #####
######################
print("Evaluating Test Set")
if args.itr is None:
    itr_range = list(range(args.n_itr))
else:
    itr_range = [args.itr]

if args.noise_curve:
    noise_scales = np.array([0.1, 0.5])#), 1., 5., 10.])
    is_wandb = False
    noise_history = None
else:
    noise_scales = np.array([args.noise_scale])
    is_wandb = True

for ns_scale in noise_scales:
    test_loader.dataset.add_noise_transform(
        args.noise, args.noise_drift, args.noise_var, ns_scale
    )
    metrics = []
    for ii in itr_range:
        # Get Configuration
        config = configs.import_config(args.model_config, setting, args, ii)
        # Get tokenizer
        tokenizer = tokenizers.get_tokenizer(config, train_data, device)

        # Get model
        if args.model == 'PatchTST':
            model = PatchTST(config, device, tokenizer=tokenizer)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(config, device, tokenizer=tokenizer)
            model.to(device)
        else:
            model = GPT4TS(config, device, tokenizer=tokenizer)

        tk_label = "None" if tokenizer is None else tokenizer.get_hash()
        ls_label = f"{config.loss_config['type']}{utils.get_hash(json.dumps(config.loss_config, sort_keys=True))}"
        model_folder = f"tk-{tk_label}_ls-{ls_label}_rc-{config.recursive}_predTkns-{config.predict_tokens}_preTrn-{config.pretrain}"
        checkpoint_model_dir = os.path.join(args.checkpoints, setting, model_folder)
        checkpoint_handler = utils.CheckpointHandler(
            os.path.join(checkpoint_model_dir, f"itr_{ii}"),
            "state",
            args.keep_checkpoints
        )
        prev_train_values = checkpoint_handler.load_latest(
            state=utils.CheckpointState(model, None, None),
            device=device,
            is_optimal=True,
            is_expected=True, 
        )
        if args.noise_curve and noise_history is None:
            filename = os.path.join(checkpoint_model_dir, "noise_curve.json")
            if os.path.exists(filename):
                with open(filename, "r") as file:
                    noise_history = json.load(file)
            noise_history = None

        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
    
        metrics.append(utils.evaluate_dataset(
            model, test_loader, criterion, args, device, ii, as_dict=False
        ))

        #best_model_path = path + '/' + 'checkpoint.pth'
        #model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        #mse, mae = test(model, test_data, test_loader, args, device, ii)
        #mses.append(mse)
        #maes.append(mae)
    if len(metrics) == 1:
        total_metrics = utils.metrics_to_dict(metrics[0])
        keys = list(total_metrics.keys())
        for k in keys:
            total_metrics[k+"_std"] = np.nan
    else:    
        metrics = torch.stack(metrics)
        total_metrics = utils.metrics_to_dict(torch.mean(metrics, dim=0), torch.std(metrics, dim=0))
    
    # Record eval metrics
    if args.noise_curve:
        if noise_history is None:
            noise_history = {}
            noise_history["std_scale"] = [ns_scale]
            for k, v in total_metrics.items():
                noise_history[k] = [v]
        else:
            noise_history["std_scale"].append(ns_scale)
            for k, v in total_metrics.items():
                noise_history[k].append(v)

    mse_message = "mse_mean = {:.4f}, mse_std = {:.4f}".format(total_metrics['mse'], total_metrics['mse_std'])
    mae_message = "mae_mean = {:.4f}, mae_std = {:.4f}".format(total_metrics['mae'], total_metrics['mae_std'])
    print(mse_message)
    print(mae_message)

# Saving results
if args.itr is not None:
    checkpoint_dir = os.path.join(checkpoint_model_dir, f"itr_{args.itr}")
if is_wandb:
    run = utils.setup_wandb(config, checkpoint_dir, setting, args.debug)
    metrics = {
        "test_loss" : total_metrics['loss'],
        "test_mse" : total_metrics['mse'],
        "test_mae" : total_metrics['mae'],
        "test_rmse" : total_metrics['rmse'],
        "test_mape" : total_metrics['mape'],
        "test_mspe" : total_metrics['mspe'],
        "test_smape" : total_metrics['smape'],
        "test_nd" : total_metrics['nd']
    }
    wandb.log(metrics, step=prev_train_values["epochs"])

if args.noise_curve:
    with open(os.path.join(checkpoint_model_dir, "noise_curve.json"), "w") as file:
        json.dump(noise_history, file)
else:
    with open(os.path.join(checkpoint_dir, "test_results.json"), "w") as file:
        json.dump(total_metrics, file)
    with open(os.path.join(checkpoint_dir, "test_results.txt"), "w") as file:
        #file.write("iter \t   MSE \t   MAE\n")
        file.write(f"loss \t{total_metrics['loss']:.4f} +/- {total_metrics['loss_std']:.4f}\n")
        for lbl in utils.get_labels():
            file.write(f"{lbl} \t{total_metrics[lbl]:.4f} +/- {total_metrics[lbl+'_std']:.4f}\n")
        file.write("\n")