from data_provider.data_factory import data_provider
import utils
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from configs import configs
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models import tokenizers
from utils import metrics

import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import wandb
from copy import copy

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
if args.seed is not None:
    seed = args.seed
elif args.itr is None:
    seed = fix_seed
else:
    seed = 100 + args.itr*50
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)





# Get configuration
if args.model_config == "OFA":
    args.label_len = 0 
setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}'.format(args.model_id, args.model_config, args.seq_len, args.label_len, args.pred_len,
                                                                args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                args.d_ff, args.embed)
if args.debug:
    setting = "debug_" + setting
device = torch.device('cuda')
print(f"Using config {args.model_config}")

# Get Data
if args.freq == 0:
    args.freq = 'h'
train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')
if args.freq != 'h':
    args.freq = utils.SEASONALITY_MAP[test_data.freq]
    print("freq = {}".format(args.freq))


#########################
#####  Train Model  #####
#########################

if args.train or args.pretrain_embeddings:
    
    print("TRAINING")
    # Get Configuration
    config = configs.import_config(
        args.model_config, setting, args, args.itr, is_training=True)
    # Get tokenizer
    print("getting tokenizer")
    tokenizer = tokenizers.get_tokenizer(config, train_data, device)

    # Get model
    print("Getting Model")
    if args.model == 'PatchTST':
        model = PatchTST(config, device, tokenizer=tokenizer)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(config, device, tokenizer=tokenizer)
        model.to(device)
    else:
        model = GPT4TS(config, device, tokenizer=tokenizer)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    print("Getting Optimizer")
    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)


    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    criterion = metrics.get_loss(config, tokenizer) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    
    # Logging and checkpointing
    model_folder, checkpoint_dir = utils.get_folder_names(args, setting, config, tokenizer)
    """
    tk_label = "None" if tokenizer is None else tokenizer.get_hash()
    ls_val_label, ls_tkn_label = "", ""
    if config.predict_values:
        ls_val_label = f"{config.value_loss_config['type']}{utils.get_hash(json.dumps(config.value_loss_config, sort_keys=True))}"
    if config.predict_tokens:
        ls_tkn_label = f"{config.token_loss_config['type']}{utils.get_hash(json.dumps(config.token_loss_config, sort_keys=True))}"
        if config.predict_values:
            ls_tkn_label = "_" + ls_tkn_label
    model_folder = f"tk-{tk_label}_ls-{ls_val_label}{ls_tkn_label}_rc-{config.recursive}_predTkns-{config.predict_tokens}_preTrn-{config.pretrain}"
    checkpoint_dir = os.path.join(args.checkpoints, setting, model_folder, f"itr_{args.itr}")
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_handler = utils.CheckpointHandler(
        checkpoint_dir,
        "state",
        args.keep_checkpoints,
        is_pretrain=args.pretrain_embeddings
    )
    if config.value_loss_config:
        with open(os.path.join(checkpoint_dir, "value_loss_config.json"), "w") as f:
            json.dump(config.value_loss_config, f)
    if config.token_loss_config:
        with open(os.path.join(checkpoint_dir, "token_loss_config.json"), "w") as f:
            json.dump(config.token_loss_config, f)
    if config.tokenizer_config is not None:
        with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config.tokenizer_config, f)
    print("checkpoint dir:", checkpoint_dir)
    
    start_epoch = 0
    min_valid_loss = np.inf
    if args.restart_latest:
        #TODO: integrate the lr_scheduler into checkpointing
        prev_train_values = checkpoint_handler.load_latest(
            state=utils.CheckpointState(model, model_optim, None),
            device=device,
            is_optimal=False,# not args.train,
            is_expected=args.require_restart
        )
        if prev_train_values is not None:
            start_epoch = prev_train_values["epochs"]
            prev_train_steps = prev_train_values["train_steps"]
            min_valid_loss = prev_train_values["min_valid_loss"]
            #ema_train_metrics.load_metrics(prev_train_values["metrics"])
    else:
        print("Starting model from scratch")

    is_wandb = True
    run = utils.setup_wandb(config, checkpoint_dir, setting, args.debug, itr=args.itr)
    """
    print("FORWARD WEIGHTS")
    for idx, lyr in enumerate(tokenizer.fwd_layers):
        print("FWD: "+str(idx), lyr.weight[:3,:2,:4])
    print("BACKWARD WEIGHTS")
    for idx, lyr in enumerate(tokenizer.inv_layers):
        print("INV: "+str(idx), lyr.weight[:3,:2,:4])
    adsf
    """

    time_now = time.time()
    train_steps = len(train_loader)
    for epoch in range(start_epoch, args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), disable=args.is_slurm):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

            if args.debug and i > 3:
                break
            #T0 = time.time()
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            #print("get data time", time.time() - T0)
            
            #t0 = time.time()
            predictions = model(batch_x)
            #print("Model Run time", time.time() - t0)

            #values = predictions.values[:, -args.pred_len:, :]
            #logits = predictions.logits[:, -args.pred_len:, :]
            #batch_y = batch_y[:, -args.pred_len:, :].to(device)
            if args.pretrain_embeddings:
                #t0 = time.time()
                loss, loss_results = criterion(predictions, batch_x)
                #print("Loss PT time", time.time() - t0)
            else:
                #t0 = time.time()
                loss, loss_results = criterion(predictions, batch_y)
                #print("Loss time", time.time() - t0)
            train_loss.append(loss.item())
            #print("LOSS", i, loss)

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


            #t0 = time.time()
            loss.backward()
            model_optim.step()
            #print("update time", time.time() - t0)
            #print("Iteration time", time.time() - T0)

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        checkpoint_handler.save(
            state=utils.CheckpointState(model, model_optim, None),
            epochs=epoch+1,
            train_steps=epoch,
            min_valid_loss=float(min_valid_loss),
            #metrics=ema_train_metrics.get_metrics(),
            is_optimal=False,
        )
        

        train_loss = np.average(train_loss[-50:])
        valid_metrics = utils.evaluate_dataset(
            model, 
            tokenizer,
            vali_loader,
            criterion,
            args,
            device,
            args.itr,
            is_training=True,
            pretrain_embeddings=args.pretrain_embeddings,
            debug=args.debug
        )
        #valid_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, valid_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, valid_metrics['loss']))
        if is_wandb:
            log_metrics = {
                k : v for k, v in valid_metrics.items()\
                    if 'std' not in k and\
                    ('loss' in k or 'mse' in k or 'mae' in k\
                     or 'cross_entropy' in k or 'accuracy' in k) 
            }
            log_metrics['train_loss'] = train_loss
            log_metrics['valid_loss'] = valid_metrics['loss']
            log_metrics['epoch'] = epoch + 1
            del log_metrics['loss']
            wandb.log(log_metrics, step=epoch+1)
        
        if valid_metrics['loss'] < min_valid_loss:
            min_valid_loss = valid_metrics['loss']
            checkpoint_handler.save(
                state=utils.CheckpointState(model, model_optim, None),
                epochs=epoch+1,
                train_steps=epoch,
                min_valid_loss=float(min_valid_loss),
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
#args.label_len = 0
#train_data, train_loader = data_provider(args, 'train')
#vali_data, vali_loader = data_provider(args, 'val')
"""
test_data, test_loader = data_provider(args, 'test')
if args.freq != 'h':
    args.freq = utils.SEASONALITY_MAP[test_data.freq]
    print("freq = {}".format(args.freq))
"""
data_loader = test_loader
#data_loader = vali_loader
if args.itr is None:
    itr_range = list(range(args.n_itr))
else:
    itr_range = [args.itr]

if args.noise_curve:
    if args.noise is None:
        raise ValueError("Must specify --noise tag when using --noise_curve")
    noise_scales = np.array([0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
    if args.debug:
        noise_scales = noise_scales[:3]
    is_wandb = False
    noise_history = None
else:
    noise_scales = np.array([np.nan])
    #noise_scales = np.array([args.noise_scale])
    is_wandb = True

noise_label = f"{args.noise}_{args.noise_var}"
calculations = {
    'losses' : [],
    'values' : [],
    'tokens' : []
}
for ii in itr_range:
    # Get Configuration
    config = configs.import_config(args.model_config, setting, args, ii)
    # Get tokenizer
    tokenizer = tokenizers.get_tokenizer(config, train_data, device, embed_size=args.d_model)

    # Get model
    if args.model == 'PatchTST':
        model = PatchTST(config, device, tokenizer=tokenizer)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(config, device, tokenizer=tokenizer)
        model.to(device)
    else:
        model = GPT4TS(config, device, tokenizer=tokenizer)
    
    criterion = metrics.get_loss(config, tokenizer) 

    model_folder, checkpoint_model_dir = utils.get_folder_names(
        args, setting, config, tokenizer, itr=ii
    )
    checkpoint_handler = utils.CheckpointHandler(
        checkpoint_model_dir,
        "state",
        args.keep_checkpoints,
        is_pretrain=args.pretrain_embeddings
    )
    prev_train_values = checkpoint_handler.load_latest(
        state=utils.CheckpointState(model, None, None),
        device=device,
        is_optimal=True,
        is_expected=True, 
    )
    """
    if args.noise_curve:
        filename = os.path.join(checkpoint_model_dir, "noise_results.json")
        if os.path.exists(filename):
            with open(filename, "r") as file:
                itr_noise_results = json.load(file)
        else:
            itr_noise_results = None
    """
    
    noise_curves_losses = []
    noise_curves_values = []
    noise_curves_tokens = []
    prev_results = None
    for ns_scale in noise_scales:
        print("Evaluating noise scale", ns_scale)
        if args.noise_curve:
            """
            prev_results = metrics.load_noise_metrics(
                itr_noise_results, ns_scale, has_loss=True
            )
            """
            prev_results = metrics.load_noise_metrics(
                checkpoint_model_dir, noise_label, ns_scale, require_loss=True
            )
            if prev_results is not None:
                loss_results, value_results, token_results = prev_results
        if not args.noise_curve or prev_results is None:
            if args.noise_curve:
                data_loader.dataset.add_noise_transform(
                    args.noise, args.noise_drift, args.noise_var, ns_scale
                )
            if args.plot_pred:
                utils.plot_samples(
                    model,
                    data_loader.dataset,
                    args.pred_len,
                    os.path.join("plots", setting, model_folder, "predictions"),
                    device,
                    seed=args.plot_seed
                )
                continue
            loss_results, value_results, token_results = utils.evaluate_dataset(
                model,
                tokenizer,
                data_loader,
                criterion,
                args,
                device,
                ii,
                as_dict=False,
                pretrain_embeddings=args.pretrain_embeddings,
                plot_dir=os.path.join("plots", setting, model_folder),
                debug=args.debug
            )

        noise_curves_losses.append(loss_results)
        noise_curves_values.append(value_results)
        noise_curves_tokens.append(token_results)

        if not args.noise_curve:
            metric_dict = utils.MetricCalculator.loss_metrics_to_dict(
                loss_results, value_results, token_results, values_and_tokens=True
            )
            utils.save_test_results(metric_dict, checkpoint_model_dir)

        #best_model_path = path + '/' + 'checkpoint.pth'
        #model.load_state_dict(torch.load(best_model_path))
        print("------------------------------------")
        #mse, mae = test(model, test_data, test_loader, args, device, ii)
        #mses.append(mse)
        #maes.append(mae)
    if args.plot_pred:
        sys.exit(0)
    calculations['losses'].append(np.array(noise_curves_losses))
    calculations['values'].append(np.array(noise_curves_values))
    calculations['tokens'].append(np.array(noise_curves_tokens))
    if args.noise_curve:
        """
        print("shape 1",
            calculations['losses'][-1].shape,
            calculations['values'][-1].shape,
            calculations['tokens'][-1].shape
        )
        print("shape 2",
            calculations['losses'][-1].transpose().shape,
            np.transpose(calculations['values'][-1], (1,2,0)).shape,
            np.transpose(calculations['tokens'][-1], (1,2,0)).shape,
        )
        """
        metric_dict = utils.MetricCalculator.loss_metrics_to_dict(
            calculations['losses'][-1].transpose(),
            np.transpose(calculations['values'][-1], (1,2,0)),
            np.transpose(calculations['tokens'][-1], (1,2,0)),
            values_and_tokens=True
        )
        #print("MET DICT NOISE", metric_dict)
        utils.save_noise_results(
            metric_dict, noise_scales, checkpoint_model_dir, noise_label
        )
    #for k, calc in calculations.items():
    #    print(k, np.array(calc).shape)
    #calculations.append(
    #    np.concatenate(
    #        [noise_curves_losses, noise_curves_values, noise_curves_tokens]
    #    )
    #)

#if args.itr is None:
total_means = {}
total_stds = {}
for k in calculations.keys():
    calculations[k] = np.array(calculations[k])
#print("CALCS", calculations)
for k, calc in calculations.items():
    #calculations = np.array(calculations)
    total_means[k] = np.mean(calc, axis=0)
    if calc.shape[0] == 1:
        total_stds[k] = np.nan*np.ones_like(total_means[k])
    else:
        total_stds[k] = np.std(calc, axis=0)

    if args.noise_curve:
        if len(total_means[k].shape) == 2:
            total_means[k] = np.transpose(total_means[k])
            total_stds[k] = np.transpose(total_stds[k])
        else:
            total_means[k] = np.transpose(total_means[k], (1,2,0))
            total_stds[k] = np.transpose(total_stds[k], (1,2,0))
    else:
        total_means[k] = total_means[k][0]
        total_stds[k] = total_stds[k][0]
metric_dict = utils.MetricCalculator.loss_metrics_to_dict(
    total_means['losses'],
    total_means['values'],
    total_means['tokens'],
    total_stds['losses'],
    total_stds['values'],
    total_stds['tokens'],
    values_and_tokens=True
)

#print("MET DICT 1", metric_dict)
checkpoint_dir = checkpoint_model_dir[:checkpoint_model_dir.rfind('/')]
if args.noise_curve:
    utils.save_noise_results(metric_dict, noise_scales, checkpoint_dir, noise_label)
else:
    utils.save_test_results(metric_dict, checkpoint_dir)

if not args.noise_curve:
    for lbl in ['value_value', 'token_value']:
        msemae_message = lbl\
            + ": MSE = {:.4f} +/- {:.4f} \t MAE = {:.4f} +/- {:.4f}".format(
                metric_dict[lbl+'_mse'],
                metric_dict[lbl+'_mse_std'],
                metric_dict[lbl+'_mae'],
                metric_dict[lbl+'_mae_std']
            )
        print(msemae_message)
    for lbl in ['value_token', 'token_token']:
        xentaccr_message = lbl\
            + ": cross entropy = {:.4f} +/- {:.4f} \t accuracy = {:.4f} +/- {:.4f}".format(
                metric_dict[lbl+'_cross_entropy'],
                metric_dict[lbl+'_cross_entropy_std'],
                metric_dict[lbl+'_accuracy'],
                metric_dict[lbl+'_accuracy_std']
            )
        print(xentaccr_message)

    """
    if calculations.shape[0] == 1:
        metric_dict = utils.metrics_to_dict(np.squeeze(calculations, 0))
        keys = list(metric_dict.keys())
        for k in keys:
            metric_dict[k+"_std"] = np.nan
    else:    
        #calculations = torch.stack(calculations)
        metric_dict = utils.metrics_to_dict(np.mean(calculations, dim=0), np.std(calculations, dim=0))
    """

# Record eval metrics
"""
if args.noise_curve:
    if noise_history is None:
        noise_history = {}
        noise_history["std_scale"] = [ns_scale]
        for k, v in metric_dict.items():
            noise_history[k] = [v]
    else:
        noise_history["std_scale"].append(ns_scale)
        for k, v in metric_dict.items():
            noise_history[k].append(v)
"""


if len(itr_range) > 1:
    # Saving results
    if is_wandb:
        run = utils.setup_wandb(config, checkpoint_model_dir, setting, args.debug)
        log_metrics = {
                f"test_{k}" : v for k, v in metric_dict.items()\
                    if 'std' not in k and\
                    ('loss' in k or 'mse' in k or 'mae' in k\
                     or 'cross_entropy' in k or 'accuracy' in k) 
            }
        log_metrics['test_loss'] = metric_dict['loss']
        log_metrics['epoch'] = prev_train_values["epochs"] 
        del log_metrics['loss']
        """
        calculations = {
            "test_loss" : metric_dict['loss'],
            "test_mse" : metric_dict['mse'],
            "test_mae" : metric_dict['mae'],
            "test_rmse" : metric_dict['rmse'],
            "test_mape" : metric_dict['mape'],
            "test_mspe" : metric_dict['mspe'],
            "test_smape" : metric_dict['smape'],
            "test_nd" : metric_dict['nd']
        }
        """
        wandb.log(log_metrics, step=prev_train_values["epochs"])

    """
        with open(os.path.join(checkpoint_model_dir, "noise_curve.json"), "w") as file:
            json.dump(noise_history, file)
    else:
        utils.save_test_results(metric_dict, checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "test_results.json"), "w") as file:
            json.dump(metric_dict, file)
        with open(os.path.join(checkpoint_dir, "test_results.txt"), "w") as file:
            #file.write("iter \t   MSE \t   MAE\n")
            file.write(f"loss \t{metric_dict['loss']:.4f} +/- {metric_dict['loss_std']:.4f}\n")
            for lbl in utils.get_labels():
                file.write(f"{lbl} \t{metric_dict[lbl]:.4f} +/- {metric_dict[lbl+'_std']:.4f}\n")
            file.write("\n")
    """