import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from . import hashes
from . import metrics

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def evaluate_dataset(
        model,
        tokenizer,
        data_loader,
        criterion,
        args,
        device,
        itr,
        as_dict=True,
        is_training=False,
        pretrain_embeddings=False):
    
    if is_training:
        if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
            model.eval()
        else:
            model.in_layer.eval()
            for layer in model.out_layers:
                layer.eval()
    total_samples = 0
    total_loss = 0
    total_losses = 0
    total_value_metrics = 0
    total_token_metrics = 0
    value_value_loss = 0
    value_token_loss = 0
    token_value_loss = 0
    token_token_loss = 0
    metric_calc = metrics.MetricCalculator(tokenizer)
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if i > 3:
                break
            if np.sum(np.isnan(batch_x.numpy())) > 0:
                raise ValueError("GOT A NAN VALUE AT BATCH", i)
            n_samples = batch_x.shape[0]
            """
            if i > 9:
                break
            fig, ax = plt.subplots()
            idxs = np.arange(n_samples)
            np.random.shuffle(idxs)
            ax.plot(orig[idxs[0]], '-k', linewidth=1.5)
            ax.plot(batch_x[idxs[0]], ':b', alpha=0.75)
            fig.savefig(os.path.join(
                "plots", "noise", f"{args.noise}_{args.noise_var}",
                f"noise_{data_loader.dataset.noise_transform.std_scale}_{i*len(batch_x)+idxs[0]}.png"
            ))
            """
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if pretrain_embeddings:
                batch_y = batch_x

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            predictions = model(batch_x)
            #print("PREDICTION INFO", torch.mean(predictions.value_values).item(), torch.std(predictions.value_values).item())
            
            # encoder - decoder
            #values = predictions.values[:, -args.pred_len:, :]
            #batch_y = batch_y[:, -args.pred_len:, :].to(device)

            loss, loss_results = criterion(predictions, batch_y)
            losses = torch.tensor([
                loss.detach(),
                loss_results.value_value,
                loss_results.value_token,
                loss_results.token_value,
                loss_results.token_token
            ])
            total_losses += losses*n_samples
           
            batch_value_metrics, batch_token_metrics = metric_calc(predictions, batch_y)
            total_value_metrics = total_value_metrics + batch_value_metrics*n_samples
            total_token_metrics = total_token_metrics + batch_token_metrics*n_samples
            
            total_samples = total_samples + n_samples
 
    if is_training:
        if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
            model.train()
        else:
            model.in_layer.train()
            for layer in model.out_layers:
                layer.train()
           
    total_losses = total_losses.cpu().numpy()/total_samples
    
    total_value_metrics = total_value_metrics.detach().cpu()/total_samples
    total_value_metrics = total_value_metrics.numpy()
    total_token_metrics = total_token_metrics.detach().cpu()/total_samples
    total_token_metrics = total_token_metrics.numpy()
    if as_dict:
        return metric_calc.loss_metrics_to_dict(
            total_losses, total_value_metrics, total_token_metrics
        )
        total_metrics = metric_calc.metrics_to_dict(total_value_metrics, type='values')
        total_metrics = total_metrics.update(
            metric_calc.metrics_to_dict(total_token_metrics, type='tokens')
        )
        total_metrics['loss'] = total_loss
        total_metrics['value_value'] = value_value_loss
        total_metrics['value_token'] = value_token_loss
        total_metrics['token_value'] = token_value_loss
        total_metrics['token_token'] = token_token_loss
        return total_metrics
    else:
        #total_losses = np.array(
        #    [total_loss, value_value_loss, value_token_loss, token_value_loss, token_token_loss]
        #)
        return total_losses, np.array(total_value_metrics), np.array(total_token_metrics)


    if not pretrain_embeddings:
        total_metrics = total_metrics.detach().cpu()/total_samples
        total_metrics = total_metrics.tolist()
        if as_dict:
            total_metrics = metrics.metrics_to_dict(total_metrics)
            total_metrics["loss"] = total_loss
        else:
            total_metrics = np.concatenate([np.array([total_loss]), total_metrics])
    elif as_dict:
        total_metrics = {"loss" : total_loss}
    
    if as_dict:
        total_metrics['value_value'] = value_value_loss
        total_metrics['value_token'] = value_token_loss
        total_metrics['token_value'] = token_value_loss
        total_metrics['token_token'] = token_token_loss
    else:
        total_metrics = np.array(
            [total_loss, value_value_loss, value_token_loss, token_value_loss, token_token_loss]
        )
    return total_metrics


def vali(model, vali_data, vali_loader, criterion, args, device, itr):
    total_loss = []
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.eval()
    else:
        model.in_layer.eval()
        model.out_layer.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, itr)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.train()
    else:
        model.in_layer.train()
        model.out_layer.train()
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    # mases = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            
            outputs = model(batch_x[:, -args.seq_len:, :], itr)
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def save_test_results(value_dict, checkpoint_dir, filename='test_results', print_results=True):
    with open(os.path.join(checkpoint_dir, f"{filename}.json"), "w") as file:
        json.dump(value_dict, file, sort_keys=True)
    if print_results:
        with open(os.path.join(checkpoint_dir, f"{filename}.txt"), "w") as file:
            #file.write("iter \t   MSE \t   MAE\n")
            file.write(f"loss \t{value_dict['loss']:.4f} +/- {value_dict['loss_std']:.4f}\n")
            for lbl in metrics.get_labels():
                file.write(f"{lbl} \t{value_dict[lbl]:.4f} +/- {value_dict[lbl+'_std']:.4f}\n")
            file.write("\n")

def save_noise_results(value_dict, noise_scales, checkpoint_dir, label, merge_prev_results=True):
    value_dict['std_scales'] = list(noise_scales)
    for k in value_dict.keys():
        print(k, value_dict[k])
        value_dict[k] = list(value_dict[k])
    
    if merge_prev_results:
        filename = os.path.join(checkpoint_dir, f"noise_{label}_results.json")
        if os.path.exists(filename):
            with open(filename, "r") as file:
                prev_results = json.load(file)
            if prev_results is not None:
                for idx, ns_scale in enumerate(prev_results['std_scales']):
                    if ns_scale in value_dict['std_scales']:
                        continue
                    for key, value in prev_results.items():
                        value_dict[key].append(value[idx])
    save_test_results(value_dict, checkpoint_dir, f"noise_{label}_results", False)


def get_folder_names(args, setting, config, tokenizer, itr=None, has_itr=True):
    tk_label = "None" if tokenizer is None else tokenizer.get_hash()
    ls_val_label, ls_tkn_label = "", ""
    if config.predict_values:
        ls_val_label = f"{config.value_loss_config['type'].lower()}{hashes.get_hash(json.dumps(config.value_loss_config, sort_keys=True))}"
    if config.predict_tokens:
        ls_tkn_label = f"{config.token_loss_config['type'].lower()}{hashes.get_hash(json.dumps(config.token_loss_config, sort_keys=True))}"
        if config.predict_values:
            ls_tkn_label = "_" + ls_tkn_label
    model_folder = f"tk-{tk_label}_ls-{ls_val_label}{ls_tkn_label}"\
        + f"_rc-{config.recursive}_predTkns-{config.predict_tokens}"\
        + f"_preTrn-{config.from_pretrain_model}"
    checkpoint_dir = os.path.join(args.checkpoints, setting, model_folder)
    if itr is None:
        itr = args.itr
    checkpoint_itr_dir = os.path.join(checkpoint_dir, f"itr_{itr}")

    if has_itr:
        return model_folder, checkpoint_itr_dir
    else:
        return model_folder, checkpoint_itr_dir, checkpoint_dir