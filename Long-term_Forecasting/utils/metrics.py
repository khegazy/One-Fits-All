import os
import json
import einops
import torch
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple


class LossResults(NamedTuple):
    """a docstring"""
    total_loss : torch.tensor
    value_value : torch.tensor
    value_token : torch.tensor
    token_value : torch.tensor
    token_token : torch.tensor

class Loss:
    def __init__(
            self,
            value_loss_config,
            token_loss_config,
            tokenizer=None,
            pretrain_embeddings=False
        ):
        self.value_config = value_loss_config
        self.token_config = token_loss_config
        self.tokenizer = tokenizer
        self.pretrain_embeddings = pretrain_embeddings
        
        self.value_criterion = self.get_criterion(self.value_config)
        self.token_criterion = self.get_criterion(self.token_config)
        
    def __call__(self, pred, truth):
        value_value = 0
        value_token = 0
        token_value = 0
        token_token = 0

        use_tkn_y_value = self.value_criterion is not None\
            and self.value_config['token_pred_scale'] is not None
        use_tkn_y_token = self.token_criterion is not None\
            and self.token_config['token_pred_scale'] is not None 
        if self.tokenizer is not None and (use_tkn_y_value or use_tkn_y_token):
            truth_tokens = self.tokenizer(
                einops.rearrange(truth, 'b l m -> b m l')
            )
            truth_tokens = einops.rearrange(truth_tokens, 'b m l -> b l m')
        
        if self.value_criterion is not None:
            if self.value_config['value_pred_scale'] is not None:
                value_value = self.value_config['value_pred_scale']\
                    *self.value_criterion(pred.value_values, truth)
            #print(self.value_config)
            #print(self.value_config['token_pred_scale'])
            if self.value_config['token_pred_scale'] is not None:
                value_token = self.value_config['token_pred_scale']\
                    *self.token_criterion(self.tokenizer(pred.value_values, return_logits=True), truth_tokens)
        if self.token_criterion is not None:
            if self.token_config['value_pred_scale'] is not None:
                token_value = self.token_config['value_pred_scale']\
                    *self.value_criterion(pred.token_values, truth)
            if self.token_config['token_pred_scale'] is not None:
                token_token = self.token_config['token_pred_scale']\
                    *self.token_criterion(pred.token_logits, truth_tokens)
         
            """
            loss += self.token_criterion(
                pred.token_values,
                true,
                pred.token_logits,
                self.tokenizer(true),
                'token'
            )
            """
        total_loss = value_value + value_token + token_value + token_token
        results = LossResults(
            total_loss,
            value_value,
            value_token,
            token_value,
            token_token
        )

        return total_loss, results

    def get_criterion(self, config):
        if config is None:
            return None
        
        if config['type'] == 'mse':
            return MSE
        elif config['type'] == 'mse_cross_entropy':
            #self.params[f'{prefix}_mse_weight'] = params['mse_weight']
            #self.params[f'{prefix}_cross_entropy_weight'] = params['cross_entropy_weight']
            #self.mse = torch.nn.MSELoss
            return self.MSE_cross_entropy
        elif config['type'] == 'cross_entropy':
            return cross_entropy
        elif config['type'] == 'smape':
            return SMAPE
        
        raise ValueError(f"Cannot handle loss type {config['type']}")

    def MSE_cross_entropy(self, pred, true, pred_logits, true_tokens, prefix):
        return self.params[f'{prefix}_mse_weight']*MSE(pred, true)\
            + self.params[f'{prefix}_xentropy_weight']\
            * F.cross_entropy(pred_logits, true_tokens)

def get_loss(config, tokenizer):
    if config.pretrain_embeddings:
        value_loss_config = config.embedding_value_loss_config
        token_loss_config = config.embedding_token_loss_config
    else:
        value_loss_config = config.value_loss_config
        token_loss_config = config.token_loss_config
    
    return Loss(
        value_loss_config,
        token_loss_config,
        tokenizer=tokenizer,
        pretrain_embeddings=config.pretrain_embeddings
    )

def RSE(pred, true, *args):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true, *args):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true, *args):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true, *args):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true, *args):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true, *args):
    return torch.mean(torch.abs(100 * (pred - true) / (true +1e-8)))


def MSPE(pred, true, *args):
    return torch.mean(torch.square((pred - true) / (true + 1e-8)))

def SMAPE(pred, true, *args):
    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    # return torch.mean(200 * torch.abs(pred - true) / (pred + true + 1e-8))

def ND(pred, true, *args):
    return torch.mean(torch.abs(true - pred)) / torch.mean(torch.abs(true))


value_labels = ["mae", "mse", "rmse", "mape", "mspe", "smape", "nd"]
token_labels = ["cross_entropy", "accuracy"]
class MetricCalculator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.n_values = len(value_labels)
        self.n_tokens = len(token_labels)
    
    def __call__(self, pred, truth):
        if self.tokenizer is not None:
            truth_tokens = self.tokenizer(
                einops.rearrange(truth, 'b l m -> b m l')
            )
            truth_tokens = einops.rearrange(truth_tokens, 'b m l -> b l m')
            truth_tokens = truth_tokens.to(truth.device)

        if pred.value_values is not None:
            value_values = MetricCalculator.value_metrics(
                pred.value_values, truth
            )
            if self.tokenizer is not None:
                logits = self.tokenizer(
                    einops.rearrange(pred.value_values, 'b l m -> b m l'),
                    return_logits=True
                ).to(truth.device)
                logits = einops.rearrange(logits, 'b m l c -> b l m c')
                value_tokens = MetricCalculator.token_metrics(logits, truth_tokens)
            else:
                value_tokens = torch.ones(self.n_tokens)*torch.nan 
        else:
            value_values = torch.ones(self.n_values)*torch.nan
            value_tokens = torch.ones(self.n_tokens)*torch.nan 
        
        if pred.token_values is not None:
            token_values = MetricCalculator.value_metric(
                pred.token_values, truth
            )
        else:
            token_values = torch.ones(self.n_values)*torch.nan
        
        if pred.token_logits is not None:
            token_tokens = MetricCalculator.token_metrics(
                pred.token_logits, truth_tokens
            )
        else:
            token_tokens = torch.ones(self.n_tokens)*torch.nan

        value_results = torch.stack((value_values, token_values))
        token_results = torch.stack((value_tokens, token_tokens))
        return value_results, token_results

    @staticmethod
    def value_metrics(pred, truth, *args):
        mae = MAE(pred, truth)
        mse = MSE(pred, truth)
        rmse = RMSE(pred, truth)
        mape = MAPE(pred, truth)
        mspe = MSPE(pred, truth)
        smape = SMAPE(pred, truth)
        nd = ND(pred, truth)

        return torch.tensor([mae, mse, rmse, mape, mspe, smape, nd])
    
    @staticmethod
    def token_metrics(pred, truth, *args):
        xentropy = cross_entropy(pred, truth)
        accr = accuracy(pred, truth)

        return torch.tensor([xentropy, accr])

    @staticmethod
    def loss_metrics_to_dict(
        losses,
        value_metrics,
        token_metrics,
        losses_std=None,
        value_std=None,
        token_std=None,
    ):
        output_value = MetricCalculator.metrics_to_dict(
            value_metrics, value_std, type='value'
        )
        output_token = MetricCalculator.metrics_to_dict(
            token_metrics, token_std, type='token'
        )
        output = {**output_value, **output_token}

        loss_labels = [
            'loss',
            'value_value_loss',
            'value_token_loss',
            'token_value_loss',
            'token_token_loss'
        ]
        if losses_std is None:
            losses_std = np.ones_like(losses)*np.nan
        for i, (val, std) in enumerate(zip(losses, losses_std)):
            output[loss_labels[i]] = val
            output[loss_labels[i] + '_std'] = std

        return output

    @staticmethod
    def metrics_to_dict(metrics, metric_std=None, type='values', prefix=''):
        if metric_std is None:
            metric_std = np.ones_like(metrics)*np.nan
        
        if len(metrics.shape) == 2:
            if type[-1] == 's':
                type = type[:-1]
            output = {}
            for i, (mtrs, mtrs_std) in enumerate(zip(metrics, metric_std)):
                prefix = 'value' if i == 0 else 'token'
                prefix += '_' + type + "_"
                output = {**output, **MetricCalculator.metrics_to_dict(
                    mtrs, mtrs_std, type=type, prefix=prefix
                )}
            return output

        if type == 'values' or type == 'value':
            labels = value_labels
        elif type == 'tokens' or type == 'token':
            labels = token_labels
        else:
            raise ValueError(f"Cannot handle type '{type}'")
        
        if len(labels) == len(metrics)-1:
            labels = ['loss'] + labels
        if len(labels) == len(metrics):
            output = {prefix + labels[i] : metrics[i] for i in range(len(labels))}
            for i in range(len(labels)):
                output[prefix + labels[i]+"_std"] = metric_std[i]
        else:
            raise ValueError("Input metrics are not the right length")

        return output

"""
def get_labels():
    return [
        "mae", "mse", "rmse", "mape", "mspe", "smape", "nd"
    ]
"""

def cross_entropy(logits, tokens_y):
    return F.cross_entropy(
        einops.rearrange(logits, 'b l m c -> b c l m'), tokens_y
    )

def accuracy(input, tokens_y):
    # If input is logits convert to tokens
    if len(input.shape) > len(tokens_y.shape):
        input = torch.argmax(input, -1)
    return torch.mean((input == tokens_y).to(torch.float32))

def load_noise_metrics(results_dir, label, noise, require_loss=True):
    filename = os.path.join(results_dir, f"noise_{label}_results.json")
    if os.path.exists(filename):
        with open(filename, "r") as file:
            noise_dict = json.load(file)
    else:
        return None
    
    labels = get_labels()
    if require_loss:
        labels.insert(0, 'loss')
    
    if noise not in noise_dict['std_scales']:
        return None
    idx = noise_dict['std_scales'].index(noise)
    results = []
    for lbl in labels:
        if lbl not in noise_dict.keys():
            return None
        results.append(noise_dict[lbl][idx])
    
    return results


"""
def MAE_tokens(logits, data_y, tokenizer, tokens_y=None):
    tokens_pred = torch.argmax(logits, dim=-1)
    differences = torch.abs(tokenizer.invert(tokens_pred) - data_y)
    return torch.mean(differences.to(torch.float32))

def MSE_tokens(logits, data_y, tokenizer):
    tokens_pred = torch.argmax(logits, dim=-1)
    return F.mse_loss(tokenizer.invert(tokens_pred), data_y)

def accuracies(logits, tokens_y, epsilons=None):
    if epsilons is not None:
        delta = torch.abs(torch.argmax(logits, dim=-1) - tokens_y).flatten()
        bools = delta.unsqueeze(dim=-1) <= epsilons
        return torch.mean(bools.to(torch.float32), dim=0)
    else:
        return torch.mean(
            (torch.argmax(logits, dim=-1) == tokens_y).to(torch.float32)
        )

class xEntropyMSEloss():
    def __init__(self, tokenizer, x_entropy_scale=1, x_entropy_ratio=1):
        self.tokenizer = tokenizer
        self.x_entropy_ratio = x_entropy_ratio
        self.x_entropy_scale = x_entropy_scale

    def __call__(self, logits, data_y=None, tokens_y=None):
        x_entropy = cross_entropy(logits, tokens_y=tokens_y)
        mse = MSE(logits, data_y, self.tokenizer)
        ratio = self.x_entropy_ratio*mse
        return self.x_entropy_scale*ratio*x_entropy + mse

def get_loss_fxn(loss_name, tokenizer=None, x_entropy_ratio=1, x_entropy_scale=1):
    loss_name = loss_name.lower()
    losses = {
        #"mse": torch.nn.MSELoss,
        "cross_entropy" : cross_entropy,
        "cross_entropy_mse" : xEntropyMSEloss(
            tokenizer,
            x_entropy_ratio=x_entropy_ratio,
            x_entropy_scale=x_entropy_scale
        )
    }
    assert(loss_name in losses) 
    return losses[loss_name]

class metricCalculator():
    def __init__(self, tokenizer, loss_fxn=None, acc_epsilons=None):
        self.tokenizer = tokenizer
        self.acc_epsilons = acc_epsilons
        self.loss_fxn = loss_fxn
    
    def __call__(self, logits, data_y=None, tokens_y=None, as_dict=False):
        x_entropy = cross_entropy(logits, tokens_y=tokens_y)
        mse = MSE(logits, data_y=data_y, tokenizer=self.tokenizer)
        mae = MAE(logits, data_y=data_y, tokens_y=tokens_y, tokenizer=self.tokenizer)
        results = [mse, mae, x_entropy]
        if self.loss_fxn is not None:
            results.insert(
                0, self.loss_fxn(logits, data_y=data_y, tokens_y=tokens_y)
            )
        
        accs = accuracies(logits, tokens_y, self.acc_epsilons)
        if torch.is_tensor(accs):
            results = torch.concat(
                [torch.tensor(results).to(accs.device), accs]
            )
        else:
            results.append(accs)
            results = torch.tensor(results)
        
        if as_dict:
            names = self.get_names()
            return {name : val for name, val in zip(names, results)}            
        else:
            return results

    def get_names(self):
        names = ["MSE", "MAE", "cross_entropy"]
        if self.acc_epsilons is None:
            names.append("accuracy")
        else:
            for eps in self.acc_epsilons:
                names.append(f"accuracy_{eps}")
        
        if self.loss_fxn is not None:
            names.insert(0, "loss")
        
        return names

class emaMetricCalculator():
    def __init__(self, scale):
        self.scale = scale
        self.ema_metrics = {}
    
    def update(self, metrics):
        for key, val in metrics.items():
            if key not in self.ema_metrics:
                self.ema_metrics[key] = val
            else:
                self.ema_metrics[key] =\
                    (1 - self.scale)*self.ema_metrics[key] + self.scale*val
    
    def get_metrics(self, prefix=False):
        if prefix:
            return {prefix+key : val for key, val in self.ema_metrics.items()}
        else:
            return self.ema_metrics
    
    def load_metrics(self, metrics):
        self.ema_metrics = metrics
"""