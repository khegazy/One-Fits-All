import torch
import torch.nn.functional as F
import numpy as np


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return torch.mean(torch.abs(100 * (pred - true) / (true +1e-8)))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / (true + 1e-8)))

def SMAPE(pred, true):
    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    # return torch.mean(200 * torch.abs(pred - true) / (pred + true + 1e-8))

def ND(pred, true):
    return torch.mean(torch.abs(true - pred)) / torch.mean(torch.abs(true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)

    return torch.tensor([mae, mse, rmse, mape, mspe, smape, nd])

def get_labels():
    return [
        "mae", "mse", "rmse", "mape", "mspe", "smape", "nd"
    ]

def metrics_to_dict(metrics, metric_std=None):
    labels = get_labels()
    if len(labels) == len(metrics):
        output = {labels[i] : metrics[i] for i in range(len(labels))}
        if metric_std is not None:
            for i in range(len(labels)):
                output[labels[i]+"_std"] = metric_std[i]
    elif len(labels) == len(metrics)-1: 
        output = {labels[i] : metrics[i+1] for i in range(len(labels))}
        output["loss"] = metrics[0]
        if metric_std is not None:
            for i in range(len(labels)):
                output[labels[i]+"_std"] = metric_std[i+1]
            output["loss_std"] = metric_std[0]
    else:
        raise ValueError("Input metrics are not the right length")

    return output

def cross_entropy(logits, data_y=None, tokens_y=None):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens_y.view(-1))


class Loss:
    def __init__(self, names, weights):
        self.names = names
        self.weights = weights

        self.loss_dict = {
            "mse" : MSE
        }

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