import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_samples(model, dataset, pred_len, output_dir, device, n_samples=10, seed=123):
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices = indices[:n_samples]
    model.eval()
    for idx in indices:
        #token_x, token_y, x, y, x_mark, y_mark = dataset[idx]
        x, y, x_mark, y_mark = dataset[idx]
        pred = model(
            torch.tensor(x).unsqueeze(0).float().to(device),
            #torch.tensor(token_x).unsqueeze(0).to(torch.int64).to(device),
            #max_new_tokens=y.shape[-1],
            ##return_last=len(y)
        )
        #pred = tokenizer.invert(torch.argmax(logits, dim=-1))
        pred = pred.value_values.detach().cpu().numpy()
        pred = pred[0,-1*pred_len:,0]
        y = y[-1*pred_len:,0]
        y_mark = y_mark[-1*pred_len:,0]
        x = x[-100:,0]
        x_mark = x_mark[:,0]
        time = np.concatenate([x_mark, y_mark])
        truth = np.concatenate([x, y])
        pred_ = np.concatenate([np.ones_like(x)*np.nan, pred])
        fig, ax = plt.subplots()
        ax.plot(truth[:], '-k', label="Truth")
        #ax.plot(y, '-k', label="Truth")
        ax.plot(pred_, '-b', label="Prediction")
        y_max = np.amax([np.amax(truth), np.amax(pred)])
        y_max = np.amax([y_max*0.9, y_max*1.1])
        y_min = np.amin([np.amin(truth), np.amin(pred)])
        y_min = np.amin([y_min*0.9, y_min*1.1])
        ax.plot([len(x), len(x)], [y_min, y_max], ':k')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, len(truth))
        ax.set_xlabel("Time")
        ax.legend()

        fig.savefig(os.path.join(
            output_dir, f"pred-{len(y)}_sample-{idx}.png")
        )
