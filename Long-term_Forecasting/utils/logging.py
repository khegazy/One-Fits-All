import wandb
import dataclasses

def setup_wandb(config, wandb_dir, run_name, debug, itr=None):
    # wandb logging
    wandb_project = "time_series_OFA"
    wandb_dir = wandb_dir
    wandb_run = run_name
    wandb_config = dataclasses.asdict(config)
    tags = [config.loss_config['type']]
    if itr is not None:
        tags.append(f"itr_{itr}")
    if config.tokenizer is not None:
        tags.append(config.tokenizer.lower())
    if config.predict_tokens:
        tags.append("predict_tokens")
    if config.recursive:
        tags.append("recursive")
    if config.pretrain:
        tags.append("pretrain")
    else:
        tags.append("randInit")
    
    if debug:
        wandb_run = "debug_" + wandb_run
        tags.append("debug")
    
    return wandb.init(
        config=wandb_config,
        mode="disabled" if debug else "online",
        project=wandb_project,
        name=wandb_run,
        tags=tags,
        dir=wandb_dir,
        resume=True,
    )