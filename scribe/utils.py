import sys
from pathlib import Path

import torch
#from adam_mini import Adam_mini

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RUNS_DIR

def model_params_setup(model, args, model_config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    """ optimizer = Adam_mini(named_parameters = model.named_parameters(), 
                          lr=args.lr, 
                          eps=1e-8,
                          dim = model_config.n_embd,
                          n_heads = model_config.n_head)
    optimizer.wv_names = {}
    print("testing adam mini", model_config.n_embd, model_config.n_head) """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    save_args(args)
    return optimizer, scheduler

def save_args(args):
    #function to save the arguments incase we want to rerun the model
    with open(RUNS_DIR / args.run / "model" / f"{args.run}_args.txt", "w") as f:
        f.write(str(args))
    return