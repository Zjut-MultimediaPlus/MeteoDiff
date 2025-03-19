import concurrent.futures
from tc_diffuser import TCDDIFFUSER
import argparse
import yaml
from easydict import EasyDict
import numpy as np
import torch
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--train_dataset', default='WP') #['ALL','EP', 'NA', 'NI', 'SI', 'SP', 'WP']
    parser.add_argument('--eval_dataset', default='WP')  # ['ALL','EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    return parser.parse_args()

def inference(test):
    # parse arguments and load config
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    config["exp_name"] = 'add-ERA5-16-z-uv2-vocen'

    config["train_dataset"] = args.train_dataset
    config["eval_dataset"] = args.eval_dataset
    config = EasyDict(config)
    i = test
    config.eval_at = i
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    agent = TCDDIFFUSER(config)

    sampling = "ddim"
    step = 5
    config.eval_at = i
    agent.eval(sampling, 100 // step, i)


# if __name__ == '__main__':
cuda_idx = 0
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

i=290
while i>0:
    inference(i)
    i=i-5
    if i<120:
        i=i-5