from tc_diffuser import TCDDIFFUSER
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb
import torch
import random
import dill


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--train_dataset', default='WP') #['ALL','EP', 'NA', 'NI', 'SI', 'SP', 'WP']
    parser.add_argument('--eval_dataset', default='WP')  # ['ALL','EP', 'NA', 'NI', 'SI', 'SP', 'WP']
    # d
    return parser.parse_args()


def main():
    cuda_idx = 0
    device = torch.device('cuda:' + str(cuda_idx))
    torch.cuda.set_device(device)

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # parse arguments and load config
    args = parse_args()
    with open(args.config,encoding='utf-8') as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = 'MeteoDiff_ori' #

    config["train_dataset"] = args.train_dataset
    config["eval_dataset"] = args.eval_dataset
    #pdb.set_trace()
    config = EasyDict(config)


    if config["eval_mode"]:
        test = [320,310,300,290,280,270, 265, 260, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170,
                165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65,
                60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
        test = [210]


        for i in test:
            config.eval_at = i
            seed = 123
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)  # Numpy module.
            random.seed(seed)  # Python random module.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            agent = TCDDIFFUSER(config)

            sampling = "ddim"
            step = 5
            config.eval_at = i
            agent.eval(sampling, 100//step, i)
    else:
        agent = TCDDIFFUSER(config)

        sampling = "ddim"
        step = 5
        agent.train()



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
