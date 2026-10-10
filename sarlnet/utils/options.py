import argparse
import os
import random

import numpy as np
import torch
import yaml


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option yml.")
    args = parser.parse_args()
    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    opt["opt_path"] = args.opt
    return opt


def mkdir_and_rename(path):
    os.makedirs(path, exist_ok=True)


def set_random_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

