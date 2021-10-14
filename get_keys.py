from comet_ml import Experiment

import os
import argparse
from datetime import datetime
from pathlib import Path
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
from tokaido_depth import Dataset
from lmdb_data_depth import TokaidoLMDBDepth
from torch.utils.data import DataLoader
from data_aug import get_training_augmentation, get_validation_augmentation, get_preprocessing


def train(cfg):

    hyperparams = vars(cfg)
    experiment = Experiment(project_name="tokaido_segmentation", api_key="Bm8mJ7xbMDa77te70th8PNcT8", disabled=not cfg.comet)
    experiment.log_parameters(hyperparams)

    # create experiment directory and save config
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_path = os.path.join(cfg.root_path, 'runs', dt_string)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    x_dir = os.path.join(cfg.data_path, 'img_syn_raw', 'train')
    y_dir = os.path.join(cfg.data_path, 'synthetic', 'train', 'labcmp')
    z_dir = os.path.join(cfg.data_path, 'synthetic', 'train', 'depth')

    files = os.listdir(x_dir)
    cases = np.zeros(len(files), dtype=int)
    for idx, filename in enumerate(files):
        case = int(re.sub("[^0-9]", "", filename.split('_')[1]))
        cases[idx] = case

    sequences = np.unique(cases)
    np.random.shuffle(sequences)
    training, validation = sequences[:int(.8 * len(sequences))], sequences[int(.8 * len(sequences)):]

    train_files, train_masks, train_depths, train_keys, valid_files, valid_masks, valid_depths, val_keys, = [], [], [], [], [], [], [], []

    for img_file in os.listdir(x_dir):
        mask_name = img_file.replace('_Scene.png', '.bmp')
        depth_name = img_file.replace('_Scene.png', '.png')
        if os.path.isfile(os.path.join(y_dir, mask_name)) and os.path.isfile(os.path.join(z_dir, depth_name)):
            case = int(re.sub("[^0-9]", "", img_file.split('_')[1]))
            frame = int(re.sub("[^0-9]", "", img_file.split('_')[2]))
            key = key = ((1 << case) << 16) + (1 << frame)

            if case in training:
                train_files.append(img_file)
                train_masks.append(mask_name)
                train_depths.append(depth_name)
                train_keys.append(key)
            if case in validation:
                valid_files.append(img_file)
                valid_masks.append(mask_name)
                valid_depths.append(depth_name)
                val_keys.append(key)

    print(train_keys)

    import pickle

    with open('train_keys', 'wb') as fp:
        pickle.dump(train_keys, fp)
    with open('val_keys', 'wb') as fp:
        pickle.dump(val_keys, fp)

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='/home/kb/Documents/data/Tokaido_dataset')

    # Model parameters
    parser.add_argument('--arch', type=str, default='pan')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2')
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--device', type=str, default='cpu')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lmdb', action='store_true')
    parser.add_argument('--comet', action='store_true')

    # Machine parameters
    args = parser.parse_args()
    train(args)
