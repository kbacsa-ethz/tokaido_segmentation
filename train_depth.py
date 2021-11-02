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
    save_path = os.path.join(cfg.data_path, 'runs', dt_string)
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

    classes = ["nonbridge", "slab", "beam", "column", "nonstructural", "rail", "sleeper"]

    model = smp.create_model(
        arch=cfg.arch,
        encoder_name=cfg.backbone,
        encoder_weights=cfg.pretrained,
        classes=len(classes)+1,
        activation=cfg.activation
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.backbone, cfg.pretrained)

    if not cfg.lmdb:
        train_dataset = Dataset(
            x_dir,
            y_dir,
            z_dir,
            train_files,
            train_masks,
            train_depths,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )

        valid_dataset = Dataset(
            x_dir,
            y_dir,
            z_dir,
            valid_files,
            valid_masks,
            valid_depths,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )

    else:
        train_dataset = TokaidoLMDBDepth(
            db_path=os.path.join(cfg.data_path, 'tokaido_depth_lmdb'),
            keys=train_keys,
            classes=classes,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn)
        )

        valid_dataset = TokaidoLMDBDepth(
            db_path=os.path.join(cfg.data_path, 'tokaido_depth_lmdb'),
            keys=val_keys,
            classes=classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn)
        )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True if cfg.device == 'cuda' else False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True if cfg.device == 'cuda' else False)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    #loss = smp.utils.losses.DiceLoss()
    #loss = smp.utils.losses.FocalLoss()

    # penalty weight per class
    gamma = 2.
    alpha = [0.9977271, 0.59993025, 0.85252386, 0.8816177, 0.84685007, 0.9727033, 0.99270273, 1.]
    loss = smp.utils.base.SumOfLosses(
        smp.utils.losses.DiceLoss(),
        smp.utils.losses.FocalLoss(gamma=gamma, alpha=alpha)
    )

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=cfg.learning_rate),
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=cfg.device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpochMonteCarlo(
        model,
        loss=loss,
        metrics=metrics,
        monte_carlo_it=cfg.monte_carlo,
        device=cfg.device,
        verbose=True,
    )

    # plot parameters
    t = 1
    cmap = {
        0: [1.0, 0.0, 0.0, t],
        1: [0.7, 0.3, 0.1, t],
        2: [1.0, 0.6, 0.2, t],
        3: [0.0, 1.0, 0.0, t],
        4: [0.1, 0.7, 0.3, t],
        5: [0.2, 1.0, 0.6, t],
        6: [0.0, 0.0, 1.0, t],
    }

    max_score = 0
    with experiment.train():
        for epoch in range(0, cfg.n_epochs):
            print('\nEpoch: {}'.format(epoch))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            experiment.log_metric("iou_score", valid_logs['iou_score'], step=epoch)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, os.path.join(save_path, 'best_model.pth'))
                print('Model saved!')

            if epoch == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='/home/kb/Documents/data/Tokaido_dataset')

    # Model parameters
    parser.add_argument('--arch', type=str, default='fpn')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--monte_carlo', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lmdb', action='store_true')
    parser.add_argument('--comet', action='store_true')

    # Machine parameters
    args = parser.parse_args()
    train(args)
