import os
import argparse
import re
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tokaido_data import Dataset
from torch.utils.data import DataLoader
from data_aug import get_training_augmentation, get_validation_augmentation, get_preprocessing


def train(cfg):

    x_dir = os.path.join(cfg.data_path, 'img_syn_raw', 'train')
    y_dir = os.path.join(cfg.data_path, 'synthetic', 'train', 'labcmp')

    files = os.listdir(x_dir)
    cases = np.zeros(len(files), dtype=int)
    for idx, filename in enumerate(files):
        case = int(re.sub("[^0-9]", "", filename.split('_')[1]))
        cases[idx] = case

    sequences = np.unique(cases)
    np.random.shuffle(sequences)
    training, validation = sequences[:int(.8 * len(sequences))], sequences[int(.8 * len(sequences)):]

    train_files, train_masks, valid_files, valid_masks = [], [], [], []

    for img_file in os.listdir(x_dir):
        mask_name = img_file.replace('_Scene.png', '.bmp')
        if os.path.isfile(os.path.join(y_dir, mask_name)):
            case = int(re.sub("[^0-9]", "", img_file.split('_')[1]))
            if case in training:
                train_files.append(img_file)
                train_masks.append(mask_name)
            if case in validation:
                valid_files.append(img_file)
                valid_masks.append(mask_name)

    classes = ["slab", "beam", "column", "nonstructural components", "rail", "sleeper"]

    model = smp.Unet(
        encoder_name=cfg.backbone,
        encoder_weights=cfg.pretrained,
        classes=len(classes),
        activation=cfg.activation
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.backbone, cfg.pretrained)

    train_dataset = Dataset(
        x_dir,
        y_dir,
        train_files,
        train_masks,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    valid_dataset = Dataset(
        x_dir,
        y_dir,
        valid_files,
        valid_masks,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp.utils.losses.DiceLoss()
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

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=cfg.device,
        verbose=True,
    )

    # train model for 1 epochs

    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='/home/kb/Documents/data/Tokaido_dataset')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--device', type=str, default='cpu')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)

    # Machine parameters
    args = parser.parse_args()
    train(args)
