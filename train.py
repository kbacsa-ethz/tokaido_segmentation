import os
import re
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tokaido_data import Dataset
from torch.utils.data import DataLoader
from data_aug import get_training_augmentation, get_validation_augmentation, get_preprocessing


def train():
    data_dir = "/home/kb/Documents/data/Tokaido_dataset"

    x_dir = os.path.join(data_dir, 'img_syn_raw', 'train')
    y_dir = os.path.join(data_dir, 'synthetic', 'train', 'labcmp')

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

    encoder = 'efficientnet-b0'
    encoder_weights = 'imagenet'
    activation = 'sigmoid'
    device = 'cpu'

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(classes),
        activation=activation
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
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
    train()
