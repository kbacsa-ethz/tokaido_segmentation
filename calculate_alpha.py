import os
import argparse
import re
from tqdm import tqdm
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tokaido_depth import Dataset
from lmdb_data_depth import TokaidoLMDBDepth
from torch.utils.data import DataLoader
from data_aug import get_training_augmentation, get_validation_augmentation, get_preprocessing


def calculate_alpha(cfg):

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

    model = torch.load("./models/fpn_mobilenetv2.pth", map_location=torch.device('cpu'))
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.backbone, cfg.pretrained)

    if not cfg.lmdb:
        valid_dataset = Dataset(
            x_dir,
            y_dir,
            z_dir,
            valid_files+train_files,
            valid_masks+train_masks,
            valid_depths+train_depths,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )
    else:
        valid_dataset = TokaidoLMDBDepth(
            db_path=os.path.join(cfg.data_path, 'tokaido_depth_lmdb'),
            keys=val_keys+train_keys,
            classes=classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn)
        )

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=cfg.num_workers)

    class_count = np.zeros([len(classes)])
    for _, gt_mask in tqdm(valid_loader):
        gt_mask = gt_mask[:, :len(classes)].detach().long().numpy()
        gt_mask = np.argmax(gt_mask, axis=1)
        unique, counts = np.unique(gt_mask, return_counts=True)
        dict_counts = dict(zip(unique, counts))
        for key, value in dict_counts.items():
            class_count[key] += value

    print(class_count)
    class_count /= class_count.sum()
    alpha = np.exp(-class_count)
    print(alpha)
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='/home/kb/Documents/data/Tokaido_dataset')

    # Model parameters
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2')
    parser.add_argument('--pretrained', type=str, default='imagenet')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lmdb', action='store_true')
    parser.add_argument('--comet', action='store_true')

    # Machine parameters
    args = parser.parse_args()
    calculate_alpha(args)
