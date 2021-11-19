import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
from tokaido_depth import TestDataset
from lmdb_data_depth_test import TokaidoLMDBDepth
from torch.utils.data import DataLoader
from data_aug import get_validation_augmentation, get_preprocessing
from submission_helper import pred2kaggle


def test_model(cfg):
    print("=" * 30 + "TESTING" + "=" * 30)

    classes = ["nonbridge", "slab", "beam", "column", "nonstructural", "rail", "sleeper"]
    model_name = Path(cfg.model_path).stem

    model = torch.load(cfg.model_path, map_location=torch.device(cfg.device))
    model.eval()
    for each_module in model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.backbone, cfg.pretrained)

    save_path = os.path.join(cfg.root_path, 'test', model_name)
    target_path = os.path.join(cfg.data_path, 'synthetic', 'test', 'labcmp')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(target_path).mkdir(parents=True, exist_ok=True)

    if not cfg.lmdb:
        x_dir = os.path.join(cfg.data_path, 'img_syn_raw', 'test')
        files = os.listdir(x_dir)
        test_dataset = TestDataset(
            x_dir,
            files,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )
    else:
        with open('test_keys', 'rb') as fp:
            test_keys = pickle.load(fp)

        test_dataset = TokaidoLMDBDepth(
            db_path=os.path.join(cfg.data_path, 'tokaido_depth_test_lmdb'),
            keys=test_keys,
            classes=classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn)
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

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

    labels = {idx: class_name for idx, class_name in enumerate(classes)}

    for image, file_name in tqdm(test_loader):
        image = image.to(cfg.device)

        predictions = model.predict(image)
        pr_mask = predictions.squeeze()
        pr_mask, pr_depth = torch.split(pr_mask, [7, 1], dim=0)
        pr_mask = pr_mask.cpu().numpy().round()
        pr_depth = pr_depth.cpu().numpy()

        # adapt for plot
        img = image.cpu().detach().numpy().squeeze()
        img = np.moveaxis(img, 0, -1)
        pr_mask = np.moveaxis(pr_mask, 0, -1)
        pr_depth = np.moveaxis(pr_depth, 0, -1)
        pr_mask = np.argmax(pr_mask, axis=-1)

        fig = plt.figure(figsize=(20, 10))
        plt.tight_layout()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.title.set_text('Image')
        ax2.title.set_text('Model segmentation')
        ax3.title.set_text('Predicted depth')

        ax1.imshow(img)

        array_show = np.array([[cmap[i] for i in j] for j in pr_mask])
        patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        ax2.imshow(array_show)
        ax2.legend(handles=patches, loc=4, borderaxespad=0.)
        ax3.imshow(pr_depth)
        plt.savefig(os.path.join(save_path, file_name[0].replace('.bmp', '.png')))

        pr_mask = pr_mask.astype(np.uint8) + 1

        Image.fromarray(pr_mask).save(os.path.join(target_path, file_name[0]))

    # component labels
    csv_read = os.path.join(cfg.data_path, 'files_test.csv')
    im_col = 0
    lab_col = 1
    select_col = 5
    csv_write = os.path.join(cfg.data_path, 'component_submission_sample.csv')
    labels = [2, 3, 4, 5, 6, 7]
    pred2kaggle(target_path, csv_read, csv_write, im_col, lab_col, labels, select_col)
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='/home/kb/Documents/data/Tokaido_dataset')
    parser.add_argument('--model-path', type=str, default='/home/kb/ownCloud/data/fpn_resnet50_monte-carlo.pth')

    # Model parameters
    parser.add_argument('--monte-carlo', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained', type=str, default='imagenet')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lmdb', action='store_true')
    parser.add_argument('--comet', action='store_true')

    # Machine parameters
    args = parser.parse_args()
    test_model(args)
