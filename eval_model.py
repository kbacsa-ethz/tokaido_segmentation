import os
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp
from tokaido_depth import Dataset
from lmdb_data_depth import TokaidoLMDBDepth
from torch.utils.data import DataLoader
from data_aug import get_training_augmentation, get_validation_augmentation, get_preprocessing


def eval_model(cfg):

    model_path = "./models/fpn_resnet50_focal.pth"
    model_name = Path(model_path).stem

    save_path = os.path.join(cfg.root_path, 'results', model_name)
    Path(save_path).mkdir(parents=True, exist_ok=True)

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

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.backbone, cfg.pretrained)

    if not cfg.lmdb:
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
        valid_dataset = TokaidoLMDBDepth(
            db_path=os.path.join(cfg.data_path, 'tokaido_depth_lmdb'),
            keys=val_keys,
            classes=classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn)
        )

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=cfg.num_workers)

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

    sample_num = 0
    confusion_array = np.zeros([7, 7])
    for image, gt_mask in tqdm(valid_loader):
        # inference
        gt_mask = gt_mask.squeeze()
        gt_depth = gt_mask[-1]
        gt_mask = gt_mask[:len(classes)]
        #x_tensor = image.to(cfg.device).unsqueeze(0)
        pr_mask = model.predict(image)
        pr_mask = pr_mask.squeeze()
        pr_mask, pr_depth = torch.split(pr_mask, [7, 1], dim=0)
        pr_mask = pr_mask.cpu().numpy().round()
        pr_depth = pr_depth.cpu().numpy()

        # adapt for plot
        img = image.detach().numpy().squeeze()
        img = np.moveaxis(img, 0, -1)
        gt_mask = np.moveaxis(gt_mask.detach().numpy(), 0, -1)
        pr_mask = np.moveaxis(pr_mask, 0, -1)
        pr_depth = np.moveaxis(pr_depth, 0, -1)
        gt_mask = np.argmax(gt_mask, axis=-1)
        pr_mask = np.argmax(pr_mask, axis=-1)

        confusion_array += confusion_matrix(gt_mask.flatten(), pr_mask.flatten(), labels=list(range(7)), normalize=None)

        fig = plt.figure(figsize=(20, 10))
        plt.tight_layout()
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax1.title.set_text('Image')
        ax2.title.set_text('Model segmentation')
        ax3.title.set_text('Ground truth')
        ax4.title.set_text('Predicted depth')
        ax5.title.set_text('True depth')

        ax1.imshow(img)

        array_show = np.array([[cmap[i] for i in j] for j in pr_mask])
        patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        ax2.imshow(array_show)
        ax2.legend(handles=patches, loc=4, borderaxespad=0.)

        array_show = np.array([[cmap[i] for i in j] for j in gt_mask])
        patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        ax3.imshow(array_show)
        ax3.legend(handles=patches, loc=4, borderaxespad=0.)

        ax4.imshow(pr_depth)
        ax5.imshow(gt_depth)
        plt.savefig(os.path.join(save_path, "{:05d}.png".format(sample_num)))
        sample_num += 1
        #plt.show()

    import pandas as pd
    import seaborn as sn

    df_cm = pd.DataFrame(confusion_array, range(confusion_array.shape[0]), range(confusion_array.shape[0]))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.savefig(os.path.join(save_path, "confusion.png"))
    #plt.show()

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parse args")

    # I/O parameters
    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='/home/kb/Documents/data/Tokaido_dataset')

    # Model parameters
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
    eval_model(args)
