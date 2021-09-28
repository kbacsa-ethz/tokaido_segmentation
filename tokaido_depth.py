import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import Dataset as BaseDataset
import data_aug


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    """Tokaido Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["nonbridge", "slab", "beam", "column", "nonstructural", "rail", "sleeper"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            depth_dir,
            image_files,
            mask_files,
            depth_files,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in image_files]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in mask_files]
        self.depth_fps = [os.path.join(depth_dir, image_id) for image_id in depth_files]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[::3, ::3, :]
        mask = cv2.imread(self.masks_fps[i], 0)
        depth = cv2.imread(self.depth_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            depth = torch.from_numpy(np.float32(depth)) / 255

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, depth

    def __len__(self):
        return len(self.images_fps)


if __name__ == "__main__":
    x_dir = os.path.join('/home/kb/Documents/data/Tokaido_dataset', 'img_syn_raw', 'train')
    y_dir = os.path.join('/home/kb/Documents/data/Tokaido_dataset', 'synthetic', 'train', 'labcmp')
    z_dir = os.path.join('/home/kb/Documents/data/Tokaido_dataset', 'synthetic', 'train', 'depth')

    classes = ["nonbridge", "slab", "beam", "column", "nonstructural", "rail", "sleeper"]

    dataset = Dataset(x_dir,
                      y_dir,
                      z_dir,
                      os.listdir(x_dir),
                      os.listdir(y_dir),
                      os.listdir(z_dir),
                      classes,
                      augmentation=data_aug.get_validation_augmentation())

    img, msk, dpt = dataset[8]  # get some sample

    print(dpt)
    print(dpt.max())
    print(dpt.min())
    exit()

    msk = np.argmax(msk, axis=-1)

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)

    values = list(range(5))
    t = 1
    cmap = {
        1: [1.0, 0.0, 0.0, t],
        2: [0.7, 0.3, 0.1, t],
        3: [1.0, 0.6, 0.2, t],
        4: [0.0, 1.0, 0.0, t],
        5: [0.1, 0.7, 0.3, t],
        6: [0.2, 1.0, 0.6, t],
        7: [0.0, 0.0, 1.0, t],
    }

    labels = {idx+1: class_name for idx, class_name in enumerate(classes)}
    arrayShow = np.array([[cmap[i] for i in j] for j in msk])
    ## create patches as legend
    patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
    plt.imshow(arrayShow)
    plt.legend(handles=patches, loc=4, borderaxespad=0.)

    plt.subplot(1, 3, 3)
    plt.imshow(dpt)
    plt.show()

