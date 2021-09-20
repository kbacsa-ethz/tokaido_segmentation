import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
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

    # TODO this is not the right order
    CLASSES = ["slab", "beam", "column", "nonstructural components", "rail", "sleeper"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            image_files,
            mask_files,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in image_files]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in mask_files]

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

        #if mask is None:
        #   mask = np.zeros((360, 640), dtype=np.uint8)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


if __name__ == "__main__":
    # Lets look at data we have

    dataset = Dataset("/home/kb/Documents/data/Tokaido_dataset/img_syn_raw/train",
                      "/home/kb/Documents/data/Tokaido_dataset/synthetic/train/labcmp",
                      ["slab", "beam", "column", "nonstructural components", "rail", "sleeper"],
                      augmentation=data_aug.get_training_augmentation())

    img, msk = dataset[1]  # get some sample
    visualize(
        image=img,
        mask=msk[..., 5].squeeze(),
    )
    print("ok")
