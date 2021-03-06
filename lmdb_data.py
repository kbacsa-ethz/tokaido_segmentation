import pyarrow as pa
import os
import re
import lmdb
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import data_aug


class TokaidoLMDB(BaseDataset):
    CLASSES = ["nonbridge", "slab", "beam", "column", "nonstructural", "rail", "sleeper"]

    def __init__(self, db_path, keys=None, classes=None, augmentation=None, preprocessing=None):
        self.db_path = db_path

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))

        self.keys = [u'{}'.format(key).encode('ascii') for key in keys]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        image, mask = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        image = unpacked[0]

        # load label
        mask = unpacked[1]

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
        return len(self.keys)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(data_path, x_dir, y_dir, lmdb_name, write_frequency=100):
    image_files, mask_files = [], []
    for img_file in os.listdir(x_dir):
        mask_name = img_file.replace('_Scene.png', '.bmp')
        if os.path.isfile(os.path.join(y_dir, mask_name)):
            image_files.append(os.path.join(x_dir, img_file))
            mask_files.append(os.path.join(y_dir, mask_name))

    target_path = os.path.join(data_path, lmdb_name)
    isdir = os.path.isdir(target_path)

    db = lmdb.open(target_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)

    keys = []

    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[::3, ::3, :]
        mask = cv2.imread(mask_file, 0)

        case = int(re.sub("[^0-9]", "", image_file.split('_')[4]))
        frame = int(re.sub("[^0-9]", "", image_file.split('_')[5]))
        key = ((1 << case) << 16) + (1 << frame)

        txn.put(u'{}'.format(key).encode('ascii'), dumps_pyarrow((image, mask)))
        keys.append(key)
        if idx % write_frequency == 0 or idx == (len(image_files) - 1):
            print("[%d/%d]" % (idx, len(image_files)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return 0


if __name__ == "__main__":
    data_path = '/home/kb/Documents/data/Tokaido_dataset'

    x_dir = os.path.join('/home/kb/Documents/data/Tokaido_dataset', 'img_syn_raw', 'train')
    y_dir = os.path.join('/home/kb/Documents/data/Tokaido_dataset', 'synthetic', 'train', 'labcmp')
    lmdb_name = 'tokaido_lmdb'

    folder2lmdb(data_path, x_dir, y_dir, lmdb_name)

    classes = ["nonbridge", "slab", "beam", "column", "nonstructural", "rail", "sleeper"]

    ks = []
    for img_file in os.listdir(x_dir):
        foo = img_file.split('_')
        case = int(re.sub("[^0-9]", "", img_file.split('_')[1]))
        frame = int(re.sub("[^0-9]", "", img_file.split('_')[2]))
        k = ((1 << case) << 16) + (1 << frame)
        ks.append(k)

    dataset = TokaidoLMDB(db_path=os.path.join(data_path, lmdb_name),
                          keys=ks,
                          classes=classes,
                          augmentation=data_aug.get_validation_augmentation()
                          )

    img, msk = dataset[8]  # get some sample

    msk = np.argmax(msk, axis=-1)

    from skimage import color
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(color.label2rgb(msk, img))
    plt.show()
