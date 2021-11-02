# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:58:46 2021

@author: narazaki
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    # pixels= img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def preprocess(img, labels):
    img = np.array(img)
    lab_1hot = np.zeros((img.shape[0], img.shape[1], len(labels)), dtype=np.bool)
    for i in range(len(labels)):
        lab_1hot[:, :, i] = img == labels[i]
    return lab_1hot


def pred2kaggle(pred_path, csv_read, csv_write, im_col, lab_col, labels, select_col=None):
    # read a csv file associated with the testing data and create Kaggle solution csv file
    # im_col: csv column idx that points to image files
    # lab_col: csv column idx that points to label files
    # select_col: csv column idx that points to selected files (default:None)
    df = pd.read_csv(csv_read, header=None)
    labFiles = list(df[lab_col])
    imFiles = list(df[im_col])
    if select_col is not None:
        imFiles = [imFiles[i] for i in range(len(imFiles)) if df[select_col][i]]
        labFiles = [labFiles[i] for i in range(len(labFiles)) if df[select_col][i]]

    data = pd.DataFrame()
    for i in range(len(imFiles)):
        try:
            m = preprocess(Image.open(os.path.join(pred_path, labFiles[i].split('\\')[-1])), labels)
        except:
            m = preprocess(np.zeros([640, 360]), labels)
        for j in range(len(labels)):
            name = labFiles[i][:-4] + '_' + str(labels[j])
            temp = pd.DataFrame.from_records([
                {
                    'ImageId': name,
                    'EncodedPixels': mask2rle(m[:, :, j]),
                }]
            )
            data = pd.concat([data, temp], ignore_index=True)
        print(i)
    if csv_write is not None:
        data.to_csv(csv_write, index=False)
    return data


if __name__ == '__main__':
    pred_path = os.path.join('D:\\', 'pred_sample')  # path to the folder that contains predicted masks
    csv_path = "D:\\Tokaido_dataset"  # path to the Tokaido Dataset folder

    # component labels
    csv_read = os.path.join(csv_path, 'files_test.csv')
    im_col = 0
    lab_col = 1
    select_col = 5
    csv_write = 'component_submission_sample.csv'
    labels = [2, 3, 4, 5, 6, 7]
    pred2kaggle(pred_path, csv_read, csv_write, im_col, lab_col, labels, select_col)

    # damage labels (viaduct images)
    csv_read = os.path.join(csv_path, 'files_test.csv')
    im_col = 0
    lab_col = 2
    select_col = 6
    labels = [2, 3]
    df0 = pred2kaggle(pred_path, csv_read, None, im_col, lab_col, labels, select_col)

    # damage labels (pure texture images)
    csv_read = os.path.join(csv_path, 'files_puretex_test.csv')
    im_col = 0
    lab_col = 1
    select_col = None
    labels = [2, 3]
    df1 = pred2kaggle(pred_path, csv_read, None, im_col, lab_col, labels, select_col)

    # write damage solution csv file
    df = pd.concat([df0, df1], ignore_index=True)
    csv_write = 'damage_submission_sample.csv'
    df.to_csv(csv_write, index=False)
