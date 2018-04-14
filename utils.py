import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom


def plot_arr(arr, fs=(6,6), title=None):
    if len(arr.shape) > 2:
        arr = arr.squeeze()
    plt.figure(figsize=fs)
    plt.imshow(arr.astype('uint8'),cmap=plt.cm.gray)
    plt.title(title)
    plt.show()

def plot_mask_overlay(img, mask, figsize=(12, 12)):
    if len(img.shape) > 2:
        img = img.squeeze()
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    mask = np.ma.masked_where(mask == 0, mask)
    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(mask, cmap='cool', alpha=0.4)
    plt.subplot(1, 3, 3)
    masked_img = np.ma.masked_where(mask == 0, img)
    plt.imshow(masked_img, cmap=plt.cm.bone)
    plt.show()

def plot_masks(img, masks):
    colors = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
    ]
    img = np.repeat(img.copy(), 3, axis=2)
    overlay = np.zeros(shape=img.shape)
    for i, mask in enumerate(masks):
        for c in range(3):
            img[:,:,c][mask.squeeze() == 1] = colors[i][c]
    plot_arr(img, title='Red=Inner, Green=Pred')

def mask_arr(arr, mask):
    return np.ma.masked_where(mask == 0, arr).filled(0)

def plot_dataset(slices, mask_type='inner', shuffle=True, limit=10):
    """
    slices: dictionary of slice metadata (img/mask fpaths)
    mask_type: inner, outer
    """
    if shuffle:
        random.shuffle(slices)
    for slice_ in slices[:limit]:
        dcm = pydicom.read_file(slice_['dicom_fpath'])
        assert dcm.PatientID == slice_['patient_id']
        print("PatientId", slice_['patient_id'])
        print("SliceId", slice_['slice_id'])
        print("ImagePosition", dcm.ImagePositionPatient)
        print("ImageOrientation", dcm.ImageOrientationPatient)
        print("SliceLocation", dcm.SliceLocation)
        print("PercentSampling", dcm.PercentSampling)
        print("PixelSpacing", dcm.PixelSpacing)
        arr = np.load(slice_['img_fpath'])

        if mask_type == 'inner' and slice_['i_mask_fpath'] is not None:
            i_mask = np.load(slice_['i_mask_fpath'])
            plot_mask_overlay(arr, i_mask)
        if mask_type == 'outer' and slice_['o_mask_fpath'] is not None:
            o_mask = np.load(slice_['o_mask_fpath'])
            plot_mask_overlay(arr, o_mask)

def plot_hist(slice_dict, bins=256):
    data = utils.load_slice(slice_dict)
    img = data['img']
    color = ('b','g')
    masks = ['i_mask','io_mask']
    for i,col in enumerate(color):
        hist = cv2.calcHist([img],[0],data[masks[i]],[bins],[0,256])
        plt.plot(hist, color = col)
        plt.xlim([0,bins])
    plt.show()
