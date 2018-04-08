import random
import numpy as np
import matplotlib.pyplot as plt
import pydicom


def plot_arr(arr, fs=(6,6), title=None):
    if len(arr.shape) > 2:
        arr = img_arr.squeeze()
    plt.figure(figsize=fs)
    plt.imshow(arr.astype('uint8'), cmap=plt.cm.bone)
    plt.title(title)
    plt.show()

def plot_mask_overlay(img, mask, figsize=(12, 12)):
    if len(img.shape) > 2:
        img = img.squeeze()
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    mask = np.ma.masked_where(mask == 0, mask)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(mask, cmap='cool', alpha=0.4)
    plt.show()

def plot_dataset(slices, shuffle=True, limit=10):
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
        mask = np.load(slice_['mask_fpath'])
        plot_mask_overlay(arr, mask)
