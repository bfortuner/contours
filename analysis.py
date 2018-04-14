import cv2
import numpy as np
import matplotlib.pyplot as plt

import preprocess
import utils


def accuracy(o_mask, i_mask, i_pred):
    """Pixelwise accuracy of inner contour predictions

    Calculates sums of correct inner contour predictions divided by
    total pixels inside the outer contour.

    :param o_mask: outer contour mask (numpy array)
    :param i_mask: inner contour mask (numpy array)
    :param pred_i_mask: predicted inner contour mask (numpy array)
    :return: accuracy score (float)
    """
    i_mask = i_mask.flatten()
    i_pred = i_pred.flatten()
    o_mask = o_mask.flatten()
    correct = np.where((i_pred==i_mask) & (o_mask==1))[0]
    total = np.sum(o_mask)
    return round(len(correct) / total, 4)

def get_avg_hist(slice_dict, mask_type, bins=256):
    total = np.zeros((bins,1))
    for slice_meta in slice_dict:
        slice_ = preprocess.load_slice(slice_meta)
        img = slice_['img']
        mask = slice_[mask_type]
        hist = cv2.calcHist([img],[0],mask,[bins],[0,256])
        total += hist
    return total / len(slice_dict)

def get_hist(img, mask, bins=256):
    hist = cv2.calcHist([img],[0],mask,[bins],[0,256])
    return hist

def plot_mask_type_hists(slices, bins=256):
    ax = plt.subplot(1,1,1)
    mask_types = ['i_mask', 'io_mask']
    for mask in mask_types:
        hist = get_avg_hist(slices, mask, bins)
        plt.plot(hist, label=mask)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()

def plot_slice(slice_dict, bins=256, figsize=(12, 12)):
    data = preprocess.load_slice(slice_dict)
    img = data['img']
    mask = data['io_mask']
    if len(img.shape) > 2:
        img = img.squeeze()
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    mask = np.ma.masked_where(mask == 0, mask)
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(3, 3, 1)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(mask, cmap='cool', alpha=0.4)
    ax1.set_title(slice_dict['patient_id'] + '-' + str(slice_dict['slice_id']))

    ax2 = plt.subplot(3, 3, 2)
    masked_img = np.ma.masked_where(mask == 0, img)
    plt.imshow(masked_img, cmap=plt.cm.bone)
    ax2.set_title('io_mask')

    ax3 = plt.subplot(3, 3, 3)
    color = ('b','g')
    masks = ['i_mask','io_mask']
    for i,col in enumerate(color):
        hist = cv2.calcHist([img],[0],data[masks[i]],[bins],[0,256])
        _, = plt.plot(hist, color = col, label = masks[i])
        plt.xlim([0,bins])
    plt.title('Intensity Hist')

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)
    plt.show()

def plot_patient_slices(slices, patient_id, bins):
    print("Patient_Id", patient_id)
    for slice_ in slices:
        if slice_['patient_id'] == patient_id:
            plot_slice(slice_, bins)

def test_segmentation_technique(slice_dicts, plot=False):
    total_acc = []
    for dct in slice_dicts:
        data = preprocess.load_slice(dct)
        img = data['img']
        o_mask = data['o_mask']
        i_mask = data['i_mask']
        masked_img = utils.mask_arr(img, o_mask)
        masked_img = np.repeat(masked_img, 3, axis=2)

        acc_score = accuracy(o_mask, i_mask, dct['i_pred'])
        total_acc.append(acc_score)

        if plot:
            print("Accuracy", acc_score)
            tmp = np.zeros(shape=(256,256,1))
            utils.plot_masks(tmp, [i_mask, dct['i_pred']])

    print("Mean/Med Accuracy", round(np.mean(total_acc),4), np.median(total_acc))
