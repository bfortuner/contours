import numpy as np
import cv2

import preprocess
import utils


## Morphological Ops

def opening(img, ksize=(3,3), iters=(1,1)):
    # Erode --> Dilate
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    img = cv2.erode(img, erode_kernel, iterations=iters[0])
    img = cv2.dilate(img, dilate_kernel, iterations=iters[1])
    return img

def closing(img, ksize=(3,3), iters=(1,1)):
    # Dilate --> Erode
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    img = cv2.dilate(img, dilate_kernel, iterations=iters[0])
    img = cv2.erode(img, erode_kernel, iterations=iters[1])
    return img

def clean_segmentation(mask):
    mask = cv2.GaussianBlur(mask, ksize=(3,3), sigmaX=2)
    mask = closing(mask, ksize=(5,5), iters=(1,1))
    mask = opening(mask, ksize=(4,4), iters=(1,1))
    return mask



## Thresholding ##

def range_threshold(img, low, high, sigma=1.0):
    mask = cv2.inRange(img, low, high)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img, mask

def threshold_segment(slice_data, low, high):
    thresh_img, thresh_mask = range_threshold(slice_data['img'], low, high)
    out = slice_data['o_mask'].copy()
    out[thresh_mask > 1] = 0
    out = clean_segmentation(out)
    return out

def threshold_slices(slice_dicts, low, high):
    for dct in slice_dicts:
        data = preprocess.load_slice(dct)
        out = threshold_segment(data, low, high)
        dct['i_pred'] = out
    return slice_dicts



## Contours ##

def get_contours(mask):
    image, contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image

def get_max_contour(contours):
    max_cnt, max_area = None, None
    for cnt in contours:
        if max_cnt is None or cv2.contourArea(cnt) > max_area:
            max_cnt = cnt
            max_area = cv2.contourArea(cnt)
    return max_cnt

def get_nearest_poly(mask):
    contours, image = get_contours(mask.copy())
    cnt = get_max_contour(contours)

    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.01*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    tmp = np.zeros(shape=(256,256,3))
    _ = cv2.drawContours(tmp, [approx], -1, (1, 1, 1), thickness=cv2.FILLED)
    return tmp[:,:,0]



## K-Means ##

def kmeans_segment(slice_dct):
    data = preprocess.load_slice(slice_dct)
    o_mask = data['o_mask']
    i_mask = data['i_mask']
    masked_img = utils.mask_arr(data['img'], o_mask)
    masked_img = np.repeat(masked_img, 3, axis=2)
    out,poly,kmask = kmeans_poly_segment(masked_img, o_mask, k=3)
    return out,poly

def kmeans_poly_segment(img, o_mask, k):
    k_mask = quantize(img.copy(), k=3)
    mid_pixel = np.unique(k_mask)[1]
    k_mask[k_mask != mid_pixel] = 0
    k_mask[k_mask == mid_pixel] = 255
    out = o_mask.copy()
    out[k_mask[:,:,0] > 1] = 0
    out = clean_segmentation(out)
    poly = get_nearest_poly(out)
    return out, poly, k_mask

def quantize(img, k):
    Z = img.copy().reshape((-1,3)).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K=k, bestLabels=None, criteria=criteria, attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2



## Hough Circles ##

def get_circles(slice_data, min_radius_pct=0.5, max_radius_pct=0.9):
    blurred_img = cv2.GaussianBlur(slice_data['img'], ksize=(5,5), sigmaX=0)
    blurred_img = np.expand_dims(blurred_img, axis=2)

    contours, img = get_contours(slice_data['o_mask'].copy())
    center, radius = cv2.minEnclosingCircle(contours[0])

    masked_img = utils.mask_arr(blurred_img, slice_data['o_mask'])
    circles = cv2.HoughCircles(
        masked_img, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=int(radius*min_radius_pct),
        maxRadius=int(radius*1))
    if circles is None:
        circles = [[[center[0], center[1], 1]]]
    return np.uint16(np.around(circles))

def fill_circles(img, circles):
    cimg = img.copy()
    for i in circles[0,:]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (1,1,1), thickness=cv2.FILLED)
    return cimg

def plot_circles(img, circles):
    cimg = fill_circles(img, circles)
    cimg[cimg == 1] = 255
    utils.plot_arr(cimg)



## Canny Edges ##

def get_canny_edges(slice_data):
    blurred_img = cv2.GaussianBlur(slice_data['img'], ksize=(5,5), sigmaX=0)
    blurred_img = np.expand_dims(blurred_img,axis=2)

    masked_img = utils.mask_arr(blurred_img, slice_data['o_mask'])

    lower_threshold = 100
    upper_threshold = 200
    edges = cv2.Canny(masked_img, lower_threshold, upper_threshold)
    return edges
