import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

import pydicom
from PIL import Image, ImageDraw

import config as cfg


def get_dicom_fpath(patient_id, slice_id):
    return os.path.join(cfg.DICOM_DIR, patient_id, str(slice_id)+'.dcm')

def get_contour_fpath(original_id, slice_id, ctr_type):
    ctr_fname = 'IM-0001-{:04d}-{:s}contour-manual.txt'.format(slice_id, ctr_type)
    ctr_fpath = Path(cfg.CONTOUR_DIR, original_id, ctr_type + '-contours', ctr_fname)
    return ctr_fpath

def parse_contour_file(fpath):
    """Extract x, y coords from contour txt file

    :param fpath: contour text file
    :return: x, y coordinates
    """
    coords_lst = []
    with open(fpath, 'r') as infile:
        for line in infile:
            coords = line.strip().split()
            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))
    return coords_lst

def load_contour(original_id, slice_id, ctr_type):
    """Get contour coords if they exist, otherwise return None

    :param original_id: integer id used to find contour files
    :param slice_id: slice_id
    :param ctr_type: 'i' for inner and 'o' for outer contour types
    :return: x, y coordinates, otherwise None if contour does not exist
    """
    cntr_fpath = get_contour_fpath(original_id, slice_id, ctr_type)
    if cntr_fpath.exists():
        return parse_contour_file(cntr_fpath)
    return None

def load_dicom(patient_id, slice_id):
    """Load DICOM file by patient_id and slice_id

    :param patient_id: patient_id
    :param slice_id: slice_id
    :return: pydicom file obj
    """
    dcm_fpath = get_dicom_fpath(patient_id, slice_id)
    dcm = pydicom.read_file(dcm_fpath)
    return dcm

def extract_resample_pixel_arr(dcm):
    """Extract and resample DICOM pixel data and handle missing attributes

    TODO: Handle Pixel Spacing Differences

    :param dcm: pydicom file obj
    :return: numpy array
    """
    pixel_arr = dcm.pixel_array.copy()

    slope = dcm.get('RescaleSlope', 0.0)
    intercept = dcm.get('RescaleIntercept', 0.0)
    if intercept != 0.0 and slope != 0.0:
        pixel_arr = pixel_arr*slope + intercept

    return pixel_arr

def save_dicom_arr(dcm, patient_id, slice_id):
    """Normalize and save DICOM pixel data into numpy array

    :param dcm: pydicom file obj
    :param patient_id: patient_id
    :param slice_id: integer id attached to the dicom filename
    :return: file path of numpy pixel array
    """
    patient_dir = Path(cfg.NUMPY_DIR, patient_id)
    patient_dir.mkdir(exist_ok=True, parents=True)

    pixel_arr = extract_resample_pixel_arr(dcm)
    pixel_arr = np.expand_dims(pixel_arr, 2)

    arr_fname = '{:s}-{:d}.npy'.format(patient_id, slice_id)
    outpath = os.path.join(patient_dir, arr_fname)
    np.save(outpath, pixel_arr)

    return outpath

def coords_to_mask(coords, width, height):
    """Convert polygon coords to mask

    :param coords: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=coords, outline=0, fill=1)
    mask = np.array(img).astype('uint8')
    return mask

def save_contour_mask(coords, patient_id, slice_id, width, height, ctr_type='i'):
    """Convert x, y coords to boolean mask and save to numpy arr

    :param coords: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
    :param patient_id: patient_id
    :param slice_id: integer id attached to the dicom filename
    :param width: image width
    :param height: image height
    :return: file path of numpy mask array
    """
    patient_dir = Path(cfg.NUMPY_DIR, patient_id)
    patient_dir.mkdir(exist_ok=True, parents=True)

    mask = coords_to_mask(coords, width, height)
    mask = np.expand_dims(mask, 2)

    ctr_fname = '{:s}-{:d}-{}mask.npy'.format(patient_id, slice_id, ctr_type)
    outpath = os.path.join(cfg.NUMPY_DIR, ctr_fname)
    np.save(outpath, mask)

    return outpath

def get_patient_slice_ids(patient_id):
    """Return patient's dicom image ids (12.dcm = 12)

    :param patient_id: patient_id
    :return: List of integer ids
    """
    patient_dir = Path(cfg.DICOM_DIR, patient_id)
    fpaths = patient_dir.glob('[0-9]*.dcm')
    slice_ids = [int(f.name.split('.')[0]) for f in fpaths]
    slice_ids.sort()
    return slice_ids

def load_slice(slice_dict):
    """Store raw image and contour array data in slice metadata

    :param slice_dict: dictionary of dicom slice metadata
    :return: new slice_dict (metadata + raw data)
        {
            patient_id: patient_id str
            ...
            'img': slice img (numpy array),
            'i_mask': inner contour mask (numpy array),
            'o_mask': outer contour mask (numpy array),
            'io_mask': outer ring around inner contour (numpy array),
        }
    """
    arr = np.load(slice_dict['img_fpath'])
    i_mask_fpath = slice_dict['i_mask_fpath']
    o_mask_fpath = slice_dict['o_mask_fpath']

    i_mask, o_mask, io_mask = (None, None, None)
    if i_mask_fpath is not None and o_mask_fpath is not None:
        i_mask = np.load(i_mask_fpath)
        o_mask = np.load(o_mask_fpath)
        io_mask = np.bitwise_xor(i_mask, o_mask)
    elif i_mask_fpath is not None:
        i_mask = np.load(i_mask_fpath)
    elif o_mask_fpath is not None:
        o_mask = np.load(o_mask_fpath)

    return {
        'patient_id': slice_dict['patient_id'],
        'slice_id': slice_dict['slice_id'],
        'img': arr.astype('uint8'),
        'i_mask': i_mask.astype('uint8'),
        'o_mask': o_mask.astype('uint8'),
        'i_coords': slice_dict['i_coords'],
        'o_coords': slice_dict['o_coords'],
        'io_mask': io_mask.astype('uint8'),
    }

def process_patient_study(patient_id, original_id):
    """Create training data (array, mask) for patient's scans

    NOTE: This method ignores images that do not have both i/o contours

    :param patient_id: patient_id
    :param original_id: secondary id used to find contour files
    :return: List of dictionaries
        { 'patient_id': patient id
          'slice_id': slice id
          'img_fpath': fpath to slice numpy array
          'i_mask_fpath': fpath to i-contour numpy mask
          'o_mask_fpath': fpath to o-contour numpy mask
          'dicom_fpath': path to dicom file,
        }
    """
    slices = []
    slice_ids = get_patient_slice_ids(patient_id)
    for slice_id in slice_ids:
        dcm_fpath = get_dicom_fpath(patient_id, slice_id)
        dcm = pydicom.read_file(dcm_fpath)
        assert dcm.PatientID == patient_id

        img_fpath = save_dicom_arr(dcm, patient_id, slice_id)
        i_coords = load_contour(original_id, slice_id, ctr_type='i')
        o_coords = load_contour(original_id, slice_id, ctr_type='o')

        i_mask_fpath, o_mask_fpath = None, None
        if i_coords is not None:
            i_mask_fpath = save_contour_mask(
                i_coords, patient_id, slice_id,
                dcm.Columns, dcm.Rows, ctr_type='i')
        if o_coords is not None:
            o_mask_fpath = save_contour_mask(
                o_coords, patient_id, slice_id,
                dcm.Columns, dcm.Rows, ctr_type='o')

        # For simplicity, keep only slices with both contours
        if i_coords is not None and o_coords is not None:
            slices.append({
                'patient_id': patient_id,
                'slice_id': slice_id,
                'img_fpath': img_fpath,
                'i_mask_fpath': i_mask_fpath,
                'o_mask_fpath': o_mask_fpath,
                'i_coords': i_coords,
                'o_coords': o_coords,
                'dicom_fpath': dcm_fpath,
            })
    return slices

def create_dataset(patient_list_fpath, patient_ids=None):
    """Create training data for all patients

    :param patient_list_fpath: `link.csv`
    :param patient_ids: list of patient ids
    """
    df = pd.read_csv(patient_list_fpath)
    slices = []
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        original_id = row['original_id']
        if patient_ids is None or patient_id in patient_ids:
            patient_slices = process_patient_study(patient_id, original_id)
            slices.extend(patient_slices)
    return slices


if __name__ == "__main__":
    slices = create_dataset(cfg.PATIENT_LIST_FPATH)
