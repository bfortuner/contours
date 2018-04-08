import os
from pathlib import Path

DATA_DIR = Path('final_data')
DICOM_DIR = Path(DATA_DIR, 'dicoms')
CONTOUR_DIR = Path(DATA_DIR, 'contourfiles')
NUMPY_DIR = Path(DATA_DIR, 'numpyfiles')

PATIENT_LIST_FPATH = os.path.join(DATA_DIR, 'link.csv')
