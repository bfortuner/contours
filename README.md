## Setup

**Requirements**

* Python 3.4+
* Tested on Ubuntu 16.04

**Install dependencies**

```
conda env create -f environment.yml
source activate contours
```

## Unit tests

```
python -m pytest tests.py (all tests)
python -m pytest tests.py::mytestfunc (single test)
```

## Writeups

* PHASE1.md
* PHASE2.ipynb


## Resources

* http://dicomlookup.com/lookup.asp
* https://github.com/pydicom/pydicom
* https://en.wikipedia.org/wiki/DICOM
* https://github.com/KeremTurgutlu/dicom-contour
* https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
