## Instructions (Phase 1)

### Part 1: Parse the DICOM images and Contour Files

Using the functions given above, build a pipeline that will parse the DICOM images and i-contour contour files,
making sure to pair the correct DICOM image(s) with the correct contour file.
After parsing each i-contour file, make sure to translate the parsed contour to a boolean mask.

After building the pipeline, please answer the following questions:

 How did you verify that you are parsing the contours correctly?
  * I primarily relied on visual inspection, given our ground truth is the images themselves.
  * I also compared the coordinate values in the file to the coordinate values in the image

 What changes did you make to the code, if any, in order to integrate it into our production code base?
  * I created a method `process_patient_study` which handles the task of matching DICOMs with contours and ignoring images with missing contours.
  * I broke up the method `parse_dicom_file` into separate reading and cleaning/normalization components.
  * I noticed Intercept and Slope do not appear in our DICOM metadata, but it's possible they appear in other datasets, so I left them in.
  * I removed the try/catch statements when loading DICOMs to avoid swallowing unintentional bugs. Eventually try/catch may be useful, but until I have a better understanding of the types of errors we can expect, I think it's safest to leave it out.

Note: I made the assumption that we can exclude dicom image files from our training set if there isn't a corresponding i-contour.

### Part 2: Model training pipeline

Using the saved information from the DICOM images and contour files, add an additional step to the pipeline that will
load batches of data for input into a 2D deep learning model. This pipeline should meet the following criteria:

 Cycles over the entire dataset, loading a single batch (e.g. 8 observations) of inputs (DICOM image data) and targets (boolean masks) at each training step.
 A single batch of data consists of one numpy array for images and one numpy array for targets.
 Within each epoch (e.g. iteration over all studies once), samples from a batch should be loaded randomly from the entire dataset.

After building the pipeline, please answer the following questions:

 Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2?
  * I created a method `create_dataset` to preprocess all files upfront (convert DICOMs/contours into numpy arrays for training)
  * I added PyTorch as a dependency and used their built-in DataLoader class (http://pytorch.org/docs/master/data.html)
  * I implemented a custom PyTorch Dataset, `SliceDataset` which handles matching images and masks and makes data augmentation easy

 How do you/did you verify that the pipeline was working correctly?
  * Again I primarily relied on visual inspection to verify the images and masks were being stored correctly. This make me nervous :(
  * I also added meta information about patient_id and scan_id to each mini-batch for further verification

 Given the pipeline you have built, can you see any deficiencies that you would change if you had more time?
  * Right now my code expects a very specific directory structure and file naming conventions, would love to make it more extensible
  * I would like to add more tests, do some simple logging
  * I would also like to learn more about DICOM files and preprocessing
  * If I had more time I would try resampling the DICOMs to normalize pixel spacing across patients
  * I would like to add more ways to plot/visualize the images, perhaps in 3D
