import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import numpy as np
import skimage.transform as skTrans
from skimage.transform import resize


def show_slices(slices: np.array, columns: int = 4):
    """Shows images on the grid of given width
    Parameters
    ----------
    slices : array_like
        Array of arrays, where each array is an image.
    columns : int
        Number of columns on the grid
    Returns
    -------
    None
    """
    rows = int(np.ceil((len(slices) / columns)))
    fig_width = 4 * columns
    fig_height = 4 * rows
    fig, ax = plt.subplots(rows, columns, sharex='col', sharey='row', figsize=(fig_width, fig_height))
    for i in range(rows):
        for j in range(columns):
            image_index = i * columns + j
            if rows == 1:
                index = j
            else:
                index = (i, j)
            if image_index < len(slices):
                ax[index].imshow(slices[image_index].T, cmap="gray", origin="lower")
            else:
                dummy = np.ones_like(slices[0])
                dummy[0][0] = 0
                ax[index].imshow(dummy, cmap="gray", origin="lower")


def sample_masks(path: Path, max_samples: int = None, distinct_values: int = 2,
                 show: bool = False, thresholds=None):
    """Get a sample of valid images with at least required distinct_values. E.g. one can get all valid samples for
    0/1 or 0/1/2 segmentation.
    Example 1:
        sample_masks(path, None, 2, True)
        Will return and show all the slices with any masks (i.e. for 0/1/2 segmentation the masks where only 0/1 are
        present will still be picked)
    Example 2:
        sample_masks(path, 5, 3, True)
        Will return and show 5 slices that have values 0/1/2 (i.e. 2 objects are present)
    Parameters
    ----------
    thresholds : list
        Minimal number of pixels that should be present to consider mask valid
    path :  str
        Path to nii.gz. volume
    max_samples : int
        How many valid samples to pick
    distinct_values : int
        Indicates type of segmentation. E.g. if the mask is 0/1 distinct values should be set to 2. If the segmentation
        is 0/1/2, then distinct_values = 3. It's possible to pass 2 as
    show : bool
        Set to true if you want to show masks.
    Returns
    -------
    valid_slices: np.array
        All or n_samples slices that contain distinct_values number of unique values.
    """
    if thresholds is None:
        thresholds = [0]
    im = nib.load(str(path)).get_fdata()
    depth = im.shape[-1]
    data_array = im # skTrans.resize(im, (256, 256, depth), order=1, preserve_range=True)
    # an array to store images where both objects are present
    all_present = []
    # create a dictionary to hold a separate array for each task
    valid_slices = {}
    for i in range(1, distinct_values):
        valid_slices[i] = []
    # set to show all slices if max_samples is not specified
    if max_samples is None:
        max_samples = data_array.shape[-1]
    # for each slice
    for j in range(data_array.shape[-1]):
        # select j-th slice
        slice_ = data_array[:, :, j]
        # check if the slice contains all objects
        all_objects_present = (len(np.unique(slice_)) >= distinct_values)
        # split masks, by checking each slice for each value separately
        # set to True by default
        all_samples_found = True
        # flag if should be added to general picture
        add_all = False
        # for each value -> meaning for each separate mask value
        for k in range(1, distinct_values):
            # check threshold
            num_pixels = (slice_ == k).sum()
            if num_pixels > thresholds[k - 1]:
                # remove everything that's not the object labelled by k
                selection = (slice_ != k)
                slice_to_add = slice_.copy()
                slice_to_add[selection] = 0
                slice_to_add[slice_ == k] = 1
                if len(valid_slices[k]) < max_samples:
                    # SUPER IMPORTANT TO RESIZE  ONLY BEFORE ADDING
                    # OTHERWISE WE LOSE INFORMATION
                    slice_to_add = resize(slice_to_add, (256, 256))
                    valid_slices[k].append(slice_to_add)
                # add object to general list if any object is present enough
                add_all = True
            all_samples_found = all_samples_found and (len(valid_slices[k]) >= max_samples)
        # if all objects are there and the list is not full yet and the min threshold is passed
        if (all_objects_present and (len(all_present) < max_samples)
                and add_all):
            # SUPER IMPORTANT TO RESIZE HERE, OTHERWISE WE LOSE INFORMATION
            slice_ = resize(slice_, (256, 256))
            all_present.append(slice_)
        if all_samples_found and (len(all_present) >= max_samples):
            break
    if show:
        for k in range(1, distinct_values):
            print("Number of masks with 0/{} is {}".format(k, len(valid_slices[k])))
            show_slices(valid_slices[k])
        show_slices(all_present)
    return valid_slices, all_present
