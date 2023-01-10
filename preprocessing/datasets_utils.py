import argparse
import json
import re
from random import random

import pylab
from matplotlib import cm
from skimage import img_as_ubyte
import nibabel as nib
from pathlib import Path
from skimage.transform import resize
from skimage.io import imsave
import shutil
import numpy as np
brain_slices = ["BRATS_006","BRATS_021","BRATS_283"]
TASKS = {
    "Task01_BrainTumour": {"name": "brain",
                           "classes": ["edema", "ne_tumour", "e_tumour"],
                           "modes": ["FLAIR", "T1w", "t1gd", "T2w"],
                           "include_modes": [True, True, True, True],
                           "thresholds": [1000, 400, 600],
                           "normalize": False,
                           "type": "MRI"
                           },

    "Task02_Heart": {"name": "heart",
                     "classes": [""],
                     "modes": [""],
                     "include_modes": [True],
                     "thresholds": [700],
                     "normalize": False,
                     "type": "T2"},

    "Task04_Hippocampus": {"name": "hippocampus",
                           "classes": ["anterior", "posterior"],
                           "modes": [""],
                           "include_modes": [True],
                           "thresholds": [100, 100],
                           "normalize": False,
                           "type": "MRI"},

    "Task05_Prostate": {"name": "prostate",
                        "classes": ["peripheral", "transitional"],
                        "modes": ["T2", "ADC"],
                        "include_modes": [True, False],
                        "thresholds": [300, 600],
                        "normalize": False,
                        "type": "T2"},

    "Task06_Lung": {"name": "lung",
                    "classes": [""],
                    "modes": [""],
                    "include_modes": [True],
                    "thresholds": [1000],
                    "normalize": True,
                    "type": "CT"},

    "Task07_Pancreas": {"name": "pancreas",
                        "classes": ["", "cancer"],
                        "modes": [""],
                        "include_modes": [True],
                        "thresholds": [1000, 1000],
                        "normalize": True,
                        "type": "CT"},

    "Task08_HepaticVessel": {"name": "vessel",
                             "classes": ["", "cancer"],
                             "modes": [""],
                             "include_modes": [True],
                             "thresholds": [1000, 1000],
                             "normalize": True,
                             "type": "CT"},

    "Task09_Spleen": {"name": "spleen",
                      "classes": [""],
                      "modes": [""],
                      "include_modes": [True],
                      "thresholds": [600],
                      "normalize": True,
                      "type": "CT"},

    "Task10_Colon": {"name": "colon",
                     "classes": [""],
                     "modes": [""],
                     "include_modes": [True],
                     "thresholds": [400],
                     "normalize": True,
                     "type": "CT"}
}


# processes one volume
def clip_range(vol):
    # remove bones from CT scan
    return np.clip(vol, -1000, 400)


def norm_volume(vol, scaling="mean"):
    # ignore zero values
    if scaling == "mean":
        mask = (vol != 0.0)
        non_zero_values = vol[mask]
        vol_mean = np.mean(non_zero_values)
        print(vol_mean)
        vol_std = np.std(non_zero_values)
        vol = (vol - vol_mean) / vol_std
    else:
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    return vol


def convert_and_save(image_volume, mask_volume, task_metadata, norm_ct=False, normalize_volume=False,
                     output_format="png",
                     size=(256, 256), root_dir=Path().cwd() / "data" / "organs"):
    base_name = (image_volume.name).split(".")[0]
    volume = nib.load(str(image_volume)).get_fdata()
    mask = nib.load(str(mask_volume)).get_fdata()
    # check that number of slices is that same in image and mask
    n_slices = mask.shape[2]
    assert volume.shape[2] == mask.shape[2]

    # split into volumes by mode
    volumes_by_mode = []
    if len(task_metadata["modes"]) > 1:
        for i in range(len(task_metadata["modes"])):
            if task_metadata["include_modes"][i]:
                single_mode_volume = volume[:, :, :, i]
                volumes_by_mode.append(single_mode_volume)
            else:
                volumes_by_mode.append(None)
    else:
        volumes_by_mode = [volume]
    # print("Task {}".format(task_metadata["name"]))
    for i, mode in enumerate(task_metadata["modes"]):
        if task_metadata["include_modes"][i]:
            volume_i = volumes_by_mode[i]
            # TODO: clip volume
            if norm_ct and task_metadata["type"] == "CT":
                volume_i = clip_range(volume_i)
            # TODO: normalize volume. ACHNTUNG: ignore zero pixels
            if normalize_volume:
                volume_i = norm_volume(volume_i, scaling="min")
            # TODO: find mean and SD of the volume to normalize later after the
            for j in range(n_slices):
                # get slice
                image_slice = volume_i[:, :, j]
                image_resized = resize(image_slice, size)
                mask_slice = mask[:, :, j]
                assert image_slice.shape == mask_slice.shape
                # here the fun with different things on one image starts:
                for k, subtask in enumerate(task_metadata["classes"]):
                    directory_name = (task_metadata["name"] +
                                      ("_" + subtask if subtask != "" else "") +
                                      ("_" + mode if mode != "" else ""))
                    # define target directories (depend on the subtask)
                    image_target_dir = root_dir / directory_name / "images"
                    mask_target_dir = root_dir / directory_name / "masks"
                    # check if the mask satisfies threshold
                    add_slice = (mask_slice == (k + 1)).sum() >= task_metadata["thresholds"][k]
                    if add_slice:
                        image_target = image_target_dir / (base_name + "_slice" + str(j) + ".{}".format(output_format))
                        mask_target = mask_target_dir / (base_name + "_slice" + str(j) + ".{}".format(output_format))
                        # print("Slice should be added")
                        k_mask = mask_slice.copy()
                        # zero everything that's not region of interest
                        k_mask[k_mask != (k + 1)] = 0
                        k_mask[k_mask == (k + 1)] = 1
                        k_mask = resize(k_mask, size)
                        # if random() > 0.8:
                        #    print("We are going to save image {} with shape {},\nmask with shape {}, to {} and {}".
                        #          format(base_name, str(image_resized.shape),
                        #                 str(k_mask.shape), str(image_target), str(mask_target)))
                        pylab.imsave(str(image_target), image_resized, format=output_format, cmap=cm.Greys_r)
                        pylab.imsave(str(mask_target), k_mask, format=output_format, cmap=cm.Greys_r)
                        # imsave(str(image_target), img_as_ubyte(image_resized))
                        # imsave(str(mask_target), img_as_ubyte(k_mask))
                        # TODO: count ignored slices
                        # else
                        # print("--")
                        # print("No, should not be added")

                    # print("Subtask {} with values {}".format(directory_name, k + 1))


def process_all_tasks(root: Path = Path.cwd().parent, tasks_list: dict = TASKS.keys(), norm_ct=False,
                      normalize_volume=False,
                      output_format="png"):
    for key in tasks_list:
        path = root / key
        images_path = path / "imagesTr"
        masks_path = path / "labelsTr"
        target_dir_root = root / "data" / "organs"
        target_dir_root.mkdir(exist_ok=True)
        # create necessary directories structure
        for subtask in TASKS[key]["classes"]:
            # generate directories for subtask
            subtask_global_name = TASKS[key]["name"] + (("_" + subtask) if subtask != "" else "")
            for (mode, include) in zip(TASKS[key]["modes"], TASKS[key]["include_modes"]):
                if include:
                    subtask_name = subtask_global_name + (("_" + mode) if mode != "" else "")
                    subtask_directory = target_dir_root / subtask_name
                    subtask_directory.mkdir(exist_ok=True)
                    subtask_directory_images = subtask_directory / "images"
                    subtask_directory_images.mkdir(exist_ok=True)
                    subtask_directory_masks = subtask_directory / "masks"
                    subtask_directory_masks.mkdir(exist_ok=True)
                    # get paths to raw images and masks
        images_file_paths = images_path.glob("**/*.nii.gz")
        # masks_file_paths = masks_path.glob("**/*.nii.gz")
        for path in images_file_paths:
            # parts = str(path).split("\\")
            if not (path.name).startswith("._"):
                # file name - must be the same for masks and image
                image_base_name = (path.name).split(".")[0]
                image_path = images_path / (image_base_name + ".nii.gz")
                mask_path = masks_path / (image_base_name + ".nii.gz")
                # TODO: replace with param name
                convert_and_save(image_volume=image_path,
                                 mask_volume=mask_path, task_metadata=TASKS[key], norm_ct=norm_ct,
                                 normalize_volume=normalize_volume, output_format=output_format,
                                 size=(256, 256))
        # process labels first to decide, which images and in which dimension are going


def generate_smaller_dataset(regime="perc", value=0.01, root=None, target_dir=None):
    """Generates a smaller dataset for testing purposes
    Parameters
    ----------
    regime : str
        Regime to resample the dataset. Available options:
        - "perc". Samples certain % of data (value) from each dataset
        - "fixed". Samples specified number (value) of images from each dataset
        - "periodial". Samples with specified step (value)
    value : Union[float, int]
        Depending on the regime, means:
        - percentage for "perc" regime
        - number of images for "fixed" regime
        - step for "periodical" regime
    root : Path
        Root directory containing the original datasets
    target_dir : Path
        Target directory where the sample should be copied
    Returns
    -------
        None
    """
    # script to generate a smaller dataset
    if root is None:
        root = Path.cwd() / "data" / "organs"
    if target_dir is None:
        target_dir = Path.cwd() / "data" / "organs_small"
    target_dir.mkdir(exist_ok=True)
    all_tasks = [d.name for d in root.iterdir() if d.is_dir()]
    for task in all_tasks:
        task_dir = root / task
        task_target_dir = target_dir / task
        task_target_dir.mkdir(exist_ok=True)
        images_dir = task_dir / "images"
        images_target_dir = task_target_dir / "images"
        images_target_dir.mkdir(exist_ok=True)
        masks_dir = task_dir / "masks"
        masks_target_dir = task_target_dir / "masks"
        masks_target_dir.mkdir(exist_ok=True)
        all_images = np.array([im.name for im in (images_dir).iterdir()])
        if regime is "perc":
            assert value is not None
            assert ((value > 0) and (value < 1))
            num_values = int(len(all_images) * value)
            # sample image names (images and masks have exactly the same names)
            choices_names = random.sample(list(all_images), num_values)
        elif regime is "fixed":
            assert (value > 1)
            choices_names = random.sample(list(all_images), min(value, len(all_images)))
        elif regime is "periodical":
            # select every nth element
            choices_names = all_images[::value]
        else:
            raise Exception("Regime is not recognized")
        for choice in choices_names:
            shutil.copy(images_dir / choice, images_target_dir / choice)
            shutil.copy(masks_dir / choice, masks_target_dir / choice)


def remove_close_images(path_to_organ, cut_every=2, image_format="png"):
    brains = {}
    paths = path_to_organ.glob('**/*.{}'.format(image_format))
    current_slice = 0
    stem = None
    for path in paths:
        split = str(path).split("_")
        if stem is None:
            stem = split[-3]
            stem = re.findall(r'\w+', split[-3])[-1]
        if (current_slice != split[-2]) or (current_slice == 0):
            current_slice = split[-2]
            brains[current_slice] = [int(re.findall(r'\d+', split[-1])[0])]
        else:
            brains[current_slice] += [int(re.findall(r'\d+', split[-1])[0])]
    for key in brains:
        outputs = []
        slices = np.array(brains[key])
        slices.sort()
        if len(slices) > 20:
            diffs = np.diff(slices)
            large_diffs = list(np.where(diffs > 1))[0]
            if len(large_diffs) > 0:
                outputs += [slices[i + 1] for i in large_diffs]
                slices = np.delete(slices, large_diffs)
                outputs += list(slices[::cut_every])
                brains[key] = outputs
            else:
                brains[key] = list(slices[::cut_every])
    output_paths = []
    for key in brains:
        output_paths += [("{}_{}_slice{}.{}".format(stem, key, i, image_format)) for i in brains[key]]
    return output_paths
