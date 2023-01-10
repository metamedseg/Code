from typing import Union, Optional

import numpy as np
import copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

COLOR_MAPPING = {
    'TP': (0, 255, 0),  # green
    'FN': (255, 0, 0),  # red
    'FP': (255, 255, 0),  # yellow
}


def visual_confusion_matrix(output: Union[torch.Tensor, np.array],
                            mask: Union[torch.Tensor, np.array]):
    """Compute confusion matrix to compare image mask and predicted output.
    Important: output should be mapped to 0-1
    Parameters
    ----------
    output : Union[torch.Tensor, np.array]
        Predicted pixels for the image
    mask : Union[torch.Tensor, np.array]
        Actual image mask
    Returns
    -------

    """
    output_arr = output[0, 0, :, :]
    mask_arr = mask[0, 0, :, :]
    result = {"TP": np.logical_and(mask_arr, output_arr).bool().numpy(),
              "TN": np.logical_and(np.logical_not(mask_arr), np.logical_not(output_arr)).bool().numpy(),
              "FP": np.logical_and(np.logical_not(mask_arr), output_arr).bool().numpy(),
              "FN": np.logical_and(mask_arr, np.logical_not(output_arr)).bool().numpy()}
    return result


def convert_tensor_to_image_array(tensor: torch.Tensor):
    """Convert PyToch tensor to valid numpy RGB array
    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------

    """
    image = copy.copy(tensor[0, 0, :, :].numpy())
    # map 0-1 to 0-255
    image = image * 255
    # map to RGB
    image = np.dstack([image.astype(np.uint8)] * 3).copy(order='C')
    return image


def get_patch_compatible_color_mapping(color_mapping=None):
    """Generate 0-1 mapping that can be used for patches, required
    to create custom image legend.
    Parameters
    ----------
    color_mapping : dict

    Returns
    -------
    dict:
        Normalized mapping
    """
    if color_mapping is None:
        color_mapping = COLOR_MAPPING
    mapping = {}
    for key in color_mapping.keys():
        mapping[key] = tuple(c / 255 for c in color_mapping[key])
    return mapping


def generate_patches(color_mapping=None) -> list:
    """Generate patches for matplotlib legend (color values must be normalized to 0-1)
    Parameters
    ----------
    color_mapping : dict

    Returns
    -------
    list
        List of applicable patches that can be passed as handles parameter in plt.legend.
    """
    if color_mapping is None:
        color_mapping = COLOR_MAPPING
    patches = []
    for key in color_mapping.keys():
        patch = mpatches.Patch(color=tuple(c / 255 for c in color_mapping[key]), label=key)
        patches.append(patch)
    return patches


def overlay_images(processed_image: np.array, confusion_map: dict,
                   color_mapping=None, show: bool = False) -> np.array:
    """
    Parameters
    ----------
    processed_image : np.array
        valid base RGB image with the object
    confusion_map : dict
        dictionary for TP, TN, FP, (FN) with the respective True/False numpy masks (arrays)
        indicating respective entities
    color_mapping : dict
        color mapping to be used for TP, TN, FP, (FN)
    show : bool
        Indicates whether the image should be plotted. Useful for debugging. However, normally we're interested
        in simply getting the array back in order to use in in the grid image.
    Returns
    -------
    np.array
        Image with overlayed masks
    """
    if color_mapping is None:
        color_mapping = COLOR_MAPPING
    image = processed_image.copy()
    for key in color_mapping.keys():
        image[confusion_map[key]] = color_mapping[key]
    if show:
        patches = generate_patches(color_mapping)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.imshow(image, interpolation="bilinear")
    return image


def get_overlay_image(image, mask, output, show=False):
    """Overlay True positives, false positives and false negatives on the original image.
    Parameters
    ----------
    show : bool
        Indicates whether the image show be shown
    image : torch.Tensor
        Expected to have shape (1,1,M,N)
    mask : torch.Tensor
        Expected to have shape (1,1,M,N)
    output : torch.Tensor
        Expected to have shape (1,1,M,N)
    Returns
    -------
    np.array
        Image with overlay masks
    """
    confusion_map = visual_confusion_matrix(output, mask)
    rgb_image = convert_tensor_to_image_array(image)
    image = overlay_images(rgb_image, confusion_map, show=show)
    return image


def show_segmentation_results(images: list, title: str = "AAA", columns: int = 3, max_rows: int = 4):
    """Plots segmentations results on the grid
    Parameters
    ----------
    images : list
        List of processed images with overlayed results
    title : str
        Image title for the whole grid.
    columns : int
        Number of columns in the grid. By default 3
    max_rows : int
        Maximal number of rows. Extra images will be ignored.
    Returns
    -------
    None
    """
    if len(images) > columns * max_rows:
        # TODO: consider sampling randomly rather than showing first N
        images = images[:columns * max_rows]

    rows = int(np.ceil((len(images) / columns)))
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
            if image_index < len(images):
                ax[index].imshow(images[image_index])
            else:
                ax[index].text(124, 127, "x", fontsize=20)
    patches = generate_patches()
    fig.legend(handles=patches, bbox_to_anchor=(1.02, 0.88), loc='upper right', borderaxespad=0.)
    plt.suptitle(title)


def show_results_for_task(segmentation_output: dict, task_name: str):
    """Show results on a grid for single selected task.
    Parameters
    ----------
    segmentation_output : dict
        Output of prepare_data_for_plotting from reptile method
    task_name : str
    Returns
    -------
    None
    """
    outputs = segmentation_output[task_name]["outputs"]
    images = segmentation_output[task_name]["images"]
    masks = segmentation_output[task_name]["masks"]
    overlayed_images = []
    for i in range(len(outputs)):
        overlayed_images.append(get_overlay_image(images[i], masks[i], outputs[i], show=False))
    show_segmentation_results(overlayed_images, "Results for {}".format(task_name))
