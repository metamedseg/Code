import pickle
import re
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
import torch
import random
from sklearn.model_selection import train_test_split
import numpy as np
import math


def remove_close_images(path_to_organ, cut_every=2):
    brains = {}
    paths = list(path_to_organ.glob('*.*'))
    image_format = str(paths[0]).split(".")[-1]
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


def get_names_list(path: Path, cut_every=None) -> list:
    paths = path.glob("**/*.*")
    paths = [p.name for p in paths]
    if cut_every is not None:
        reduced_list = remove_close_images(path, cut_every=2)
        paths = [path for path in paths if (path in reduced_list)]
    return paths


def get_paths_list(path: Path) -> list:
    paths = path.glob("**/*.*")
    paths = [p for p in paths]
    return paths


class MetaDataset:
    """Meta dataset that contains a collection of datasets

    Attributes:
        root: Pathlike or string indicating the root directory with all datasets folders
        test_size: size of the test set. Default 0.2
        split: Information about all the train/val/test splits in each datasets. Used to maintain the same splits during
            training
        seed: Integer used to set the seed for reproducibility
        val_size: Size of validations set
        fixed_shots: Number of fixed shots to be used in training: used for fine-tuning
        dataset: Optional. Ready dataset to be wrapped in MetaDataset. Useful, when we need to pretrain on all datasets
            simultaneously. Can by any PyTorch dataset.
        volumetric: A boolean indicating whether the processing should be done based on volumes rather than images.
    """

    def __init__(self, root, target_tasks, test_size=0.2, meta_train=True, seed=6,
                 val_size=None, fixed_shots=15, dataset=None, volumetric=False, max_images=None):
        if meta_train:
            # get all directories names that are not targets
            all_tasks = [d.name for d in root.iterdir() if d.is_dir()]
            self.tasks = [task for task in all_tasks if task not in target_tasks]
        else:
            if dataset is not None:
                self.tasks = ["mixed"]
            else:
                self.tasks = target_tasks
        self.root = root
        self.test_size = test_size
        self.split = {}
        self.seed = seed
        self.val_size = val_size
        self.fixed_shots = fixed_shots
        self.dataset = dataset
        self.volumetric = volumetric
        self.max_images = max_images

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, dataset_id: int):
        # return prepared mixed dataset: required for compatibility with test setting
        if self.dataset is not None:
            return self.dataset
        dir_path: Path = self.root / self.tasks[dataset_id]
        if self.volumetric:
            dataset = VolumeDataset(dir_path, max_images_in_volume=self.max_images)
            choices = list(dataset.volumes_dict.keys())
        else:
            dataset = SegmentationDataset(dir_path)
            choices = list(range(len(dataset)))
        # TODO: manage volumetric setting
        if dataset_id not in self.split.keys():
            self.split[dataset_id] = {}
            self.split[dataset_id]["train"], self.split[dataset_id]["test"] = train_test_split(
                choices, test_size=self.test_size, random_state=self.seed)
            if self.val_size:
                self.split[dataset_id]["train"], self.split[dataset_id]["val"] = train_test_split(
                    self.split[dataset_id]["train"], test_size=self.val_size, random_state=self.seed)
            random.seed(self.seed)
            self.split[dataset_id]["fixed_train"] = random.sample(self.split[dataset_id]["train"],
                                                                  k=self.fixed_shots)
        # enforce the same split
        dataset.set_split(self.split[dataset_id])
        return dataset


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform=None, max_samples=None):
        self.root = root
        self.image_names = get_names_list((root / "images"))
        if (max_samples is not None) and ((len(self.image_names) / max_samples) >= 2):
            cut_every = (len(self.image_names) // max_samples)
            self.image_names = get_names_list((root / "images"), cut_every)
        if (max_samples is not None) and len(self.image_names) > max_samples:
            self.image_names = self.image_names[:max_samples - 1]
        self.transform = transform
        # fix and store train-test split for dataset (note - not the same as meta-train split!)
        self.splits = {}

    def set_split(self, split: dict):
        self.splits = split

    @property
    def name(self):
        return self.root.name

    @property
    def fixed_selection(self):
        return self.splits["fixed_train"]

    @property
    def train_indices(self):
        return self.splits["train"]

    @property
    def val_indices(self):
        if "val" in self.splits.keys():
            return self.splits["val"]
        else:
            raise KeyError("Validation split is disabled. Please pass val_size parameter when creating meta-dataset")

    @property
    def test_indices(self):
        return self.splits["test"]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int):
        # get image name - it's the same for mask and image, only the directories differ
        image_name = self.image_names[index]
        path_to_image = self.root / "images" / image_name
        path_to_mask = self.root / "masks" / image_name

        image = Image.open(path_to_image).convert("L")
        mask = Image.open(path_to_mask).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            """
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])
            """
            image = transforms.ToTensor()(image)  # transform(image)
            mask = (transforms.ToTensor()(mask) > 0.5).float()
        return image, mask


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform=None, datasets_selection=None, seed=555, test_size=0.2, max_samples=1000):
        self.root = root
        self.image_paths = []
        self.mask_paths = []
        self.splits = {"train": [], "val": [], "test": []}
        base = 0
        for dataset_name in datasets_selection:
            print("Adding {} to dataset".format(dataset_name))
            path_to_images_dir = root / dataset_name / "images"
            paths_to_masks_dir = root / dataset_name / "masks"
            # get all images names
            images_names_list = get_names_list(path_to_images_dir)
            # reduce dataset if there is a limit
            if (max_samples is not None) and ((len(images_names_list) / max_samples) >= 2):
                cut_every = (len(images_names_list) // max_samples)
                images_names_list = get_names_list(path_to_images_dir, cut_every)
            if (max_samples is not None) and (len(images_names_list) > max_samples):
                images_names_list = images_names_list[:max_samples - 1]
            images_paths_list = [(path_to_images_dir / image)
                                 for image in images_names_list]
            masks_paths_list = [(paths_to_masks_dir / image)
                                for image in images_names_list]
            # make train-val-test split by splitting indices
            train_indices, test_indices = train_test_split(list(range(len(images_paths_list))), test_size=test_size,
                                                           random_state=seed)
            train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=seed)
            # add number of previously added paths so that index is shifted with respect to them
            true_train_indices = list(base + np.array(train_indices))
            self.splits["train"] += true_train_indices
            true_val_incides = list(base + np.array(val_indices))
            self.splits["val"] += true_val_incides
            true_test_indices = list(base + np.array(test_indices))
            self.splits["test"] += true_test_indices
            # update base:
            base += len(images_paths_list)
            self.image_paths += images_paths_list
            self.mask_paths += masks_paths_list
        # self.image_names = get_names_list((root / "images"))
        self.transform = transform
        self.splits["fixed_train"] = self.splits["train"]
        assert len(self.image_paths) == len(self.mask_paths)

    @property
    def name(self):
        return "mixed"

    @property
    def fixed_selection(self):
        return self.splits["fixed_train"]

    @property
    def train_indices(self):
        return self.splits["train"]

    @property
    def val_indices(self):
        if "val" in self.splits.keys():
            return self.splits["val"]
        else:
            raise KeyError("Validation split is disabled. Please pass val_size parameter when creating meta-dataset")

    @property
    def test_indices(self):
        return self.splits["test"]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int):
        path_to_image = self.image_paths[index]
        path_to_mask = self.mask_paths[index]

        image = Image.open(path_to_image).convert("L")
        mask = Image.open(path_to_mask).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            """
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])
            """
            image = transforms.ToTensor()(image)  # transform(image)
            mask = (transforms.ToTensor()(mask) > 0.5).float()
        return image, mask


class FixedDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform=None, n_shots=None, seed=6):
        self.root = Path(root)
        self.image_names = sorted(get_names_list((root / "images")))
        self.transform = transform
        # fix and store train-test split for dataset (note - not the same as meta-train split!)

        self.splits = {}
        path_to_splits = Path().cwd() / "splits" / self.root.name
        print(path_to_splits)
        for regime in ["train", "test", "val"]:
            with open(str(path_to_splits / (regime + ".pkl")), 'rb') as f:
                self.splits[regime] = pickle.load(f)
        if n_shots is not None:
            random.seed(seed)
            self.splits["fixed_train"] = random.sample(self.splits["train"],
                                                       k=n_shots)

    def set_split(self, split: dict):
        self.splits = split

    @property
    def name(self):
        return self.root.name

    @property
    def fixed_selection(self):
        return self.splits["fixed_train"]

    @property
    def train_indices(self):
        return self.splits["train"]

    @property
    def val_indices(self):
        if "val" in self.splits.keys():
            return self.splits["val"]
        else:
            raise KeyError("Validation split is disabled. Please pass val_size parameter when creating meta-dataset")

    @property
    def test_indices(self):
        return self.splits["test"]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int):
        # get image name - it's the same for mask and image, only the directories differ
        image_name = self.image_names[index]
        path_to_image = self.root / "images" / image_name
        path_to_mask = self.root / "masks" / image_name
        image = Image.open(path_to_image).convert("L")
        mask = Image.open(path_to_mask).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            """
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])
            """
            image = transforms.ToTensor()(image)  # transform(image)
            mask = (transforms.ToTensor()(mask) > 0.5).float()
        return image, mask


def get_volume_to_name_mapping(image_names):
    vols = {}
    image_id = 0
    for name in image_names:
        volume_id = name.split("_")[1]
        if volume_id not in vols.keys():
            vols[volume_id] = [image_id]
        else:
            vols[volume_id].append(image_id)
        image_id += 1
    return vols


class VolumeDataset(torch.utils.data.Dataset):
    """Dataset that serves images based on volume for meta-training setting
    """
    def __init__(self, root: Path, transform=None, max_images_in_volume=None, fixed=False):
        self.root = root
        self.image_names = get_names_list((root / "images"))
        self.transform = transform
        # contains mapping from volume id (surrogate key) to image indices
        self.volumes_dict = get_volume_to_name_mapping(self.image_names)
        # Reduce volume size
        if max_images_in_volume is not None:
            for key in self.volumes_dict:
                step_size = max(1, math.ceil((len(self.volumes_dict[key]) / max_images_in_volume)))
                self.volumes_dict[key] = self.volumes_dict[key][::step_size]
        # splits are by volume
        self.splits = {}
        if fixed:
            path_to_splits = Path().cwd() / "splits" / self.root.name
            for regime in ["train", "test", "val"]:
                with open(str(path_to_splits / (regime + "_vol.pkl")), 'rb') as f:
                    self.splits[regime] = pickle.load(f)

    def get_single_volume_indices(self, volume_id):
        return self.volumes_dict[volume_id]

    @property
    def train_volumes_ids(self):
        # VOLUME (not image) indices
        return self.splits["train"]

    @property
    def val_volumes_ids(self):
        return self.splits["val"]

    @property
    def test_volumes_ids(self):
        return self.splits["test"]

    def set_split(self, split: dict):
        self.splits = split

    @property
    def name(self):
        return self.root.name

    def __len__(self):
        return len(self.volumes_dict)

    def __getitem__(self, index: int):
        # get image name - it's the same for mask and image, only the directories differ
        image_name = self.image_names[index]
        path_to_image = self.root / "images" / image_name
        path_to_mask = self.root / "masks" / image_name

        image = Image.open(path_to_image).convert("L")
        mask = Image.open(path_to_mask).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            """
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])
            """
            image = transforms.ToTensor()(image)  # transform(image)
            mask = (transforms.ToTensor()(mask) > 0.5).float()
        return image, mask
