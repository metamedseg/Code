from typing import Optional
import torch
from torch import optim, nn
import random
from meta_learning.utils import compute_iou, set_seed, get_weights, get_scheduler
from networks.unet import UNet_Cells
from torch.utils.data import SubsetRandomSampler, DataLoader
from pathlib import Path
from meta_learning.losses import IoULoss, TverskyLoss, CombinedLoss, LOSSES, CombinedLoss2
from timeit import default_timer as timer


class FineTuner:
    def __init__(self, meta_test_datasets, hyperparameters,
                 path_to_weights: Path = None, fine_tuning=True,
                 verbosity_level=2, seed=6, data_regime="all", save_weights=None):
        set_seed(seed)
        self.loss_function_type = hyperparameters["loss_type"]
        self.datasets = meta_test_datasets
        self.epochs = hyperparameters["epochs"]
        self.batch_size = hyperparameters["batch_size"]
        if fine_tuning:
            self.init_weights = torch.load(path_to_weights)
            self.ft_shots = hyperparameters["ft_shots"]
        else:
            self.init_weights = None
        self.lr = hyperparameters["lr"]
        self.wd = hyperparameters["wd"]
        self.scheduler = hyperparameters["scheduler"]
        self.fine_tuning = fine_tuning
        self.models = {}
        self.verbosity_level = verbosity_level
        # self.weighted_loss = hyperparameters["weighted_loss"]
        self.sigmoid = (self.loss_function_type == "bce")
        self.scheduler_params = hyperparameters["scheduler_params"]
        self.data_regime = data_regime
        self.output = {}
        self.save_weights = save_weights
        self.loss_function = None
        self.norm_type = hyperparameters["norm_type"]
        self.setup_loss_function()
        self.volumetric = hyperparameters["volumetric"]

    def setup_loss_function(self):
        try:
            self.loss_function = LOSSES[self.loss_function_type]
            print("Using {}".format(self.loss_function_type))
        except KeyError:
            raise Exception("Unrecognized loss type. Available are: bce, iou, tversky, combined")

    def get_dataloaders(self, dataset_id: int) -> (DataLoader, DataLoader, DataLoader):
        """Construct data loaders for the dataset with the give id
        Parameters
        ----------
        dataset_id : int
            Dataset id

        Returns
        -------
        (DataLoader, DataLoader, DataLoader)
            Train and test dataloaders.
        """
        if self.verbosity_level > 1:
            setting = "fine-tuning" if self.fine_tuning else "direct training"
            print("Creating data loaders for {}".format(setting))
        dataset = self.datasets[dataset_id]
        # get train dataset w.r.t. data regime
        train_loader = self.get_train_loader(dataset)

        if self.volumetric:
            test_volumes = dataset.test_volumes_ids
            test_indices = []
            for vol in test_volumes:
                test_indices += dataset.get_single_volume_indices(vol)
        else:
            test_indices = self.datasets[dataset_id].test_indices
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 sampler=test_sampler, shuffle=False)
        val_loader = None
        try:
            if self.volumetric:
                val_volumes = dataset.val_volumes_ids
                val_indices = []
                for vol in val_volumes:
                    val_indices += dataset.get_single_volume_indices(vol)
            else:
                val_indices = self.datasets[dataset_id].val_indices
            val_sampler = SubsetRandomSampler(val_indices)
            val_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sampler,
                                    shuffle=False)
        except KeyError:
            print("Warning. Validation dataset is not set. Please enable it in MetaDataset contstructor.")
        return train_loader, val_loader, test_loader

    def get_train_loader(self, dataset):
        if self.data_regime == "all":
            if self.verbosity_level > 0:
                print("Training on all train data")
            train_sampler = SubsetRandomSampler(dataset.train_indices)
        elif self.data_regime == "fixed":
            if self.volumetric:
                ft_volume = random.choice(dataset.train_volumes_ids)
                train_indices = dataset.get_single_volume_indices(ft_volume)
            else:
                train_indices = dataset.fixed_selection
            if self.verbosity_level > 0:
                print("Training  on {} shots".format(len(train_indices)))
            train_sampler = SubsetRandomSampler(train_indices)
        elif self.data_regime == "random_k":
            if self.verbosity_level > 0:
                print("Training on {} shots".format(self.ft_shots))
            k_shots = random.sample(dataset.train_indices, k=self.ft_shots)
            train_sampler = SubsetRandomSampler(k_shots)
        else:
            raise Exception("Data regime is not recognized. Available: all, fixed, random_k")
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  sampler=train_sampler, shuffle=False)
        return train_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluate_every: int, name):
        """Train given data in train_loader and validate every "evaluate_every" with data in validation loader.
        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        evaluate_every : int

        Returns
        -------
        """
        model, optimizer, lr_scheduler = self.setup_model()
        self.output[name]["history"] = {}
        self.output[name]["history"]["loss"] = {}
        self.output[name]["history"]["val_loss"] = {}
        self.output[name]["history"]["iou"] = {}
        self.output[name]["history"]["val_iou"] = {}
        # train loader will contain only 5 shots in fine-tuning setting, all images otherwise
        # batch size is 1 in fine-tuning setting, batch_size otherwise
        min_val_loss = None
        epochs_no_improve = 0
        for i in range(self.epochs):
            start = timer()
            loss = 0
            iou = 0
            model.train()
            if self.verbosity_level > 1:
                print("Epoch {}".format(i))
                print("Learning rate {:.6f}".format(optimizer.param_groups[0]["lr"]))
            divisor = 0
            for image, mask in train_loader:
                optimizer.zero_grad()
                divisor += image.size(0)
                if torch.cuda.is_available():
                    image, mask = image.cuda(), mask.cuda()
                output, _ = model(image)
                if self.loss_function_type in ["bce_weighted", "combined2"]:
                    current_loss = self.loss_function(pos_weight=get_weights(mask))(output, mask)
                else:
                    current_loss = self.loss_function()(output, mask)
                iou += (compute_iou(output, mask, sigmoid_applied=self.sigmoid).item() * image.size(0))
                loss += (current_loss.item() * image.size(0))
                current_loss.backward()
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            loss = loss / divisor
            iou = iou / divisor
            if self.verbosity_level > 0:
                print("Mean train {} loss for epoch {}: {:.4f}".format(self.loss_function_type, i, loss))
                print("Mean train IoU for epoch {}: {:.4f}".format(i, iou))
            if i == self.epochs - 1:
                self.output[name]["train_loss"] = loss
                self.output[name]["train_iou"] = iou
            self.output[name]["history"]["loss"][i] = loss
            self.output[name]["history"]["iou"][i] = iou
            if (i % evaluate_every == 0 or (i == self.epochs - 1)) and (val_loader is not None):
                mean_val_loss, mean_val_iou = self.validate(val_loader, model)
                print("Mean validation {} loss in epoch {}: {:.4f}".format(self.loss_function_type, i, mean_val_loss))
                print("Mean validation IoU in epoch {}: {:.4f}".format(i, mean_val_iou))
                self.output[name]["val_loss"] = mean_val_loss
                self.output[name]["val_iou"] = mean_val_iou.item()
                self.output[name]["history"]["val_loss"][i] = mean_val_loss
                self.output[name]["history"]["val_iou"][i] = mean_val_iou.item()
                """
                if min_val_loss is None:
                    min_val_loss = mean_val_loss
                elif mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve > 10:
                        print("Early stopping")
                        self.output[name]["train_loss"] = loss
                        self.output[name]["train_iou"] = iou
                        break
                """
            # Check validation loss divergence
            end = timer()
            one_epoch = (end - start) / 60
            print("Epoch completed in {:.2f} minutes. Remaining time (approx): {:.2f} minutes".format(one_epoch,
                                                                                                      one_epoch * (
                                                                                                                  self.epochs - i)))
        if self.save_weights is not None:
            torch.save(model.state_dict(), self.save_weights)
        return model

    def setup_model(self):
        # create clean model
        model = UNet_Cells(n_class=1, affine=False,
                           sigmoid=self.sigmoid, norm_type=self.norm_type)
        # if fine-tuning, load the pre-trained model
        if self.fine_tuning:
            # only load if we're fine-tuning existing model
            model.load_state_dict(self.init_weights)
            if self.verbosity_level > 0:
                print("Loaded pre-trained meta-model")
        if torch.cuda.is_available():
            model.cuda()
        optimizer = optim.Adam(params=model.parameters(), lr=self.lr,
                               weight_decay=self.wd)

        lr_scheduler = get_scheduler(optimizer, self.scheduler, self.scheduler_params)
        return model, optimizer, lr_scheduler

    def train_and_evaluate_all(self, evaluate_every: int = 2, num_shots=None):
        """Evaluate on all meta-test datasets
        num_shots : int
            Number of shots to fine-tune on. If fine_tuning is False, has no effect,
            since all dataset is used for training.
        Returns
        -------
        None
        """
        if num_shots is not None:
            self.ft_shots = num_shots
        for dataset_id in range(len(self.datasets)):
            name = self.datasets[dataset_id].name
            self.output[name] = {}
            train_loader, val_loader, test_loader = self.get_dataloaders(dataset_id)
            # train or fine_tune a model
            model = self.train(train_loader, val_loader, evaluate_every, name)
            self.models[name] = model
            test_loss, test_iou = self.validate(test_loader, model)
            print("Test {} loss: {}".format(self.loss_function_type, test_loss))
            print("Test IoU: {}".format(test_iou))
            self.output[name]["test_loss"] = test_loss
            self.output[name]["test_iou"] = test_iou.item()
            # self.evaluate(test_loader, name)

    def prepare_data_for_plotting(self, num_samples: Optional[int] = 5, dataset_ids: Optional[list] = None,
                                  dataset_names: Optional[list] = None):
        """Creates a dictionary with dataset names as keys, each entry contains another dictionary,
        with 3 keys: images, masks, outputs.
        Parameters
        ----------
        num_samples : int
            Number of samples to prepare for plotting
        dataset_ids : Optional[list]
            List of datasets ids to be plotted.
        dataset_names : Optional[list]
            Names of the datasets to be plotted.
        Returns
        -------

        """
        segmentation_results = {}
        if (dataset_ids is None) and (dataset_names is None):
            dataset_ids = range(len(self.datasets))
        elif dataset_ids is None:
            dataset_ids = [i for i in range(len(self.datasets)) if self.datasets[i].name in dataset_names]
        else:
            pass
        for i in dataset_ids:
            dataset = self.datasets[i]
            dataset_name = dataset.name
            segmentation_results[dataset_name] = {"images": [], "masks": [], "outputs": []}
            val_loader = self.get_test_loader(dataset, num_samples)
            for image, mask in val_loader:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        image, mask = image.cuda(), mask.cuda()
                    # Extract relevant fine-tuned model
                    output, _ = self.models[dataset_name](image)
                    if self.loss_function_type != "bce":
                        output = nn.Sigmoid()(output)
                    for j in range(output.shape[0]):
                        new_shape = (1, output.shape[1], output.shape[2], output.shape[3])
                        image_ = image[j].reshape(new_shape)
                        segmentation_results[dataset_name]["images"].append(image_.cpu())
                        mask_ = mask[j].reshape(new_shape)
                        segmentation_results[dataset_name]["masks"].append(mask_.cpu())
                        output_ = output[j].reshape(new_shape)
                        output_ = (output_ > 0.5).long()
                        segmentation_results[dataset_name]["outputs"].append(output_.cpu())
        return segmentation_results

    def get_test_loader(self, dataset, num_samples):
        if num_samples is not None:
            k_shot_indices = random.choices(dataset.test_indices,
                                            k=num_samples)
            sampler = SubsetRandomSampler(k_shot_indices)
        else:
            sampler = SubsetRandomSampler(dataset.test_indices)
        test_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 sampler=sampler,
                                 shuffle=False)
        return test_loader

    def validate(self, data_loader: DataLoader, model: nn.Module):
        """Computes BCE loss and IoU for the given data and model.
        Parameters
        ----------
        data_loader : DataLoader
        model : nn.Module

        Returns
        -------
        (float, float)
            Mean loss, mean IoU
        """
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_iou = 0
            divisor = 0
            for image, mask in data_loader:
                if torch.cuda.is_available():
                    image, mask = image.cuda(), mask.cuda()
                output, _ = model(image)
                if self.loss_function_type in ["weighted_bce", "combined2"]:
                    loss = self.loss_function(pos_weight=get_weights(mask))(output, mask)
                else:
                    loss = self.loss_function()(output, mask)
                val_loss += (loss.item() * image.size(0))
                val_iou += (compute_iou(output.cpu(), mask.cpu(), sigmoid_applied=self.sigmoid) * image.size(0))
                divisor += image.size(0)
            mean_val_loss = val_loss / divisor
            mean_val_iou = val_iou / divisor
        return mean_val_loss, mean_val_iou

    def get_results(self):
        return self.output
