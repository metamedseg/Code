import random

import torch
from torch import optim, nn
from copy import deepcopy
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from typing import Optional
from meta_learning.utils import compute_iou, set_seed, get_weights, get_scheduler
from meta_learning.losses import IoULoss, TverskyLoss, CombinedLoss, LOSSES, CombinedLoss2
from networks.unet import UNet_Cells
from timeit import default_timer as timer

DEFAULT_PSO_PARAMS = {
    "inertia": 0.2,
    "local_acceleration": 1.49,
    "global_acceleration": 0.8,
    "parameters_range": np.array([[0.00001, 0.1], [0.00001, 0.001]]),
    "log_sampling": [True, True],
    "log_sampling_velocity": [True, False],
    "velocities_range": np.array([[0.00001, 0.1], [0.00001, 0.001]]),
    "adaptive_inertia": False,
    "n_pso_iterations": 5,
    "swarm_size": 5,
    "epochs": 7
}
DATASET_WEIGHTS = {
    'brain_edema_FLAIR': 0.08,
    'brain_edema_t1gd': 0.08,
    'brain_edema_T1w': 0.08,
    'brain_edema_T2w': 0.08,
    'brain_e_tumour_FLAIR': 0.08,
    'brain_e_tumour_t1gd': 0.08,
    'brain_e_tumour_T1w': 0.08,
    'brain_e_tumour_T2w': 0.08,
    'brain_ne_tumour_FLAIR': 0.08,
    'brain_ne_tumour_t1gd': 0.08,
    'brain_ne_tumour_T1w': 0.08,
    'brain_ne_tumour_T2w': 0.08,
    'colon': 1,
    'heart': 1,
    'hippocampus_anterior': 0.5,
    'hippocampus_posterior': 0.5,
    'lung': 1,
    'pancreas': 0.5,
    'pancreas_cancer': 0.5,
    'prostate_peripheral_T2': 0.5,
    'prostate_transitional_T2': 0.5,
    'spleen': 1,
    'vessel': 0.5,
    'vessel_cancer': 0.5}

DEFAULT_PARAMS = {
    "tasks_per_iteration": 5,
    "num_shots": 5,
    "outer_epochs": 10,
    "inner_epochs": 10,
    "weighted_loss": False,
    "inner_lr": 0.001,
    "meta_lr": 0.1,
    "inner_wd": 0.001,
    "batch_size": 8
}


class Reptile:
    def __init__(self, meta_dataset, hyperparams=None,
                 verbosity_level=2, seed=6, pso_config=None, reference_set=None,
                 volumetric=False):
        set_seed(seed)
        self.segmentation_results = {}
        if hyperparams is None:
            if verbosity_level > 0:
                print("Running with default parameters")
                print(DEFAULT_PARAMS)
            hyperparams = DEFAULT_PARAMS
        self.meta_dataset = meta_dataset
        self.tasks_per_iteration = hyperparams["tasks_per_iteration"]
        self.num_shots = hyperparams["num_shots"]
        self.outer_epochs = hyperparams["outer_epochs"]
        self.inner_epochs = hyperparams["inner_epochs"]
        # self.weighted_loss = hyperparams["weighted_loss"]
        self.batch_size = hyperparams["batch_size"]
        self.loss_function_type = hyperparams["loss_type"]
        self.scheduler = hyperparams["scheduler"]
        self.sigmoid = (self.loss_function_type == "bce")
        # False if self.weighted_loss else True
        self.norm_type = hyperparams["norm_type"]
        self.model = UNet_Cells(n_class=1, affine=False,
                                sigmoid=self.sigmoid, norm_type=self.norm_type)
        if torch.cuda.is_available():
            self.model.cuda()
        self.inner_lr = hyperparams["inner_lr"]
        self.meta_lr = hyperparams["meta_lr"]
        self.inner_wd = hyperparams["inner_wd"]
        self.meta_decay = hyperparams["meta_decay"]
        self.weighted_update_type = hyperparams["weighted_update_type"]
        # initialize with copy of the original model
        # will be used to accumulate grads over tasks
        self.accumulator = None
        self.loss_history = {}
        self.task_val_loss_history = {}
        self.iou_history = {}
        self.val_loss_history = {}
        self.val_iou_history = {}
        self.avg_loss_history = np.zeros(self.outer_epochs)
        self.avg_iou_history = np.zeros(self.outer_epochs)
        self.meta_weights_history = []
        self.verbosity_level = verbosity_level
        self.scheduler_params = hyperparams["scheduler_params"]
        self.inter_epoch_decay = hyperparams["inter_epoch_decay"]
        self.val_shots = hyperparams["val_shots"]
        self.output = {}
        self.learnt_hyperparams = {}
        self.pso_config = pso_config
        self.meta_epoch_weights_acc = []
        self.loss_function = None
        self.setup_loss_function()
        self.datasets_weights = []
        self.invert_weights = hyperparams["invert_weights"]
        self.reference_set = reference_set
        self.volumetric = volumetric

    def setup_loss_function(self):
        try:
            self.loss_function = LOSSES[self.loss_function_type]
            print("Using {}".format(self.loss_function_type))
        except KeyError:
            raise Exception("Unrecognized loss type. Available are: bce, weighted_bce, iou, tversky, combined")

    def init_accumulator(self):
        """Initializes weights accumulator. Should be used before every inner loop.
        """
        # accumulator is a state dict
        current_weights = deepcopy(self.model.state_dict())
        # use multiplication to save the same datatype
        self.accumulator = {name: 0 * current_weights[name] for name in current_weights}

    def accumulate_weights(self, task_weights: dict, weights_before: dict):
        """Updates accumulator with (W-F)
        Parameters
        ----------
        weights_before : dict
            Old model state
        task_weights : dict
            Our W, learnt from single task
        Returns
        -------
        None
            Updates self.accumulator by accumulating (W-F)
        """
        # accumulate W-F
        # F - current old model
        # W - task weights
        # acc += (W-F)
        # store task weights to recompute them later
        self.meta_epoch_weights_acc.append(task_weights)
        for name in self.accumulator:
            self.accumulator[name] += (deepcopy(task_weights[name]) - deepcopy(weights_before[name]))

    def sample_tasks(self):
        """Randomly samples predefined number of tasks from all meta-train datasets. Sampling is WITH replacement.
        Returns
        -------
        iterable
            Ids of selected datasets.
        """
        # sample datasets (=tasks) for the inner loop
        all_tasks = [self.meta_dataset[i].name for i in range(len(self.meta_dataset))]
        choice_weights = [DATASET_WEIGHTS[name] for name in all_tasks]
        datasets_selection: list = random.choices(list(range(len(all_tasks))),
                                                  k=self.tasks_per_iteration, weights=choice_weights)
        return datasets_selection

    def compute_update_weights(self, datasets_selection):
        print(self.weighted_update_type)
        n = self.tasks_per_iteration
        datasets_weights = []
        if self.weighted_update_type == "mean":
            # computes average of all new weights 
            # measure the distance from this mean  
            # give more weight to those that are further from mean
            # compute average weights
            weights_avg = {name: 0 * self.model.state_dict()[name] for name in self.model.state_dict()}
            for name in weights_avg:
                for i in range(n):
                    weights_avg[name] += self.meta_epoch_weights_acc[i][name]
                weights_avg[name] = weights_avg[name] / n
            for i in range(n):
                distance = 0
                for name in self.model.state_dict():
                    distance += ((weights_avg[name] - self.meta_epoch_weights_acc[i][name]) ** 2).sum().item()
                datasets_weights.append(distance)
        elif self.weighted_update_type == "meta":
            # measure distance from the current meta-model and give more weights to farthest
            for i in range(n):
                distance = 0
                for name in self.model.state_dict():
                    distance += ((self.model.state_dict()[name] - self.meta_epoch_weights_acc[i][
                        name]) ** 2).sum().item()
                datasets_weights.append(distance)
        elif self.weighted_update_type == "loss_based":
            print("Using loss-based update")
            # check validation loss and give more weight to the largest
            for dataset_id in datasets_selection:
                dataset_name = self.meta_dataset[dataset_id].name
                last_meta_epoch_id = max(list(self.loss_history[dataset_name].keys()))
                loss = self.loss_history[dataset_name][last_meta_epoch_id][-1]
                val_loss = self.task_val_loss_history[dataset_name][last_meta_epoch_id]
                # distance = val_loss/loss
                # TODO: consider a param lambda
                # distance = 1
                # if val_loss > loss:
                distance = 1/abs(val_loss-loss)
                datasets_weights.append(distance)
        elif self.weighted_update_type == "diff_based":
            for dataset_id in datasets_selection:
                dataset_name = self.meta_dataset[dataset_id].name
                last_meta_epoch_id = max(list(self.loss_history[dataset_name].keys()))
                loss_history = np.array(self.loss_history[dataset_name][last_meta_epoch_id])
                distance = np.max(np.abs(np.diff(loss_history)))
                datasets_weights.append(distance)
        elif self.weighted_update_type == "ft_val":
            for i in range(n):
                local_model_weights = self.meta_epoch_weights_acc[i]
                initial_iou = self.evaluate_ft(local_model_weights)
                datasets_weights.append(initial_iou)
                print(self.meta_dataset[datasets_selection[i]].name, initial_iou)
            # get metamodel for each dataset
            # evaluate on ft dataset -> use initial iou as weights
        else:
            raise Exception("Unrecognized weighted update rule")

        datasets_weights = np.array(datasets_weights)
        datasets_weights = ((datasets_weights + 0.0001) / (datasets_weights.sum() + 0.0001))
        print(datasets_weights)
        # return dataset_weights
        if self.invert_weights:
            # need to renormalize if we subtract from 1, since 0.2 becomes 0.8 and it no longer sums up to 1
            datasets_weights = (1 - datasets_weights) / (1 - datasets_weights).sum()
        self.datasets_weights.append(datasets_weights)
        return datasets_weights

    def outer_loop(self, evaluate_every: int = 2, path_to_save="meta_weights.pth"):
        """Meta-learning loop. At each epoch samples tasks randomly, for each tasks trains a new network and accumulates
        W-F, then averages updates over tasks and updates the meta-model.
        Parameters
        ----------
        evaluate_every : int
            How often we should validate (every N meta-epoch)
        path_to_save :
            Where to save meta-model weights?
        Returns
        -------
        None
        """
        for meta_epoch in range(self.outer_epochs):
            start = timer()
            weights_before: dict = deepcopy(self.model.state_dict())
            # TODO: can be used to extract the meta-model at any step
            self.meta_weights_history.append(weights_before)
            if self.verbosity_level > 0:
                print("\n\nOUTER EPOCH {}".format(meta_epoch))

            # sample datasets (=tasks) for the inner loop
            datasets_selection = self.sample_tasks()

            # initialize accumulator with zero weights
            self.init_accumulator()
            self.meta_epoch_weights_acc = []

            # compute and accumulate weight updates for each dataset, using meta-model as
            # a starting point
            val_loss = 0
            val_iou = 0
            for dataset_id in datasets_selection:
                dataset_name = self.meta_dataset[dataset_id].name
                if self.verbosity_level > 1:
                    print("Dataset {}, {}".format(dataset_id, dataset_name))

                # train in inner loop and get new weights
                task_weights, loss_history, iou_history, task_val_loss, task_val_iou = self.inner_loop(dataset_id)
                val_loss += task_val_loss
                val_iou += task_val_iou
                self.record_history(dataset_name, meta_epoch, iou_history, loss_history, task_val_loss)

                # accumulate weights using (learned weights - old weights) update
                self.accumulate_weights(task_weights, weights_before)

            n_datasets = len(datasets_selection)
            mean_val_loss = val_loss / n_datasets
            mean_val_iou = val_iou / n_datasets
            if self.verbosity_level > 0:
                print("Mean val loss: {:.4f}, mean val IoU: {:.4f}".format(mean_val_loss, mean_val_iou))
            self.val_loss_history[meta_epoch] = mean_val_loss
            self.val_iou_history[meta_epoch] = mean_val_iou
            self.output["val_loss"] = mean_val_loss.item()
            self.output["val_iou"] = mean_val_iou
            # perform meta-update
            lr = self.meta_lr
            if self.meta_decay:
                lr = self.meta_lr / (meta_epoch + 1)

            self.perform_meta_update(weights_before, lr, datasets_selection)
            mean_loss = self.avg_loss_history[meta_epoch] / len(datasets_selection)
            mean_iou = self.avg_iou_history[meta_epoch] / len(datasets_selection)
            self.avg_loss_history[meta_epoch] = mean_loss
            self.avg_iou_history[meta_epoch] = mean_iou
            if self.verbosity_level > 0:
                print("Mean train {} loss in epoch {}: {:.4f}".format(self.loss_function_type, meta_epoch, mean_loss))
                print("Mean train IoU in epoch {}: {:.4f}".format(meta_epoch, mean_iou))
            # if last epoch
            if meta_epoch == (self.outer_epochs - 1):
                self.output["train_loss"] = mean_loss
                self.output["train_iou"] = mean_iou
            if (meta_epoch % evaluate_every == 0) or (meta_epoch == (self.outer_epochs - 1)):
                # is self.val_shots is None, then validation will be performed on the whole dataset
                mean_meta_val_loss, _, mean_meta_val_iou, _ = self.get_validation_loss(num_shots=self.val_shots,
                                                                                       dataset_ids=datasets_selection,
                                                                                       regime="val")
                # TODO: decide if we want to store/plot local model validation or meta-model validation
                # record validation (test)
                # self.val_loss_history[meta_epoch] = mean_val_loss
                # self.val_iou_history[meta_epoch] = mean_val_iou
                # self.output["val_bce"] = mean_val_loss.item()
                # self.output["val_iou"] = mean_val_iou
            end = timer()
            elapsed = (end - start) / 60
            print("One epoch took {} minutes. Remaining time: {} minutes".format(elapsed,
                                                                                 elapsed * (
                                                                                         self.outer_epochs - meta_epoch)))
        # save weights
        torch.save(self.model.state_dict(), path_to_save)

    def get_weights_at_epoch(self, epoch):
        return self.meta_weights_history[epoch]

    def record_history(self, dataset_name, meta_epoch, iou_history, loss_history, val_loss):
        """Adds new items to history dictionary
        Parameters
        ----------
        dataset_name : str
            Name of the dataset for which the history should be recorded
        meta_epoch : int
            Which meta-epoch we're in
        iou_history : list
            IoU history for this meta-epoch and this dataset
        loss_history : list
            Loss history for this meta-epoch and this dataset.
        Returns
        -------
        None
        """
        if dataset_name not in self.loss_history.keys():
            self.loss_history[dataset_name] = {}
        if dataset_name not in self.iou_history.keys():
            self.iou_history[dataset_name] = {}
        if dataset_name not in self.task_val_loss_history:
            self.task_val_loss_history[dataset_name] = {}
        self.loss_history[dataset_name][meta_epoch] = loss_history
        self.task_val_loss_history[dataset_name][meta_epoch] = val_loss.item()
        self.iou_history[dataset_name][meta_epoch] = iou_history
        self.avg_loss_history[meta_epoch] += loss_history[-1]
        self.avg_iou_history[meta_epoch] += iou_history[-1]

    def inner_loop(self, dataset_id: int) -> (dict, list, list, float, float):
        """Performs inner loop: samples k shots from train subset of selected dataset, trains for
        given number of epochs on these k shots.
        Parameters
        ----------
        dataset_id : int
        Returns
        -------
        (dict, list, list, float, float)
            Dictionary of learnt weights, loss history, iou history
        """
        dataset_name = self.meta_dataset[dataset_id].name
        train_loader = self.get_train_loader(dataset_id)
        val_loader = self.get_val_loader(dataset=self.meta_dataset[dataset_id], num_shots=None, regime="val")
        model = self.setup_model()
        lr, wd = self.inner_lr, self.inner_wd
        """
        if self.pso_config is not None:
            if dataset_name in self.learnt_hyperparams:
                # if we've seen this dataset before, then just extract the learnt parameters
                params = self.learnt_hyperparams[dataset_name]
                lr, wd = params["lr"], params["wd"]
            else:
                # if we see this dataset for the first time, learn them
                # TODO: pass another indicator instaed of weighted loss!
                hp_tuner = HPOptimizer(train_loader, val_loader, model, self.weighted_loss,
                                       pso_config=self.pso_config, manual_init=None)
                lr, wd = hp_tuner.learn_hyperparams()
                self.learnt_hyperparams[dataset_name] = {"lr": lr, "wd": wd}
        """
        if (dataset_name in self.loss_history.keys()) and self.inter_epoch_decay:
            lr_divisor = len(self.loss_history[dataset_name].keys()) + 1
        else:
            lr_divisor = 1
        lr = lr / lr_divisor
        # alternative learning rate update scheme
        optimizer, lr_scheduler = self.setup_training(model, lr, wd)

        # train for inner epochs to optain W
        loss_history = []
        iou_history = []
        # min_val_loss = None
        # epochs_no_improve = 0
        for j in range(self.inner_epochs):
            loss = 0
            iou = 0
            model.train()
            if self.verbosity_level > 1:
                print("Inner Epoch {}".format(j))
                print("Learning rate {:.6f}".format(optimizer.param_groups[0]["lr"]))
            divisor = 0
            for image, mask in train_loader:
                optimizer.zero_grad()
                divisor += image.size(0)
                if torch.cuda.is_available():
                    image, mask = image.cuda(), mask.cuda()
                output, _ = model(image)
                if self.loss_function_type in ["weighted_bce", "combined2"]:
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
                print("Mean train {} loss for epoch {}: {:.4f}".format(self.loss_function_type, j, loss))
                print("Mean train IoU for epoch {}: {:.4f}".format(j, iou))
            loss_history.append(loss)
            iou_history.append(iou)
            """
            if j % 2 == 0:
                temp_val_loss, _, temp_val_iou, _ = self.get_validation_loss(num_shots=
                                                                             10, dataset_ids=[dataset_id],
                                                                             regime="val",
                                                                             model=model)
                print("Val loss: {:.4f}, val IoU: {:.4f}".format(temp_val_loss, temp_val_iou))

                if min_val_loss is None:
                    min_val_loss = temp_val_loss
                elif temp_val_loss < min_val_loss:
                    min_val_loss = temp_val_loss
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve > 3:
                        print("Early stopping")
                        break
            """
        model_state = deepcopy(model.state_dict())
        val_loss, _, val_iou, _ = self.get_validation_loss(num_shots=self.val_shots, dataset_ids=[dataset_id],
                                                           regime="val",
                                                           model=model)
        if self.verbosity_level > 0:
            print("Val loss: {:.4f}, val IoU: {:.4f}".format(val_loss, val_iou))
            print("\n")
        # return W
        return model.state_dict(), loss_history, iou_history, val_loss, val_iou

    def get_train_loader(self, dataset_id):
        dataset = self.meta_dataset[dataset_id]
        if self.volumetric:
            # randomly select one volume
            volume_id = random.choice(dataset.train_volumes_ids)
            # get indices associated with this volume
            indices = dataset.get_single_volume_indices(volume_id)
        else:
            # setup sampler and loader for train dataset for k shots
            indices = random.choices(self.meta_dataset[dataset_id].train_indices,
                                             k=self.num_shots)
        # train on each K-shots of the datasets
        sampler = SubsetRandomSampler(indices)
        batch_size = self.batch_size
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  shuffle=False)
        return train_loader

    def setup_training(self, local_model, lr, wd):
        # setup optimizer and scheduler
        optimizer = optim.Adam(params=local_model.parameters(), lr=lr,
                               weight_decay=wd)
        lr_scheduler = get_scheduler(optimizer, self.scheduler, self.scheduler_params)
        return optimizer, lr_scheduler

    def setup_model(self):
        # create and setup model
        local_model = UNet_Cells(n_class=1, affine=False,
                                 sigmoid=self.sigmoid, norm_type=self.norm_type)
        if torch.cuda.is_available():
            local_model.cuda()
        # load current model state
        local_model.load_state_dict(deepcopy(self.model.state_dict()))
        return local_model

    def get_loss_history(self) -> dict:
        """Returns all loss history generated during training. This object can be directly passed to plotting
        class.
        Returns
        -------
        dict
        """
        result = {"loss": {"history": self.loss_history,
                           "average": self.avg_loss_history},
                  "iou": {"history": self.iou_history,
                          "average": self.avg_iou_history},
                  "val_loss": self.val_loss_history,
                  "val_iou": self.val_iou_history
                  }
        return result

    def get_validation_loss(self, num_shots: int = 5, dataset_ids: list = None, regime: str = "test", model=None):
        """Computes loss for validation or test set for given datasets for given or all number of shots.
        Parameters
        ----------
        model : nn.Module
        num_shots : int
            Number of test/validation shots to be sampled from the test/validation dataset. If set to None,
            uses the full test/validation dataset.
        dataset_ids : list
            List of datasets ids. If set to None, then evaluates on all meta-train datasets.
        regime : str
            Indicates whether the evaluation should be made on test or validation dataset.

        Returns
        -------
        mean_loss, all_losses, mean_iou_loss, all_iou
            (float, list, float, list)
        """
        if self.verbosity_level > 1:
            print("Validating... on {} dataset: {}".format(regime,
                                                           "full" if num_shots is None
                                                           else (str(num_shots) + " shots")))
        all_losses = {}
        all_iou = {}
        # if not specified, use all datasets to validate
        if dataset_ids is None:
            dataset_ids = range(len(self.meta_dataset))
        for i in dataset_ids:
            dataset = self.meta_dataset[i]
            dataset_name = dataset.name
            if regime == "test":
                self.segmentation_results[dataset_name] = {"images": [], "masks": [], "outputs": []}
            val_loader = self.get_val_loader(dataset, num_shots, regime)
            loss = 0
            iou = 0
            divisor = 0
            for image, mask in val_loader:
                divisor += image.size(0)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        image, mask = image.cuda(), mask.cuda()
                    # compute validation/test loss either for local or meta-model
                    if model is None:
                        output, _ = self.model(image)
                    else:
                        output, _ = model(image)
                    if self.loss_function_type in ["weighted_bce", "combined2"]:
                        loss += self.loss_function(pos_weight=get_weights(mask))(output, mask) * image.size(0)
                        # loss += (nn.BCEWithLogitsLoss(pos_weight=get_weights(mask))(output, mask)) * image.size(0)
                        # output = nn.Sigmoid()(output)
                    else:
                        loss += self.loss_function()(output, mask) * image.size(0)
                        # loss += (nn.BCELoss()(output, mask).item() * image.size(0))
                    iou += (compute_iou(output, mask, sigmoid_applied=self.sigmoid).item() * image.size(0))
                    if regime == "test":
                        self.store_segmentation_results(dataset_name, image, mask, output)
            all_losses[dataset_name] = (loss / divisor)
            all_iou[dataset_name] = (iou / divisor)
            if self.verbosity_level > 1 and (model is None):
                print("Evaluating {}:".format(dataset.name))
                print("{} loss: {:.4f}, IoU: {:.4f}".format(self.loss_function_type,
                                                            all_losses[dataset_name],
                                                            all_iou[dataset_name]))
        mean_loss = sum(list(all_losses.values())) / len(dataset_ids)
        mean_iou_loss = sum(list(all_iou.values())) / len(dataset_ids)
        if self.verbosity_level > 0 and (model is None):
            print("Mean meta validation {} loss: {:.4f}".format(self.loss_function_type, mean_loss))
            print("Mean meta validation IoU: {:.4f}".format(mean_iou_loss))
        return mean_loss, all_losses, mean_iou_loss, all_iou

    def get_val_loader(self, dataset, num_shots, regime):
        if self.volumetric:
            indices = []
            test_volumes = dataset.test_volumes_ids
            for volume in test_volumes:
                indices += dataset.volumes_dict[volume]
        else:
            indices = dataset.test_indices
        # if validation dataset is available, use it
        if regime == "val":
            indices = []
            try:
                if self.volumetric:
                    val_volumes = dataset.val_volumes_ids
                    for volume in val_volumes:
                        indices += dataset.volumes_dict[volume]
                else:
                    indices = dataset.val_indices
            except KeyError:
                print("Note: using test dataset for validation")
        # if number of test (val) shots is given, use it
        if num_shots is not None:
            indices = random.sample(indices,
                                    k=num_shots)
        sampler = SubsetRandomSampler(indices)
        # TODO: batch size
        batch_size = self.batch_size
        val_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                shuffle=False)
        return val_loader

    def get_learnt_params(self):
        return self.learnt_hyperparams

    def store_segmentation_results(self, dataset_name, image, mask, output):
        if len(self.segmentation_results[dataset_name]["images"]) > 15:
            return
        if self.loss_function_type != "bce":
            output = nn.Sigmoid()(output)
        for j in range(output.shape[0]):
            new_shape = (1, output.shape[1], output.shape[2], output.shape[3])
            image_ = image[j].reshape(new_shape)
            self.segmentation_results[dataset_name]["images"].append(image_.cpu())
            mask_ = mask[j].reshape(new_shape)
            self.segmentation_results[dataset_name]["masks"].append(mask_.cpu())
            output_ = output[j].reshape(new_shape)
            output_ = (output_ > 0.5).long()
            self.segmentation_results[dataset_name]["outputs"].append(output_.cpu())

    # TODO: consider removing
    def prepare_data_for_plotting(self, num_samples=5, dataset_ids: Optional[list] = None,
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
        # TODO: check that it doesn't break things
        # self.model.eval()
        segmentation_results = {}
        if (dataset_ids is None) and (dataset_names is None):
            dataset_ids = range(len(self.meta_dataset))
        elif dataset_ids is None:
            dataset_ids = [i for i in range(len(self.meta_dataset)) if self.meta_dataset[i].name in dataset_names]
        else:
            pass
        for i in dataset_ids:
            dataset = self.meta_dataset[i]
            dataset_name = dataset.name
            segmentation_results[dataset_name] = {"images": [], "masks": [], "outputs": []}
            if self.volumetric:
                k_shot_indices = dataset.get_single_volume_indices(dataset.test_volumes_ids[0])
            else:
                k_shot_indices = random.choices(dataset.test_indices,
                                                k=num_samples)
            sampler = SubsetRandomSampler(k_shot_indices)
            val_loader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    sampler=sampler,
                                    shuffle=False)
            for image, mask in val_loader:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        image, mask = image.cuda(), mask.cuda()
                    output, _ = self.model(image)
                    if self.loss_function_type != "bce":
                        output = nn.Sigmoid()(output)
                    output = (output > 0.5).long()
                    for j in range(output.shape[0]):
                        new_shape = (1, output.shape[1], output.shape[2], output.shape[3])
                        image_ = image[j].reshape(new_shape)
                        segmentation_results[dataset_name]["images"].append(image_.cpu())
                        mask_ = mask[j].reshape(new_shape)
                        segmentation_results[dataset_name]["masks"].append(mask_.cpu())
                        output_ = output[j].reshape(new_shape)
                        segmentation_results[dataset_name]["outputs"].append(output_.cpu())
        return segmentation_results

    def get_meta_results(self):
        return self.output

    def perform_meta_update(self, weights_before: dict, lr: float, datasets_selection: list):
        """Performs meta-update based on the specified rule. MODIFIES meta-model.
        Parameters
        ----------
        datasets_selection :
        weights_before : dict
            meta-model weights before training local-models
        lr : float
            meta-learning rate
        Returns
        -------
        None
        """
        n_datasets = len(datasets_selection)
        if self.weighted_update_type is not None:
            datasets_weights = self.compute_update_weights(datasets_selection)
            # TODO: make sure no aliasing is happening
            # compute weighted update
            update = {name: 0.0 * self.model.state_dict()[name] for name in self.model.state_dict()}
            for name in weights_before:
                for i in range(n_datasets):
                    weight = torch.tensor(datasets_weights[i])
                    update[name] += (self.meta_epoch_weights_acc[i][name] - weights_before[name])
                    weight = weight.to(device=update[name].device)
                    update[name] *= weight
            # perform update
            self.model.load_state_dict({name: lr * update[name] + weights_before[name] for name in weights_before})
        else:
            self.model.load_state_dict({name: lr * (self.accumulator[name].true_divide(n_datasets)) +
                                              weights_before[name] for name in weights_before
                                        })

    def evaluate_ft(self, local_model_weights):
        """Evaluate initial iou on target dataset on train samples
        Parameters
        ----------
        local_model_weights : dict
        Returns
        -------
        mean_iou: float
        """
        ft_loader = self.reference_set
        model = UNet_Cells(n_class=1, affine=False,
                           sigmoid=self.sigmoid, norm_type=self.norm_type)
        if torch.cuda.is_available():
            model.cuda()
        # load current model state
        model.load_state_dict(deepcopy(local_model_weights))
        divisor = 0
        iou = 0
        for image, mask in ft_loader:
            divisor += 1
            with torch.no_grad():
                if torch.cuda.is_available():
                    image, mask = image.cuda(), mask.cuda()
                output, _ = model(image)
                if self.loss_function_type != "bce":
                    output = nn.Sigmoid()(output)
                iou += (compute_iou(output, mask, sigmoid_applied=self.sigmoid).item() * image.size(0))
        return iou / divisor
