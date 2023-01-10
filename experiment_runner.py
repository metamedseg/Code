import json

import pandas as pd
from torch.utils.data import SubsetRandomSampler, DataLoader

from datasets.meta_dataset import MetaDataset, MixedDataset, FixedDataset, VolumeDataset
from pathlib import Path
from meta_learning.reptile import Reptile
from meta_learning.fine_tuning import FineTuner
from visualization.loss_plots import LossPlotter, FTLossPlotter
from meta_learning.utils import record_results, get_next_ex_name
import argparse
from timeit import default_timer as timer
from collections import Counter

# read experiment params
# which mode to run
# which exp weights for fine-tuning
organs_root = Path().cwd() / "data" / "organs"


###############################################################################################
#                                   META-LEARNING                                             #
###############################################################################################

def meta_train(root_dir, params, exclude, target_name="meta_weights.pth"):
    report_path = Path.cwd() / "meta_results.csv"
    ex_name = get_next_ex_name(report_path)
    path = Path().cwd() / "experiments"
    path.mkdir(exist_ok=True)
    save_path = path / target_name
    if "volumetric" in params.keys():
        volumetric = params["volumetric"]
        max_images = params["max_images"]
    else:
        volumetric = False
        max_images = None
    meta_dataset = MetaDataset(root_dir, target_tasks=exclude, val_size=params["val_size"],
                               volumetric=volumetric, max_images=max_images)
    if params["weighted_update_type"] == "ft_val":
        reference_dataset = FixedDataset(root_dir / "heart", n_shots=10, seed=1)
        sampler = SubsetRandomSampler(reference_dataset.fixed_selection)
        reference_loader = DataLoader(reference_dataset, batch_size=params["batch_size"], sampler=sampler)
    else:
        reference_loader = None
    reptile = Reptile(meta_dataset, hyperparams=params,
                      verbosity_level=2, seed=6, pso_config=None,
                      reference_set=reference_loader, volumetric=volumetric)
    start = timer()
    reptile.outer_loop(evaluate_every=4, path_to_save=save_path)
    end = timer()
    elapsed = end - start
    print("Meta-training completed in {} minutes".format(elapsed / 60))
    meta_results = reptile.get_meta_results()
    results = {"ex": ex_name,
               "weights": save_path,
               "elapsed": elapsed
               }
    results = {**results, **meta_results}
    results = {**results, **params}
    results = pd.DataFrame.from_records([results])
    if Path(report_path).is_file():
        report = pd.read_csv(report_path)
        results = pd.concat([report, results], ignore_index=True)
    results.to_csv(report_path, index=False)


###############################################################################################
#                                   FINE-TUNING                                               #
###############################################################################################

def fine_tune_one_selection(root_dir, meta_weights_path, params, targets, save_pth, seed=6, data_regime="fixed"):
    if "volumetric" in params.keys():
        volumetric = params["volumetric"]
        max_images = params["max_images"]
    else:
        volumetric = False
        max_images = None
    # dataset that should be used for comparing fine-tuning
    # and direct training result. It's important to use the same dataset
    if volumetric:
        fixed_dataset = VolumeDataset(root_dir / targets[0], max_images_in_volume=max_images, fixed=True)
    else:
        fixed_dataset = FixedDataset(root_dir / targets[0], n_shots=params["ft_shots"], seed=seed)
    meta_test_dataset = MetaDataset(root_dir, target_tasks=targets,
                                    meta_train=False, val_size=0.1, fixed_shots=params["ft_shots"], seed=seed,
                                    dataset=fixed_dataset)
    # fine-tune meta-weights
    trainer = FineTuner(meta_test_datasets=meta_test_dataset,
                        hyperparameters=params, path_to_weights=meta_weights_path,
                        data_regime=data_regime, save_weights=save_pth)
    results = {}
    start = timer()
    trainer.train_and_evaluate_all()
    end = timer()
    elapsed = end - start
    print("Fine_tuning completed in {} seconds.".format(elapsed))
    test_results = trainer.get_results()[targets[0]]
    results["train_loss"], results["train_iou"] = test_results["train_loss"], test_results[
        "train_iou"]
    results["val_loss"], results["val_iou"] = test_results["val_loss"], test_results["val_iou"]
    results["test_loss"], results["test_iou"] = test_results["test_loss"], test_results["test_iou"]
    results["elapsed"] = elapsed
    return results


def fine_tune(root_dir, weights_path, params, targets, save_pth, report_path=None, key="meta", num_selections=5):
    # create single dataset for testing
    # TODO: let user set the seeds
    if report_path is None:
        report_path = Path.cwd() / "ft_results.csv"
    seeds = [1, 6, 25, 55, 1100]
    seeds = seeds[:num_selections]
    results = {
        "weights": weights_path,
        "target": targets,
        "exp_type": key
    }
    data_regime = params["data_regime"]
    results = {**results, **params}
    avg_results = Counter({})
    test_iou = []
    for seed in seeds:
        partial_results = Counter(fine_tune_one_selection(root_dir=root_dir, meta_weights_path=weights_path,
                                                          params=params, targets=targets, seed=seed,
                                                          data_regime=data_regime, save_pth=save_pth))
        avg_results += partial_results
        test_iou.append(partial_results["test_iou"])
    avg_results = {name: (value if name == "elapsed" else value / len(seeds)) for name, value in avg_results.items()}
    results = {**results, **avg_results}
    results["test_iou_all"] = test_iou
    results = pd.DataFrame.from_records([results])
    if Path(report_path).is_file():
        report = pd.read_csv(report_path)
        results = pd.concat([report, results], ignore_index=True)
    results.to_csv(report_path, index=False)
    print(results)
    return results


###############################################################################################
#                                   TRANSFER WEIGHTS                                          #
###############################################################################################

pretrain_weights_path = Path.cwd() / "pretrained_global.pth"


def pre_train_on_all(root_dir, weights_path, params, max_samples=1000,
                     tasks_selection=None):
    if tasks_selection is None:
        excluded = ["heart"]
        all_tasks = [d.name for d in root_dir.iterdir() if d.is_dir()]
        tasks_selection = [task for task in all_tasks if task not in excluded]
    mixed_dataset = MixedDataset(root_dir, datasets_selection=tasks_selection, max_samples=max_samples)
    mixed_metadataset = MetaDataset(root_dir, target_tasks=["None"],
                                    meta_train=False, val_size=0.1, fixed_shots=0, dataset=mixed_dataset)
    transfer_trainer = FineTuner(meta_test_datasets=mixed_metadataset,
                                 hyperparameters=params, fine_tuning=False,
                                 data_regime="all", verbosity_level=0,
                                 save_weights=weights_path)
    start = timer()
    transfer_trainer.train_and_evaluate_all()
    end = timer()
    print("Training completed in {} seconds. Results are save in {}".format(end - start, weights_path))


###############################################################################################
#                                   STANDARD SUPERVISED                                       #
###############################################################################################


def train_directly(root_dir, params, weights_path, target):
    fixed_dataset = FixedDataset(root_dir / target)

    dataset = MetaDataset(root_dir, target_tasks=[target],
                          meta_train=False, val_size=0.1, fixed_shots=10, seed=1,
                          dataset=fixed_dataset)
    direct_trainer = FineTuner(meta_test_datasets=dataset,
                               hyperparameters=params, fine_tuning=False,
                               data_regime="all", verbosity_level=2,
                               save_weights=weights_path)
    start = timer()
    direct_trainer.train_and_evaluate_all()
    end = timer()
    print("Training completed in {} seconds. Results are save in {}".format(end - start, weights_path))


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_type", type=str, help="Experiment type", default="ft")
    parser.add_argument("--data_dir", type=str, help="Rood dir with organs datasets",
                        default=Path().cwd() / "data" / "organs")
    parser.add_argument("--params_file", type=str, help="Path to file with params", default="ft_params.json")
    parser.add_argument("--target", type=str, help="Target organ", default="heart")
    parser.add_argument("--weights", type=str, help="Weights path", default=None)
    parser.add_argument("--verbose", default=False, action='store_true')
    parser.add_argument("--save", default=None, help="Path to save weights after fine-tuning")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    exp_type = args.exp_type
    params_file = args.params_file
    with open(params_file) as json_file:
        data = json.load(json_file)
    if data.get("targets") is None:
        target = [args.target]
        print(target)
    else:
        target = data.get("targets")
    if data.get("weights_path") is None:
        print("It is none")
        weights_path = args.weights
        print(weights_path)
    else:
        weights_path = data.get("weights_path")
    if exp_type == "ft":
        print(data.get("params"))
        print("Starting fine-tuning")
        save_path = args.save
        print("Save path {}".format(save_path))
        fine_tune(data_dir, weights_path=weights_path,
                  params=data.get("params"), targets=target,
                  report_path=data.get("report_path"), key=data.get("exp_type"),
                  num_selections=data.get("num_selections"), save_pth=save_path)
    elif exp_type == "meta":
        print("Starting meta-learning")
        meta_train(data_dir, params=data.get("params"), exclude=data.get("exclude"),
                   target_name=data.get("weights_name"))
    elif exp_type == "pretrain":
        print("Starting pre-training")

        pre_train_on_all(root_dir=data_dir, weights_path=data["weights_path"],
                         max_samples=data["max_samples"],
                         params=data["params"], tasks_selection=data["datasets"])
    elif exp_type == "dt":
        train_directly(data_dir, params=data.get("params"), weights_path="dt_training.pth", target=data.get("target"))
        print("Starting standard supervised training")
    else:
        raise Exception("Unknown type of experiment. Available are: ft, meta, pretrain, dt")
