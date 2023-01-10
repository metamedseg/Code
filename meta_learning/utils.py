import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path

from torch import nn, optim


def get_weights(masks):
    # need to consider the case when it's a batch
    num_masks = masks.size(0)
    weights = torch.ones_like(masks)
    for mask_id in range(num_masks):
        pos_weights = torch.sum(masks[mask_id] == 1)
        neg_weights = torch.sum(masks[mask_id] == 0)
        neg_pos = float(neg_weights.item() / pos_weights.item())
        weights[mask_id] = weights[mask_id] * neg_pos


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


def compute_iou(outputs, masks, reduction="mean", sigmoid_applied=True):
    """Computes total or mean IoU loss for given outputs and masks.
    Parameters
    ----------
    sigmoid_applied :
    sigmoid :
    outputs :
    reduction :
    masks :

    Returns
    -------

    """
    if not sigmoid_applied:
        outputs = nn.Sigmoid()(outputs)
    # remove extra dimension
    outputs = outputs.squeeze(1)  # .byte()
    masks = masks.squeeze(1)
    outputs_ = (outputs > 0.5).long()
    # compute union and intersection and sum over image -> the result is a tensor of
    # size equal to size of the batch
    intersection = torch.logical_and(outputs_, masks).float().sum(2).sum(1)
    union = torch.logical_or(outputs_, masks).float().sum(2).sum(1)
    iou = (intersection + 0.0001) / (union + 0.0001)
    if reduction == "mean":
        return iou.mean()
    elif reduction == "sum":
        return iou.sum()
    else:
        raise Exception("Reduction not recognized")


def pack_old(res_dict, params, prefix=""):
    for key in params.keys():
        res_dict[prefix + key] = [params[key]]


def pack(prefix, res_dict, report):
    for key in report.keys():
        if key != "history":
            res_dict[prefix + key] = [report[key]]


def record_results(report_path, ex_name, meta_params, ft_params,
                   additional_params, meta_results, test_results, direct_results, transfer_results, pso_params,
                   learnt_params, scheduler_params, comment="", ):
    result = {"ex": [ex_name]}
    pack("", result, meta_params)
    pack("ft_", result, ft_params)
    pack("", result, additional_params)
    pack("meta_", result, meta_results)
    pack("ft_", result, test_results)
    pack("dt_", result, direct_results)
    pack("tr_", result, transfer_results)
    result["comment"] = comment
    result["pso_params"] = [str(pso_params)]
    result["learnt_params"] = [learnt_params]
    result["sch_params"] = str(scheduler_params)
    result = pd.DataFrame.from_dict(result)
    if Path(report_path).is_file():
        report = pd.read_csv(report_path)
        if ex_name in list(report.ex):
            raise Exception("Experiment with this name already exists")
        result = pd.concat([report, result])
    result.to_csv(report_path, index=False)


def get_next_ex_name(report_path):
    try:
        current_results = pd.read_csv(report_path)
        last_ex_num = int(current_results.ex.str.extract('(\d+)').astype(int).max())
        ex_name = "exp" + str(last_ex_num + 1)
    except FileNotFoundError:
        ex_name = "exp1"
    return ex_name


def record_direct_training_results(report_path, ex_name, dataset,
                                   params, scheduler_params, test_results, comment):
    result = {"ex": [ex_name], "dataset": [dataset]}
    pack("", result, params)
    pack("", result, test_results)
    # pack("", result, scheduler_params)
    result["comment"] = comment
    result["sch_params"] = str(scheduler_params)
    result = pd.DataFrame.from_dict(result)
    if Path(report_path).is_file():
        report = pd.read_csv(report_path)
        result = pd.concat([report, result])
    result.to_csv(report_path, index=False)


def record_ft_results(report_path, ex_group, params, scheduler_params, additional_params, ft_res, tr_res, dt_res=None):
    ft = {"ft_" + key: ft_res[key] for key in ft_res.keys() if key != "history"}
    tr = {"tr_" + key: tr_res[key] for key in tr_res.keys() if key != "history"}
    if dt_res is None:
        dt = {"dt_" + key: None for key in ft_res.keys() if key != "history"}
    else:
        dt = {"dt_" + key: dt_res[key] for key in dt_res.keys() if key != "history"}
    results = {**{"ex_group": ex_group}, **params, **additional_params, **ft, **tr, **dt}
    results["sh_params"] = str(scheduler_params)
    results = pd.DataFrame.from_records([results])
    if Path(report_path).is_file():
        report = pd.read_csv(report_path)
        results = pd.concat([report, results], ignore_index=True)
    results.to_csv(report_path, index=False)
    return results


def get_scheduler(optimizer, scheduler_type, scheduler_params):
    lr_scheduler = None
    if scheduler_type == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
        # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params["gamma"])
    elif scheduler_type == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        """
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=scheduler_params["step"],
                                                 gamma=scheduler_params["gamma"])
        """
    elif scheduler_type == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        """
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_params["gamma"],
                                                            verbose=True)
        """
    elif scheduler_type == "cyclic":
        pass
    return lr_scheduler


