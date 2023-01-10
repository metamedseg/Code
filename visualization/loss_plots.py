import matplotlib.pyplot as plt
import numpy as np


class LossPlotter:
    def __init__(self, loss_history: dict):
        # TODO replace with more levels of extraction
        self.loss_history = loss_history["loss"]["history"]
        # self.loss_history = loss_history["history"]
        self.iou_history = loss_history["iou"]["history"]
        self.avg_loss_history = loss_history["loss"]["average"]
        self.val_epochs = list(loss_history["val_loss"].keys())
        self.avg_val_loss_history = [i.item() for i in list(loss_history["val_loss"].values())]
        # self.avg_loss_history = loss_history["average"]
        self.avg_iou_history = loss_history["iou"]["average"]
        self.avg_val_iou_history = list(loss_history["val_iou"].values())
        self.datasets = list(self.loss_history.keys())
        self.inner_epochs = len(list(self.loss_history[self.datasets[0]].values())[0])
        self.path = None
        self.prefix = None

    def datasets_used(self):
        return self.loss_history.keys()

    def plot_dataset_metric(self, key, history, ylabel, save=False):
        #x_values = np.arange(self.inner_epochs)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for epoch in history.keys():
            x_values = np.arange(len(history[epoch]))
            ax.plot(x_values, history[epoch], label="Meta-epoch {}".format(epoch))
        plt.legend(loc=1)
        plt.title("Task: {}".format(key))
        plt.xlabel("Inner epoch")
        plt.ylabel(ylabel)
        if save:
            name = self.prefix + "_" + key + ".png"
            path = self.path / ylabel
            path.mkdir(exist_ok=True)
            path = path / name
            fig.savefig(path, facecolor='white', transparent=False)
        else:
            plt.show()
        plt.close()

    def loss_per_dataset(self, dataset_id, save=False):
        # need to plot loss for each meta_epoch
        key = self.datasets[dataset_id]
        dataset_loss_history = self.loss_history[key]
        self.plot_dataset_metric(key, dataset_loss_history, "Loss", save)

    def iou_per_dataset(self, dataset_id, save=False):
        # need to plot loss for each meta_epoch
        key = self.datasets[dataset_id]
        dataset_iou_history = self.iou_history[key]
        self.plot_dataset_metric(key, dataset_iou_history, "IoU", save)

    def meta_loss(self, figsize=(10, 6), save=False):
        x_values = np.arange(len(self.avg_loss_history))
        x_values2 = self.val_epochs
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(x_values, self.avg_loss_history, label="Train")
        ax.plot(x_values2, self.avg_val_loss_history, label="Val")
        ax.plot()
        plt.xticks(x_values2)
        plt.title("Average loss over meta-epoch")
        plt.xlabel("Epoch")
        plt.legend()
        if save:
            name = self.prefix + "_average_loss.png"
            path = self.path / name
            fig.savefig(path, facecolor='white', transparent=False)
        else:
            plt.show()
        plt.close()

    def meta_iou(self, figsize=(10, 6), save=False):
        x_values = np.arange(len(self.avg_iou_history))
        x_values2 = self.val_epochs
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(x_values, self.avg_iou_history, label="Train")
        ax.plot(x_values2, self.avg_val_iou_history, label="Val")
        plt.xticks(x_values2)
        plt.title("Average iou over meta-epoch")
        plt.xlabel("Epoch")
        plt.legend()
        if save:
            name = self.prefix + "_average_iou.png"
            path = self.path / name
            fig.savefig(path, facecolor='white', transparent=False)
        else:
            plt.show()
        plt.close()

    def save_all_plots(self, path, prefix):
        # TODO: add saving IoU
        self.path = path
        self.path.mkdir(exist_ok=True)
        self.prefix = prefix
        self.meta_loss(save=True)
        self.meta_iou(save=True)
        loss_history_path = self.path / "loss"
        loss_history_path.mkdir(exist_ok=True)
        iou_history_path = self.path / "iou"
        iou_history_path.mkdir(exist_ok=True)
        for dataset_id in range(len(self.datasets)):
            self.loss_per_dataset(dataset_id, save=True)
            self.iou_per_dataset(dataset_id, save=True)


TITLE_MAPPING = {"ft": "fine-tuning", "dt": "direct training", "tr": "transfer learning"}


class FTLossPlotter:
    def __init__(self, history, path=None, prefix="ft"):
        self.history = history
        self.epochs = list(history["loss"].keys())
        self.val_epochs = list(history["val_loss"].keys())
        self.prefix = prefix
        self.path = path

    def plot_loss(self, fig_size=(10, 6), save=False, log=False):
        self.plot_train_vs_val("loss", fig_size, save, log)

    def plot_iou(self, fig_size=(10, 6), save=False, log=False):
        self.plot_train_vs_val("iou", fig_size, save, log)

    def plot_train_vs_val(self, metric, fig_size, save=False, log=False):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        y1 = list(self.history[metric].values())
        if log:
            y1 = np.log(y1)
        ax.plot(self.epochs, y1, label="Train")
        y2 = list(self.history["val_" + metric].values())
        if log:
            y2 = np.log(y2)
        ax.plot(self.val_epochs, y2, label="Val")
        plt.xticks(self.val_epochs)
        plt.title("{} for {}".format(metric, TITLE_MAPPING[self.prefix.split("_")[-1]]))
        plt.xlabel("Epoch")
        plt.legend()
        if save:
            if self.path is None:
                raise Exception("Path should be specified in plotter constructor")
            if self.prefix is None:
                raise Exception("Prefix should be specified in the constructor")
            name = self.prefix + ("_{}.png".format(metric))
            path = self.path / name
            fig.savefig(path, facecolor='white', transparent=False)
        else:
            plt.show()
        plt.close()

    def save_all_plots(self, path, fig_size=(10, 6)):
        self.path = path
        self.plot_loss(fig_size, save=True)
        self.plot_iou(fig_size, save=True)


COLOURS = {"meta": "royalblue", "direct": "black", "transfer": "peru"}


def plot_methods_comparison(sources, plot_type="loss", colors=None, save_path=None):
    """Plots train and validation curves for different fine-tuning regimes
    Parameters
    ----------
    save_path :
    sources : dict
        Dictionary must have the following structure: {"regime_name": {plot_type: {0: float, ...},
        plot_type_val: {0, float, ...}}
    plot_type : str
        What should be plotted: loss, iou?
    colors : dict
        Color mpa indicated which fine-tuning regime should be plotted with which color.
    Returns
    -------
        None
    """
    if colors is None:
        colors = COLOURS
    train_values = []
    val_values = []
    for key in sources:
        train_values_ = {"label": key, "values": list(sources[key][plot_type].values())}
        val_values_ = {"label": key, "values": list(sources[key]["val_"+plot_type].values())}
        train_values.append(train_values_)
        val_values.append(val_values_)
    epochs_train = list(sources["meta"]["loss"].keys())
    epochs_val = list(sources["meta"]["val_loss"].keys())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i in range(len(train_values)):
        label = train_values[i]["label"]
        ax.plot(epochs_train, train_values[i]["values"],
                label=(label + " train"),
                linestyle="-", color=colors[label])
        ax.plot(epochs_val, val_values[i]["values"],
                label=(label + " val"),
                linestyle="--", color=colors[label])
    plt.title("Methods comparison")
    plt.legend()
    if save_path is not None:
        fig.savefig(save_path, facecolor='white', transparent=False)
    else:
        plt.show()