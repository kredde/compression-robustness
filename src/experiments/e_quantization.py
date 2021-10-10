from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mlflow

sns.set(color_codes=True)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

ML_FLOW_TRACKING_URI = "file:/data/logs/kristian/mlruns"


def plot_quantization_heatmap(experiment_id: str, path: str):
    """
      Plots the accuracies of the quantized models as a heatmap.
      In order to run this the model must have been quantized with all possible quantizion precision combinations.
    """

    # get the experiment
    runs = MlflowClient(tracking_uri=ML_FLOW_TRACKING_URI).search_runs(
        experiment_ids=experiment_id,
    )

    data = np.zeros((8, 9))
    for run in runs:
        acc = run.data.metrics["q_test/acc"]
        ap = int(run.data.params["quantization/activation_precision"])
        wp = int(run.data.params["quantization/weight_precision"])
        data[ap, wp] = acc

    fig, ax = plt.subplots(figsize=(10, 8))
    fig = sns.heatmap(data[2:, 2:], annot=True, xticklabels=[2, 3, 4, 5, 6, 7, 8],
                      yticklabels=[2, 3, 4, 5, 6, 7], annot_kws={"size": 20})
    plt.xlabel("Weight precision", fontsize=20)
    plt.ylabel("Activation precision", fontsize=20)
    fig.tick_params(labelsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(path, bbox_inches="tight")


experiments = [
    # {'id': "3", 'train_id': "3", 'name': 'pruned-resnet18' },
    # {'id': "10", 'train_id': "3", 'name': 'resnet18' },
    #{'id': "7", 'train_id': "7", 'name': 'pruned-lenet' },
    #{'id': "7", 'train_id': "7", 'name': 'lenet' },
    #{'id': "11", 'name': 'lenetplus' },
    #{'id': "11", 'name': 'pruned-lenetplus' },
    #{'id': "8", 'train_id': "8", 'name': 'pruned-squeezenet' },
    #{'id': "14", 'train_id': "8", 'name': 'squeezenet' },
    {'id': "24", 'name': 'lenet'},
    {'id': "27", 'name': 'resnet18'},
    {'id': "26", 'name': 'squeezenet'},
    {'id': "25", 'name': 'lenetplus'}
]

AP = "params.quantization/activation_precision"
WP = "params.quantization/weight_precision"


def plot_weight_quantization(save_path: str, metric="metrics.q_test_acc", experiments=experiments):
    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)
    runs = []

    for exp in experiments:

        df_runs = mlflow.search_runs(
            experiment_ids=exp['id'],
            filter_string="tags.stage = 'eval' and tags.type = 'quantization'",
            output_format="pandas"
        )

        run = df_runs[df_runs[AP] == '7'].set_index(WP)
        run.sort_index(inplace=True, ascending=False)

        run['name'] = exp['name']
        runs.append((exp['name'], run))

    df = pd.DataFrame()
    for name, run in runs:
        df[name] = run[metric]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_palette("tab10")
    grid = sns.lineplot(data=df, dashes=False, marker="o")
    plt.ylim(0, 1)

    grid.set_xlabel("Weight Precision", fontsize=16)
    grid.set_ylabel("Accuracy", fontsize=16)
    grid.tick_params(labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    plt.setp(ax.get_legend().get_texts(), fontsize=14)

    plt.savefig(save_path)


def plot_activation_precision(save_path: str, metric="metrics.q_test_acc", experiments=experiments):
    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)
    runs = []

    for exp in experiments:
        df_runs = mlflow.search_runs(
            experiment_ids=exp['id'],
            filter_string="tags.stage = 'eval' and tags.type = 'quantization'",
            output_format="pandas"
        )

        run = df_runs[df_runs[WP] == '8'].set_index(AP)
        run.sort_index(inplace=True, ascending=False)

        run['name'] = exp['name']
        runs.append((exp['name'], run))

    df = pd.DataFrame()
    for name, run in runs:
        df[name] = run[metric]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_palette("tab10")
    grid = sns.lineplot(data=df, dashes=False, marker="o")
    plt.ylim(0, 1)

    grid.set_xlabel("Activation Precision", fontsize=16)
    grid.set_ylabel("Accuracy", fontsize=16)
    grid.tick_params(labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    plt.setp(ax.get_legend().get_texts(), fontsize=14)

    plt.savefig(save_path)
