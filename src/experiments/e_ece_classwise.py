import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from src.metrics.confidence_calibration import ConfidenceCalibration as ECE

from scipy.special import softmax
sns.set(color_codes=True)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})
sns.set_palette("tab10")

WP = "params.quantization/weight_precision"
AP = "params.quantization/activation_precision"

labels = ["airplane", "automobile", "bird", "cat",
          "deer", "dog", "frog", "horse", "ship", "truck"]


ML_FLOW_TRACKING_URI = "file:/data/logs/kristian/mlruns"


def plot_classwise_ece(exp_id: str, save_path: str, file: str = "q_preds.csv"):

    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)
    df_runs = mlflow.search_runs(
        experiment_ids=exp_id,
        filter_string="tags.type = 'quantization'",
        output_format="pandas"
    )

    runs = df_runs[df_runs[AP] == '7']

    fp_run = mlflow.search_runs(
        experiment_ids=exp_id,
        filter_string="tags.type = 'eval'",
        output_format="pandas"
    ).iloc[0]

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 8), ncols=2)

    path = fp_run["params.csv_path"]
    df = pd.read_csv(path + f"/preds.csv")

    preds = df.iloc[:, 0:10].to_numpy()
    preds = softmax(preds, axis=1)
    targs = df['targets'].to_numpy()

    ece = ECE(preds, targs, n_bins=10)
    classece = ece.get_class_wise_ece()

    grid = sns.barplot(x=classece, y=labels, ax=ax1)
    grid.tick_params(labelsize=20)

    run = runs.iloc[3]
    path = run["params.csv_path"]
    wp = run[WP]

    df = pd.read_csv(path + f"/{file}")

    preds = df.iloc[:, 0:10].to_numpy()
    preds = softmax(preds, axis=1)
    targs = df['targets'].to_numpy()

    ece = ECE(preds, targs, n_bins=10)
    classece = ece.get_class_wise_ece()

    grid = sns.barplot(x=classece, y=labels, ax=ax2)
    grid.tick_params(labelsize=20)

    ax1.set_title("Full Precision", fontsize=24)
    ax2.set_title(f"Quantization ({wp} bits)", fontsize=24)
    ax1.set_xlim(0, 0.1)
    ax2.set_xlim(0, 0.1)

    plt.savefig(save_path, bbox_inches="tight")
