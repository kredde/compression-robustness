import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from src.metrics.confidence_calibration import ConfidenceCalibration as ECE
from sklearn.metrics import confusion_matrix, classification_report

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


def plot_cm(targs, preds):
    # confusion matrix
    cm = confusion_matrix(targs, preds)
    # quantized confusion matrix
    q_cm = confusion_matrix(q_targs, q_preds)

    # confusion matrix accuracy
    cm_acc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # quantized confusion matrix accuracy
    q_cm_acc = q_cm.astype('float') / q_cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    grid = sns.heatmap(q_cm_acc - cm_acc,
                       xticklabels=labels, yticklabels=labels)
    grid.tick_params(labelsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    grid.set_xlabel("Predicted Label", fontsize=20)
    grid.set_ylabel("Ground Truth", fontsize=20)
    plt.show()


def get_preds(exp, file="preds"):
    df = pd.read_csv(f"{exp['params.csv_path']}/{file}.csv")
    raw_preds = df.iloc[:, 0:10].to_numpy()
    soft_preds = softmax(raw_preds, axis=1)
    preds = soft_preds.argmax(1)
    targs = df['targets'].to_numpy()

    return targs, preds


eval = mlflow.search_runs(
    experiment_ids="26",
    filter_string="tags.type = 'eval'",
    output_format="pandas",
).iloc[0]

quant = mlflow.search_runs(
    experiment_ids="26",
    filter_string="tags.type = 'quantization'",
    output_format="pandas",
).iloc[0]

prun = mlflow.search_runs(
    experiment_ids="26",
    filter_string="tags.type = 'pruning' and tags.pruning_method = 'unstructured_l1'",
    output_format="pandas",
).iloc[0]

qp = mlflow.search_runs(
    experiment_ids="26",
    filter_string="tags.type = 'pruning + quantization' and tags.pruning_method = 'unstructured_l1'",
    output_format="pandas",
).iloc[0]


q_targs, q_preds = get_preds(quant, "q_preds")
targs, preds = get_preds(eval)

plot_cm(targs, preds)
