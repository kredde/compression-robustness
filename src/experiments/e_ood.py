import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve
sns.set(color_codes=True)

WP = "params.quantization/weight_precision"
AP = "params.quantization/activation_precision"
ML_FLOW_TRACKING_URI = "file:/data/logs/kristian/mlruns"


def plot_roc_auc(targets, probs):
    """
        Plots the ROC curve and AUC score of the models results
    """
    auc = roc_auc_score(targets, probs)
    print(' ROC AUC=%.5f' % (auc))

    fpr, tpr, _ = roc_curve(targets, probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    return auc


def get_runs(exp_id: str):
    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)
    return mlflow.search_runs(
        experiment_ids=EXP_ID,
        filter_string="tags.type = 'pruning + quantization + ood'",
        output_format="pandas"
    )


def auc_ood(run, file_prefix, name, save_path, type=None, quant=False):
    plt.subplots(figsize=(10, 8))
    sns.set_palette("tab10")

    path = run["params.csv_path"]

    df = pd.read_csv(path + f"/{file_prefix}preds.csv")
    df_ood = pd.read_csv(path + f"/{file_prefix}ood_preds.csv")

    preds = df.iloc[:, 0:9].to_numpy()
    preds = softmax(preds, axis=1)
    vals = np.max(preds, axis=1)

    preds_ood = df_ood.iloc[:, 0:9].to_numpy()
    preds_ood = softmax(preds_ood, axis=1)
    vals_ood = np.max(preds_ood, axis=1)

    targets = np.concatenate(
        (np.zeros(len(vals_ood)), np.ones(len(vals))))
    results = np.concatenate((vals_ood, vals))

    auc = plot_roc_auc(targets, results)
    prefix = type if type else 'fp'
    q = ''
    if quant and type:
        q = f" W{run[WP]}A{run[AP]}"
    plt.title(f"OOD {name} {prefix}{q} - AUC: {auc}")
    plt.savefig(save_path)


NAME = 'lenet'
EXP_ID = '24'
save_path = ""

runs = get_runs(EXP_ID)


auc_ood(runs.iloc[0], 'q_', NAME, save_path, type='qp', quant=True)


auc_ood(runs.iloc[0], 'q_', NAME, save_path, True, '')


auc_ood(runs.iloc[0], 'p_', NAME, save_path,  False)
