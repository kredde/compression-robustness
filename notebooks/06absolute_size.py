# %%
from IPython import get_ipython

# %%
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import os
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})
mlflow.set_tracking_uri("file:/data/logs/kristian/mlruns")

labels = ["airplane", "automobile", "bird", "cat",
          "deer", "dog", "frog", "horse", "ship", "truck"]
WP = "params.quantization/weight_precision"
AP = "params.quantization/activation_precision"


# %%
experiments = [
    {'id': "10", 'train_id': "3", 'name': 'resnet18'},
    {'id': "14", 'train_id': "8", 'name': 'squeezenet'},
    {'id': "11", 'name': 'lenetplus'},
    {'id': "7", 'train_id': "7", 'name': 'lenet'},
    # {'id': "27", 'name': 'resnet18' },
    # {'id': "26", 'name': 'squeezenet' },
    # {'id': "25", 'name': 'lenetplus' },
    # {'id': "24", 'name': 'lenet' },
]


# %%
data = []
for exp in experiments:
    d = mlflow.search_runs(
        experiment_ids=exp["id"],
        filter_string="tags.stage = 'eval'",
        output_format="pandas",
    )
    if 'train_id' in exp.keys():
        d2 = mlflow.search_runs(
            experiment_ids=exp["train_id"],
            filter_string="tags.stage = 'eval'",
            output_format="pandas",
        )
        d = pd.concat([d, d2])

    data.append({'name': exp["name"], 'data': d})


# %%
results = []
for exp in data:
    d = exp["data"]

    quant = d[d["tags.type"] == "quantization"]
    quant = quant[quant[AP] == "7"]
    for i in range(len(quant)):
        run = quant.iloc[i]
        params = run["params.model/params_trainable"]

        results.append({"name": exp["name"], "size": int(
            params) * int(run[WP]), "acc": run["metrics.q_test/acc"], "type": f'q-{exp["name"]}'})

    quant_prun = d[d["tags.type"] == "pruning + quantization"]
    for i in range(len(quant_prun)):
        run = quant_prun.iloc[i]
        params = run["params.model/params_trainable"]

        results.append({"name": exp["name"],
                        "size": int(params) * 0.25 * int(run[WP]),
                        "acc": run["metrics.q_test/acc"],
                        "type": f'pq-{exp["name"]}'})


# %%
df = pd.DataFrame(results)


# %%
fig, ax = plt.subplots(figsize=(10, 7))
sns.set_palette(sns.color_palette("Paired", 9))
grid = sns.lineplot(data=df, x="size", y="acc",
                    hue="type", marker="o", legend="brief")
grid.set(xscale="log")
grid.set_xlabel("Absolute Size", fontsize=16)
grid.set_ylabel("Accuracy", fontsize=16)
grid.tick_params(labelsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.setp(ax.get_legend().get_texts(), fontsize=14)

plt.ylim(0, 1)
plt.savefig(
    "/home/k.schwienbacher/quantization-robustness/data/comparison/absolute-log.pdf")
