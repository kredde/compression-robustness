# Effects of Quantization on Robustness

## Quick start
1. Install dependencies `conda env create -f environment.yml -n envname`
2. Set the `data_dir` and `log_dir` parameters in the `configs/config.yaml` and the `tracking_uri` in the `configs/logger/mlflow.yaml` config file.
3. For training: Set the `gpu_id` in the config file to the gpu you want to use.
4. Run an experiment using `python3 main.py +experiment=exp_name`


## MLFlow Experiment Tracking
The experiments wil be automatically tracked using MlFlow. The tracking dashboard can be accessed by starting the mlflow client in the directory of the `tracking_uri` using `mlflow ui`.


## Experiments

### Training a model

1. Choose the model that you want to train e.g. `config/experiment/resnet18`.
2. Set the hyperparameters in the `config/model/resnet18` and `config/experiment/resnet18`.
3. Run the experiment `python3 main.py +experiment=resnet18`
4. Start the MlFlow client to track the training.


### Quantizing a model
1. Start the MlFlow client and choose a model that you want to quantize. Each run has an unique `experiment_id`, copy the `experiment_id` of the training run you want to quantize.
2. Start quantization by running:
```bash
python3 evaluation.py +experiment=quantization exp_id={your_exp_id} quantization.activation_precision=7 quantization.weight_precision=8
```
If you want to quantize a model with multiple different precisions in order to compare them you can start a multi-run:
```bash
python3 evaluation.py -m +experiment=quantization exp_id={your_exp_id} quantization.activation_precision=2,3,4,5,6,7 quantization.weight_precision=2,3,4,5,6,7,8
```
or check out `scripts/evaluate_quantization.py`
3. Check the results using the MlFlow dashboard.


### Pruning a model
1. Start the MlFlow client and choose a model that you want to prune. Each run has an unique `experiment_id`, copy the `experiment_id` of the training run you want to prune.
2. Set the type of pruning you want to use in the `configs/experiment/pruning` file, choosing between `structured_lr`, `structured_random`, `unstructured_l1` and `unstructured_random`.
3. Set the pruning hyperparameters (number of iterations, amount to prune in each iteration etc.) in the respective configuration file `config/pruning/{type}`.
4. Start pruning by running:
```bash
python3 evaluation.py +experiment=pruning exp_id={your_exp_id}
```
5. Check the results using the MlFlow dashboard. Note that each pruned model is saved in the directory that is saved as `compressed_path` in the MlFlow experiment.

### Pruning + Quantization
1. Follow all the steps in the `pruning` guide.
2. Once you have a pruned model, copy its `compressed_path` in the MlFlow dashboard.
3. Start quantization by running

```bash
python3 evaluation.py +experiment=quantization exp_id={your_exp_id} compressed_path={your_compressed_path} quantization.activation_precision=7 quantization.weight_precision=8
```
Once again you can start a multi-run if you want to quantize the model with different precisions.

