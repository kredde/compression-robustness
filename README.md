# Effects of Quantization on Robustness

## Quick start
1. Install dependencies `conda env create -f environment.yml -n envname`
2. Set the `data_dir` and `log_dir` in the config and the `tracking_uri` in the mlflow logger config file.
3. For training: Set the `gpu_id` in the config file to the gpu you want to use.
4. Run an experiment using `python3 main.py +experiment=exp_name`


## MLFlow
1. Start the mlflow client in the directory of the `tracking_uri` using `mlflow ui`
2. Use a reverse tunnel `ssh -L 5000:localhost:5000 user@remote -N` to access the mlflow ui in the browser. E.g `ssh -L 5000:localhost:5000 k.schwienbacher@yavis -N`
