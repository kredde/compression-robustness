# lenet
python3 evaluation.py -m +experiment=quantization quantization.weight_precision=2,3,4,5,6,7,8 exp_id=8a56616d6d324f3a8f4879402ac1ab59 compressed_path=/data/logs/kristian/runs/default/06-58-22/13e8ff69-b35d-4cab-a469-f6948a17d43e/model_0.25.ckpt

# squeeze
python3 evaluation.py -m +experiment=quantization  quantization.weight_precision=2,3,4,5,6,7,8 exp_id=3aa717da42a94f9ea3b40457abc87cd1 compressed_path=/data/logs/kristian/runs/default/07-00-25/f4889e2e-31fb-4f76-bfc5-0771768a516f/model_0.25.ckpt

# resnet
python3 evaluation.py -m +experiment=quantization  quantization.weight_precision=2,3,4,5,6,7,8 exp_id=84bf62bc2d744a3da2835f6a143596f3 compressed_path=/data/logs/kristian/runs/default/07-03-29/afaf5fdb-ff21-4776-ab0c-9aa3e6ad0525/model_0.25.ckpt

# leplus
python3 evaluation.py -m +experiment=quantization  quantization.weight_precision=2,3,4,5,6,7,8 exp_id=8ff2305eb55949ad95f0c6251527a08a compressed_path=/data/logs/kristian/runs/default/07-07-06/9536d852-0dd4-4490-bab4-739bfbbfdce4/model_0.25.ckpt