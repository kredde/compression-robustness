# lenet
python3 evaluation.py -m +experiment=evaluate quantization.activation_precision=2,3,4,5,6 exp_id=77cfa9c0651d4fe694a5e38898e464c4
python3 evaluation.py -m +experiment=evaluate quantization.weight_precision=2,3,4,5,6,7,8 exp_id=77cfa9c0651d4fe694a5e38898e464c4

# squeeze
# python3 evaluation.py -m +experiment=evaluate quantization.activation_precision=2,3,4,5,6 exp_name=CIFAR_SQUEEZE_EVAL exp_id=2322cbf079bf4072a7ca40d3710b7575
# python3 evaluation.py -m +experiment=evaluate  quantization.weight_precision=2,3,4,5,6,7,8 exp_name=CIFAR_SQUEEZE_EVAL exp_id=2322cbf079bf4072a7ca40d3710b7575

# # resnet
# python3 evaluation.py -m +experiment=evaluate quantization.activation_precision=2,3,4,5,6 exp_name=CIFAR_RESNET18_EVAL exp_id=40c2862ddf934b42835f8fc21089e5be
# python3 evaluation.py -m +experiment=evaluate  quantization.weight_precision=2,3,4,5,6,7,8 exp_name=CIFAR_RESNET18_EVAL exp_id=40c2862ddf934b42835f8fc21089e5be

# leplus
python3 evaluation.py -m +experiment=evaluate quantization.activation_precision=2,3,4,5,6 exp_id=99c9ed56ad094470a2f759be807495af
python3 evaluation.py -m +experiment=evaluate  quantization.weight_precision=2,3,4,5,6,7,8 exp_id=99c9ed56ad094470a2f759be807495af