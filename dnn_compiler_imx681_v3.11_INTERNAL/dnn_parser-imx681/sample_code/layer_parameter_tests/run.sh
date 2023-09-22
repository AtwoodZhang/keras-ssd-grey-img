#!/bin/bash

config_path="./configs/layer_parameter_test_"$1".yaml"
echo $config_path

CUDA_VISIBLE_DEVICES=0,1 python3 train.py $config_path