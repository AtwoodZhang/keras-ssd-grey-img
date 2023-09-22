import os
import logging
import sys
import itertools
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
from model.sony_mobilenet_ssdlite import create_sony_mobilenet_ssdlite
from model.config import sony_mobilenet_ssd_config
from train import prepare_model

sys.path.append("../../")
import dnn_compiler

model_file = './model_quantized.pth'
label_file = './saved_models/face-hand-person-model-labels.txt'
    
# load state_dict
state_dict = torch.load(model_file)

# object classes
class_names = tuple([name.strip() for name in open(label_file).readlines()])

# set up
config = sony_mobilenet_ssd_config
num_classes = len(class_names)
model_cpu = create_sony_mobilenet_ssdlite(num_classes, quantize=True)
model_cpu = prepare_model(model_cpu)
model_cpu = torch.quantization.convert(model_cpu)
model_cpu.load_state_dict(state_dict)

overrides=[]

if len(sys.argv) > 1 and sys.argv[1] == "LE":
    overrides.append("OUTPUT_ENDIANNESS=LITTLE")

dnn_compiler.run("../../configs/test/imx681_test_pytorch_detection_sim.cfg",
    model_cpu, config_overrides=overrides )
