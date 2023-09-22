# ------------------------------------------------------------------------------
# Copyright 2022 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------

import sys
import torch
from model import SampleModel
from utils import prepare_model

sys.path.append("../../")
import dnn_compiler

# Model must be set up to match how it was trained.
model_file = './model_quantized.pth'
num_classes = 10
input_size = (120, 160)
    
# Load state_dict for quantized model.
state_dict = torch.load(model_file)

# Create model and copy weights from state_dict.
model = SampleModel(num_classes, input_size=input_size, quantize=True, mode='test')
model = prepare_model(model)
model = torch.quantization.convert(model)
model.load_state_dict(state_dict)

overrides=[]

if len(sys.argv) > 1 and sys.argv[1] == "LE":
    overrides.append("OUTPUT_ENDIANNESS=LITTLE")

# Call Sony DNN Compiler to generate binary files.
dnn_compiler.run("../../configs/imx681_pytorch_classification_i2c.cfg", 
    model, config_overrides=overrides )
