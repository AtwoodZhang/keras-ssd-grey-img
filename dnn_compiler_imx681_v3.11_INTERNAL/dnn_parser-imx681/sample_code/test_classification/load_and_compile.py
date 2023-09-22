import sys
import torch
from model import ROMClassification
from utils import prepare_model

sys.path.append("../../")
import dnn_compiler

model_file = './model_quantized.pth'
num_classes = 10
input_size = (120, 160)
    
# load state_dict
state_dict = torch.load(model_file)

# set up
model = ROMClassification(num_classes, input_size=input_size, quantize=True, mode='inference')
model = prepare_model(model)
model = torch.quantization.convert(model)
model.load_state_dict(state_dict)

overrides=[]

if len(sys.argv) > 1 and sys.argv[1] == "LE":
    overrides.append("OUTPUT_ENDIANNESS=LITTLE")

# call DNN parser
dnn_compiler.run("../../configs/test/imx681_test_pytorch_classification_sim.cfg", 
    model, config_overrides=overrides)
