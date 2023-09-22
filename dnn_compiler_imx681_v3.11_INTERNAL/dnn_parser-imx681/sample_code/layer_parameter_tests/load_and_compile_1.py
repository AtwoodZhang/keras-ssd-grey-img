import sys
import torch
import pdb
from layer_test_models.model_1 import LayerTest
from helpers import load_config, prepare_model

sys.path.append("../..")
import dnn_compiler

test_name = "layer_parameter_test_1"

model_file = "./models/quantized_models/"+test_name+"/model_quantized.pth"
config_file = "./configs/"+test_name+".yaml"
num_classes = 10
input_size = (120, 160)

config = load_config(config_file)

# load state_dict
state_dict = torch.load(model_file)

# set up
model = LayerTest(config)
model = model.cpu()
model = prepare_model(model)
model = torch.quantization.convert(model)
model.load_state_dict(state_dict)


overrides=["ML_CONV_MAX_OUT_SIZE=0", "ML_CONV_NUM_PARTITIONS=0"]

if len(sys.argv) > 1 and sys.argv[1] == "LE":
    overrides.append("OUTPUT_ENDIANNESS=LITTLE")

# call DNN parser
dnn_compiler.run("../../configs/test/imx681_test_pytorch_sim.cfg", 
    model, config_overrides=overrides )
    