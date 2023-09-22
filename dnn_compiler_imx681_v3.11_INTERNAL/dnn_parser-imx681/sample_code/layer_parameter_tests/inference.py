import os
import sys
import torch
import cv2
import numpy as np
# #from layer_test_models.model_0 import LayerTest, prepare_model
# #from layer_test_models.model_1 import LayerTest
# #from layer_test_models.model_2 import LayerTest
# #from layer_test_models.model_3 import LayerTest
# from layer_test_models.model_4 import LayerTest
# #from layer_test_models.model_6 import LayerTest
import pdb
from helpers import load_config, prepare_model, model_import
import argparse

# sys.path.append("../..")
# import dnn_compiler
def nchw_to_nhwc(nchw):
    return nchw.permute((0,2,3,1))

def nchw_to_hwcn(nchw):
    return nchw.permute((2,3,1,0))

def print_tensor(tensor, file=None):
    arr = tensor.numpy()
    if file is None:
        for x in arr.flatten():
            print(x)
    else:
        for x in arr.flatten():
            print(x, file=file)



def test(config):
    # pdb.set_trace()

    test_name = config["name"]
    test_config = config["testing"]
    model_file = os.path.join(os.path.join(test_config["model_file"], test_name), "model_quantized.pth")
    # config_file = os.path.join("./configs/", "{}.yaml".format(test_name))
    num_classes = 10
    #test_image = './test_0.pgm'
    #test_image = './test_all_1.pgm'
    test_image = test_config["test_image"]
    INPUT_SIZE = (160, 120) # width, height

        
    # load state_dict
    state_dict = torch.load(model_file)

    # create model
    # config = load_config(config)
    # model = LayerTest(config, debug=True)
    model = model_import(config['name'], config, debug=True)

    # set up model
    model = prepare_model(model)
    model = torch.quantization.convert(model)
    model.load_state_dict(state_dict)
    model.eval()


    # debug folder
    debug_dir = os.path.join(test_config["debug_folder"], test_name)
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    with open(os.path.join(debug_dir, 'state_dict.txt'), 'w') as f:
        print(model.state_dict(), file=f)
    with open(os.path.join(debug_dir, 'model_structure.txt'), 'w') as f:
        print(model, file=f)


    # Load data
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.reshape((1,1,INPUT_SIZE[1],INPUT_SIZE[0]))
    img = img.astype(np.int)
    img = torch.Tensor(img)
    image = (img - model.quant.zero_point) * model.quant.scale
    # pdb.set_trace()
    with torch.no_grad():
        confidence, activation = model(image)
        print(confidence * 100)

    # debug
    torch.set_printoptions(profile="full")
    for name in activation.keys():
        tensor = activation[name].detach().clone()
        if tensor.dtype in [ torch.quint8, torch.qint8 ]:
            # pdb.set_trace()
            # with open(os.path.join(debug_dir, 'quant.{}.{}.txt'.format(name,tensor.dtype)), 'w') as f:
            quant_debug_dir = os.path.join(debug_dir, "quant")
            float_debug_dir = os.path.join(debug_dir, "float")
            other_debug_dir = os.path.join(debug_dir, "other")

            if not os.path.exists(quant_debug_dir):
                os.makedirs(quant_debug_dir)
            if not os.path.exists(float_debug_dir):
                os.makedirs(float_debug_dir)
            if not os.path.exists(other_debug_dir):
                os.makedirs(other_debug_dir)

            # pdb.set_trace()
            with open(os.path.join(quant_debug_dir, '{}.txt'.format(name)), 'w') as f:
                out = tensor.int_repr().int()
                if tensor.dtype == torch.quint8:
                    out = out - torch.tensor([128])
                if tensor.ndim == 4:
                    out = nchw_to_nhwc(out)
                else:
                    print('{} is not 4d, skipping nchw_to_nhwc'.format(name))
                print_tensor(out, file=f)
            with open(os.path.join(float_debug_dir, 'float.{}.{}.txt'.format(name,tensor.dtype)), 'w') as f:
                out = torch.dequantize(tensor)
                if tensor.ndim == 4:
                    out = nchw_to_nhwc(out)
                else:
                    print('{} is not 4d, skipping nchw_to_nhwc'.format(name))
                print_tensor(out, file=f)
        else:
            with open(os.path.join(other_debug_dir, '{}.txt'.format(name)), 'w') as f:
                if tensor.ndim == 4:
                    out = nchw_to_nhwc(tensor)
                print_tensor(out, file=f)
            with open(os.path.join(quant_debug_dir, '{}.txt'.format(name)), 'w') as f:
                out = torch.round((tensor * 256.0) - 128.0).int()
                if tensor.ndim == 4:
                    out = nchw_to_nhwc(out)
                else:
                    print('{} is not 4d, skipping nchw_to_nhwc'.format(name))
                print_tensor(out, file=f)

if __name__ == "__main__":

    layer_parser = argparse.ArgumentParser(description='Inference - Layer Parameter Tests')
    layer_parser.add_argument('config', type=str, help='the path to config.yaml file')
    args = layer_parser.parse_args()
    
    
    config = load_config(args.config)
    # use_cuda = config["training"]["use_cuda"]
    # device = config["training"]["devices"]
  

    # DEVICE = torch.device("cuda:"+str(device) if torch.cuda.is_available() and use_cuda else "cpu")


    test(config)
