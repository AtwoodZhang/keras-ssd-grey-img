# ------------------------------------------------------------------------------
# Copyright 2022 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------

"""
Apply post training static quantization to a FP32 classification model
using FX Graph Mode Quantization.

Example Usage:
    python quantize.py model_fp32.pth
"""

import argparse
import os
import sys
import logging
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from torch.quantization import QConfig, HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from model import SampleModel

sys.path.append("../../")
import dnn_compiler

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Sample Model - FX Graph Mode Quantization')
    parser.add_argument('pth_file', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    return args


# Custom INT8 quantization configuration (per-tensor)
def quant_config_per_tensor():
    qconfig = QConfig(activation=HistogramObserver.with_args(
                        quant_min=0,
                        quant_max=255),
                      weight=MinMaxObserver.with_args(
                        quant_min=-128,
                        quant_max=127,
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric))
    return qconfig
   
   
def main():
    # Command line arguments
    args = get_args()

    # Calibration Dataset
    val_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.Grayscale(),
        transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0))
        ])
    val_data = datasets.CIFAR10("./cifar10/test/", train=False, transform=val_transform, download=True)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size = args.batch_size,
        shuffle = False)

    # Create model
    model = SampleModel(
        num_classes=args.num_classes,
        input_size=(args.height, args.width),
        mode='eval')
    model.init()
    logging.info("Model created")
  
    # Load model state_dict from file
    state_dict = torch.load(args.pth_file)
    model.load_state_dict(state_dict)
    
    # Quantize model with FX Graph Mode Quantization
    model.eval()
    qconfig = quant_config_per_tensor()
    qconfig_dict = {"": qconfig}            
    prepared_model = prepare_fx(model, qconfig_dict)  # fuse modules and insert observers

    # Calibrate quantization parameters by running representative (validation) dataset through model.
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                image = (image - 128) * 0.0039
                model(image)
    calibrate(prepared_model, val_loader)  # run calibration on sample data
    
    # Convert to quantized model
    quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

    # Save model
    savepath = os.path.join(os.getcwd(), 'model_quantized.pth')
    torch.save(quantized_model.state_dict(), savepath)
    logging.info("Model saved to {}".format(savepath))
    
    # Call Sony DNN Compiler to generate binary files.
    dnn_compiler.run("../../configs/imx681_pytorch_classification_fx_i2c.cfg", quantized_model, config_overrides=["INPUT_SCALE=0.0039","INPUT_ZEROPOINT=128"])

if __name__ == "__main__":
    main()


