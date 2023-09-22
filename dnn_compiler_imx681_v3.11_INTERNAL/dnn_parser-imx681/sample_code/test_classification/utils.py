# ------------------------------------------------------------------------------
# Copyright 2022 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------

import torch
from torch.quantization import QConfig, FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

# Custom INT8 quantization configuration (per-tensor)
def quant_config_per_tensor():
    qconfig = QConfig(activation=FakeQuantize.with_args(
                        observer=MovingAverageMinMaxObserver,
                        quant_min=0,
                        quant_max=255,
                        reduce_range=True),
                      weight=FakeQuantize.with_args(
                        observer=MovingAverageMinMaxObserver,
                        quant_min=-128,
                        quant_max=127,
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric,
                        reduce_range=True))
    return qconfig

# Custom INT8 quantization configuration (per-channel)
def quant_config_per_channel():
    qconfig = QConfig(activation=FakeQuantize.with_args(
                        observer=MovingAverageMinMaxObserver,
                        quant_min=0,
                        quant_max=255,
                        reduce_range=True),
                      weight=FakeQuantize.with_args(
                        observer=MovingAveragePerChannelMinMaxObserver,
                        quant_min=-128,
                        quant_max=127,
                        dtype=torch.qint8,
                        qscheme=torch.per_channel_symmetric,
                        reduce_range=True,
                        ch_axis=0))
    return qconfig

# Prepare a model for quantization-aware training by fusing conv/bn/relu layers.
def prepare_model(model):
    # set up model in training mode
    model.train()
    # set up all layers with per-channel quantization
    model.qconfig = quant_config_per_channel()
    # change any torch.nn.Linear layers to use per-tensor quantization
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.qconfig = quant_config_per_tensor()
    model.eval()
    # fuse conv/bn/relu together
    model = torch.quantization.fuse_modules(model, model.layers_to_fuse)
    model.train()
    # convert fp32 model to quantize-aware training model
    model = torch.quantization.prepare_qat(model)
    return model
 
