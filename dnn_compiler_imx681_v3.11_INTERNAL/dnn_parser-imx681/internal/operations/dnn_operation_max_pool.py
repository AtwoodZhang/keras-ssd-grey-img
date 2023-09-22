# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import math
import numpy as np

from internal.constants import *
from internal.utils import *


from tflite.BuiltinOperator import BuiltinOperator
from tflite.Pool2DOptions import Pool2DOptions
from tflite.Padding import Padding

# Minimum and maximum values of parameters
if DNN_COMPILER_INTERNAL_USE:
  PARAMETER_STRIDE_MIN = 1
  PARAMETER_STRIDE_MAX = 15
  PARAMETER_FILT_SIZE_MIN = 1
  PARAMETER_FILT_SIZE_MAX = 15
else:
  PARAMETER_STRIDE_MIN = 1
  PARAMETER_STRIDE_MAX = 6
  PARAMETER_FILT_SIZE_MIN = 2
  PARAMETER_FILT_SIZE_MAX = 3

PARAMETER_PADDING_MIN = 0
PARAMETER_PADDING_MAX = 15


class DNNOperationMaxPool:
  """
  Operation-specific tasks for MAX_POOL operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    parameters = dict()
    options = Pool2DOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    filter_w = options.FilterWidth()
    filter_h = options.FilterHeight()
    stride_w = options.StrideW()
    stride_h = options.StrideH()

    parameters["padding_type"] = options.Padding()
    parameters["stride_w"] = stride_w
    parameters["stride_h"] = stride_h
    parameters["filter_w"] = filter_w
    parameters["filter_h"] = filter_h
    parameters["source"] = "tflite"
    op_dict["parameters"] = parameters

  def get_op_params_pytorch(self, op_dict, module, node):
    parameters = dict()

    if module.stride.__class__ == tuple:
      parameters["stride_w"] = module.stride[0]
      parameters["stride_h"] = module.stride[1]
    else:
      parameters["stride_w"] = module.stride
      parameters["stride_h"] = module.stride
    
    if module.padding.__class__ == tuple:
      if (module.padding[0] == 0) and (module.padding[1]) == 0:
        parameters["padding_type"] = Padding.VALID
      else:
        parameters["padding_type"] = Padding.SAME
    else:
      if module.padding == 0:
        parameters["padding_type"] = Padding.VALID
      else:
        parameters["padding_type"] = Padding.SAME
    
    if module.kernel_size.__class__ == tuple:
      parameters["filter_w"] = module.kernel_size[0]
      parameters["filter_h"] = module.kernel_size[1]
    else:
      parameters["filter_w"] = module.kernel_size
      parameters["filter_h"] = module.kernel_size

    # Check that invalid parameters are not provided
    if module.dilation != 1:
      raise RuntimeError("Unsupported parameter: dilation. Only supported value is 1." % (
        module.dilation))
    if module.ceil_mode:
      raise RuntimeError("Unsupported parameter: ceil_mode=True. Only supported value is False." % (
        module.ceil_mode))

    parameters["source"] = "pytorch"

    op_dict["parameters"] = parameters

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    parameters = op_dict["parameters"]

    check_param_range("pad_left", parameters["pad_left"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("pad_right", parameters["pad_right"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("pad_top", parameters["pad_top"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("pad_bottom", parameters["pad_bottom"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("stride_w", parameters["stride_w"], PARAMETER_STRIDE_MIN, PARAMETER_STRIDE_MAX)
    check_param_range("stride_h", parameters["stride_h"], PARAMETER_STRIDE_MIN, PARAMETER_STRIDE_MAX)
    check_param_range("filter_width", parameters["filter_w"], PARAMETER_FILT_SIZE_MIN, PARAMETER_FILT_SIZE_MAX)
    check_param_range("filter_height", parameters["filter_h"], PARAMETER_FILT_SIZE_MIN, PARAMETER_FILT_SIZE_MAX)
    if not DNN_COMPILER_INTERNAL_USE:
      if parameters["stride_w"] != parameters["stride_h"]:
        raise RuntimeError("stride_w (%d) and stride_h (%d) must be equal." % (
          parameters["stride_w"], parameters["stride_h"]))
      if parameters["filter_w"] != parameters["filter_h"]:
        raise RuntimeError("filter_width (%d) and filter_height (%d) must be equal." % (
          parameters["filter_w"], parameters["filter_h"]))


  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    if endianness == Endianness.BIG:
      byte0  = (op_dict["parameters"]["pad_left"] & 0xF) << 4
      byte0 |= (op_dict["parameters"]["pad_right"] & 0xF)
      
      byte1  = (op_dict["parameters"]["pad_top"] & 0xF) << 4
      byte1 |= (op_dict["parameters"]["pad_bottom"] & 0xF)

      byte2  = (op_dict["parameters"]["stride_w"] & 0xF) << 4
      byte2 |= (op_dict["parameters"]["stride_h"] & 0xF)

      byte3  = (op_dict["parameters"]["filter_w"] & 0xF) << 4
      byte3 |= (op_dict["parameters"]["filter_h"] & 0xF)
    else:
      byte0  = (op_dict["parameters"]["pad_left"] & 0xF)
      byte0 |= (op_dict["parameters"]["pad_right"] & 0xF) << 4
      
      byte1  = (op_dict["parameters"]["pad_top"] & 0xF)
      byte1 |= (op_dict["parameters"]["pad_bottom"] & 0xF) << 4

      byte2  = (op_dict["parameters"]["stride_w"] & 0xF)
      byte2 |= (op_dict["parameters"]["stride_h"] & 0xF) << 4

      byte3  = (op_dict["parameters"]["filter_w"] & 0xF)
      byte3 |= (op_dict["parameters"]["filter_h"] & 0xF) << 4     
    bytes = [byte0, byte1, byte2, byte3]
    return uint8_array_to_val(bytes, endianness)
    
  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "filt=%dx%d, stride=(W:%d, H:%d), padding=(L:%d, R:%d, T:%d, B:%d)" % (
      params["filter_w"], params["filter_h"],
      params["stride_w"], params["stride_h"], 
      params["pad_left"], params["pad_right"], 
      params["pad_top"], params["pad_bottom"])
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    return 0

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    out_size = out_buf["data_num_cols"]*out_buf["data_num_rows"]

    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_cols = in_buf["data_num_cols"] + op_dict["parameters"]["pad_left"] + op_dict["parameters"]["pad_right"]
    in_rows = in_buf["data_num_rows"] + op_dict["parameters"]["pad_top"] + op_dict["parameters"]["pad_bottom"]
    in_size = in_cols * in_rows
    in_channels = in_buf["dimensions"][DIMS_NUM_CHANNELS]

    cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]
    cycles = 0

    # VMOVs
    pad_left = op_dict["parameters"]["pad_left"] * in_channels
    pad_right = op_dict["parameters"]["pad_right"] * in_channels
    pad_top = op_dict["parameters"]["pad_top"]
    pad_bottom = op_dict["parameters"]["pad_bottom"]
    total_rows = in_buf["data_num_rows"] + pad_top + pad_right
    total_cols = in_buf["data_num_cols"] * in_channels+ pad_left + pad_right
    num_rows = total_rows - (pad_top + pad_bottom + 1)
    num_cols = pad_left + pad_right
    if pad_top > 0:
      cycles += record_instruction("VMOV_C (Scalar 0)", cnt, pad_top*total_cols, instr_cnts, instr_cycles)
    if pad_bottom > 0:
      cycles += record_instruction("VMOV_C (Scalar 0)", cnt, pad_bottom*total_cols, instr_cnts, instr_cycles)
    if pad_left > 0:
      cycles += record_instruction("VMOV_C (Scalar 0)", cnt, pad_left*total_rows, instr_cnts, instr_cycles)
    if pad_right > 0:
      cycles += record_instruction("VMOV_C (Scalar 0)", cnt, pad_right*total_rows, instr_cnts, instr_cycles)

    # VMAXPOOLs
    filt_size = op_dict["parameters"]["filter_w"] * op_dict["parameters"]["filter_w"]
    cycles += record_instruction("VMAXPOOL_C", cnt*in_channels, out_size*filt_size, instr_cnts, instr_cycles)

    return 1000*VPU_CLOCK_PERIOD_S*cycles

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      in_buf = dnn["buffers"][op_dict["inputs"][0]]

      if in_buf["dimensions"][DIMS_NUM_BATCHES] > 1:
        # Patch 170940 is necessary if there are multiple batches and
        # padding is required
        if (
          op_dict["parameters"]["pad_left"] > 0) or (
          op_dict["parameters"]["pad_right"] > 0) or (
          op_dict["parameters"]["pad_top"] > 0) or (
          op_dict["parameters"]["pad_bottom"] > 0):
            patches.append("170940_177827")

    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  def set_padding(self, dnn, op_dict):
      # Get the input image's size
      in_idx = op_dict["inputs"][CONV_INPUT_IMAGE]
      in_buf = dnn["buffers"][in_idx]
      in_width = in_buf["data_num_cols"]
      in_height = in_buf["data_num_rows"]

      # Get the filter's size
      filter_width = op_dict["parameters"]["filter_w"]
      filter_height = op_dict["parameters"]["filter_h"]

      if (op_dict["parameters"]["padding_type"] == Padding.SAME):
        stride_w = op_dict["parameters"]["stride_w"]
        stride_h = op_dict["parameters"]["stride_h"]
        # If the stride is wider than the filter, treat the stride as the filter
        # size for all padding calculations
        if stride_w > filter_width:
          filter_width = stride_w
        if stride_h > filter_height:
          filter_height = stride_h
        # Determine the total padding needed in each direction
        horizontal_padding = filter_width - stride_w
        vertical_padding = filter_height - stride_h
        # Add padding to ensure the output size is an integer
        out_width = in_width - filter_width + horizontal_padding
        if (out_width % stride_w) != 0:
          horizontal_padding += stride_w - (out_width % stride_w)
        out_height = in_height - filter_height + vertical_padding
        if (out_height % stride_h) != 0:
          vertical_padding += stride_h - (out_height % stride_h)

        # Determine padding on each side of the image
        pad_min = math.floor(horizontal_padding/2)
        pad_max = horizontal_padding - pad_min
        if op_dict["parameters"]["source"] == "tflite":
          op_dict["parameters"]["pad_left"] = pad_min
          op_dict["parameters"]["pad_right"] = pad_max
        else:
          op_dict["parameters"]["pad_left"] = pad_max
          op_dict["parameters"]["pad_right"] = pad_min

        pad_min = math.floor(vertical_padding/2)
        pad_max = vertical_padding - pad_min
        if op_dict["parameters"]["source"] == "tflite":
          op_dict["parameters"]["pad_top"] = pad_min
          op_dict["parameters"]["pad_bottom"] = pad_max
        else:
          op_dict["parameters"]["pad_top"] = pad_max
          op_dict["parameters"]["pad_bottom"] = pad_min

      elif (op_dict["parameters"]["padding_type"] == Padding.VALID):
        op_dict["parameters"]["pad_left"] = 0
        op_dict["parameters"]["pad_right"] = 0
        op_dict["parameters"]["pad_top"] = 0
        op_dict["parameters"]["pad_bottom"] = 0
      else:
        raise RuntimeError("Unrecognized padding: " + str(op_dict["parameters"]["padding_type"]))
