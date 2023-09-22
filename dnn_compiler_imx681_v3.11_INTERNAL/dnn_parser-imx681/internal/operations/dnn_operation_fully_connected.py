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
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.ActivationFunctionType import ActivationFunctionType

import torch

class DNNOperationFullyConnected:
  """
  Operation-specific tasks for FULLY_CONNECTED operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    parameters = dict()

    # Initialize clip_max based on the activation type
    options = FullyConnectedOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    if (options.FusedActivationFunction() == ActivationFunctionType.NONE):
      parameters["relu_clip_max"] = RELU_NONE
    elif (options.FusedActivationFunction() == ActivationFunctionType.RELU):
      parameters["relu_clip_max"] = RELU_NO_MAX_CLIP
    elif (options.FusedActivationFunction() == ActivationFunctionType.RELU6):
      parameters["relu_clip_max"] = 6
    else:
      err = "Unsupported activation function: %s." % (
        flatbuf_enum_to_str(option.FusedActivationFunction(), ActivationFunctionType))
      err += "Supported values are [NONE, RELU, RELU6]."
      raise RuntimeError(err)

    # Check for unsupported parameters
    if options.KeepNumDims():
      raise RuntimeError("Unsupported parameter: KeepNumDims=True. Only supported value is False.")
    if options.WeightsFormat() != 0:
      raise RuntimeError("Unsupported parameter: WeightsFormat=%d. Only supported value is 0.")
    if options.AsymmetricQuantizeInputs():
      raise RuntimeError("Unsupported parameter: AsymmetricQuantizeInputs=True. Only supported value is False.")

    op_dict["parameters"] = parameters

  def get_op_params_pytorch(self, op_dict, module, node):
    if node.kind() == "quantized::linear_relu":
      op_dict["parameters"]["relu_clip_max"] = RELU_NO_MAX_CLIP
    else:
      op_dict["parameters"]["relu_clip_max"] = RELU_NONE

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    quant_start_idx = out_buf["quant_start_idx"]
  
    quant_scale = dnn["quant_params"][quant_start_idx][0]
    quant_zero  = dnn["quant_params"][quant_start_idx][1]

    # Quantize the relu max
    relu_clip_max = op_dict["parameters"]["relu_clip_max"]
    if (relu_clip_max != RELU_NONE) and (relu_clip_max != RELU_NO_MAX_CLIP ):
        relu_clip_max = min(127, max(-128, int(relu_clip_max / quant_scale + quant_zero)))
        op_dict["parameters"]["relu_clip_max"] = relu_clip_max
    self.__calc_scale_ratios(op_dict, dnn)
    self.__transpose_filter(dnn, op_dict)

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    # Modify bias values to include other constants
    self.__update_bias_terms(op_dict, dnn, endianness)
    # Write other parameters to misc data
    params = op_dict["parameters"]
    bytes = []
    bytes.extend(val_to_uint8_array(params["relu_clip_max"], 4, endianness))
    for sr in params["scale_ratios"]:
      fp = float_to_fp_uint32(sr, DNN_QUANT_SCALE_Q_FORMAT)
      bytes.extend(val_to_uint8_array(fp, 4, endianness))
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    return "clip_max=%d, scale_ratios=%s" % (
      params["relu_clip_max"], ", ".join(["%.4f" % sr for sr in params["scale_ratios"]]))

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    # This operation requires two temporary int32 buffers that are the same size
    # as the output
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    out_size = out_buf["dimensions"][DIMS_NUM_BATCHES] * out_buf["data_num_cols"] * out_buf["data_num_rows"]
    mem_size = 2 * 4 * out_size
    return mem_size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    filt_buf_idx = op_dict["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    in_features = in_buf["data_num_cols"]
    out_size = out_buf["data_num_cols"] * out_buf["data_num_rows"]

    cnt = out_buf["dimensions"][DIMS_NUM_BATCHES]
    cycles = 0
    cycles += record_instruction("VMML_C", cnt, out_size*in_features, instr_cnts, instr_cycles)
    cycles += record_instruction("VADD_L", cnt, out_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMML_C (Scalar 1)", cnt, out_size*in_features, instr_cnts, instr_cycles)
    if filt_buf["quant_type"] == QuantType.PER_TENSOR:
      cycles += record_instruction("VMADD_L (Scalar 1)", cnt, out_size, instr_cnts, instr_cycles)
    else:
      cycles += record_instruction("VMSUBR_L", cnt, out_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMADDC_L (Scalar 1,2)", cnt, out_size, instr_cnts, instr_cycles)
    if op_dict["parameters"]["relu_clip_max"] == RELU_NO_MAX_CLIP:
      cycles += record_instruction("VMAX_C", cnt, out_size, instr_cnts, instr_cycles)
    elif op_dict["parameters"]["relu_clip_max"] != RELU_NONE:
      cycles += record_instruction("VCLAMP_C", cnt, out_size, instr_cnts, instr_cycles)
    return 1000*VPU_CLOCK_PERIOD_S*cycles
  
  def check_for_patches(self, dnn, op_dict):
    return []

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass
  
  # ============================================================================
  # Private methods
  # ============================================================================

  def __calc_scale_ratios(self, op, dnn):
    """
    Calculate the value of the quantization scale ratio: Sa*Sb/Sc

    Parameters:
    op - the current operation's dictionary (it will be modified)
    dnn - the dnn dictionary (its misc_data will be modified)
    endianness - "big" or "little"
    """

    # Get input & output buffers
    in_buf_idx = op["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_quant_start_idx = in_buf["quant_start_idx"]

    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    filt_quant_start_idx = filt_buf["quant_start_idx"]

    out_buf_idx = op["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    out_quant_start_idx = out_buf["quant_start_idx"]
    op["parameters"]["scale_ratios"] = []
    if filt_buf["quant_type"] == QuantType.PER_TENSOR:
      out_features = 1
    else:
      out_features = out_buf["dimensions"][DIMS_NUM_COLS]
    for out in range(0, out_features):
      sa = dnn["quant_params"][in_quant_start_idx][0]
      sb = dnn["quant_params"][filt_quant_start_idx + out][0]
      sc = dnn["quant_params"][out_quant_start_idx][0]
      scale_ratio = sa * sb / sc
      op["parameters"]["scale_ratios"].append(scale_ratio)

  def __update_bias_terms(self, op, dnn, endianness):
    """
    Update bias terms for convolution to equal all constant values:
    Za*Zb*N-Za*sum(qb)+bias

    Parameters:
    op - the current operation's dictionary (it will be modified)
    dnn - the dnn dictionary (its misc_data will be modified)
    endianness - "big" or "little"
    """
    # Get input buffers
    in_buf_idx = op["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_quant_start_idx = in_buf["quant_start_idx"]

    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    filt_quant_start_idx = filt_buf["quant_start_idx"]

    out_buf_idx = op["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    
    bias_buf_idx = op["inputs"][2]
    bias_buf = dnn["buffers"][bias_buf_idx]

    in_features = in_buf["data_num_cols"]
    out_features = out_buf["data_num_cols"]

    for out in range(0, out_features):
      # Get this output channel's int32 bias term from static data
      bias_data_idx = bias_buf["start_idx"] + out * 4
      bias = np.int32(uint8_array_to_val(dnn["static_data"][bias_data_idx:(bias_data_idx+4)], endianness))
      # Get quantization parameters
      if filt_buf["quant_type"] == QuantType.PER_TENSOR:
        filt_quant_idx = filt_quant_start_idx
      else:
        filt_quant_idx = filt_quant_start_idx + out
      za = dnn["quant_params"][in_quant_start_idx][1]
      zb = dnn["quant_params"][filt_quant_idx][1]
      # Calculate sum(qb) for this output channel
      sum_qb = 0
      result = 0
      in_data = [-7,-13,-15,-7,7,-6,-8,-32,-11,-7,3,-12,-7,-7,-14,5,-7,1,-8,-7,-7,-13,-7,-8,-2,-7,11,-4,-7,-11,-10,-7,-14,-8,-9,-23,-7,-7,-4,-12,26,-5,-20,-20,-9,-19,-7,-34,-7,-1,-12,3,-7,-7,-7,-7,-13,-7,-6,-7,1,4,-7,-6]
      for i in range(0, in_features):
        filt_data_idx = filt_buf["start_idx"] + i * out_features + out
        sum_qb = sum_qb + np.int32(np.int8(dnn["static_data"][filt_data_idx]))
        in_data_idx = i % 64
        result += np.int32(np.int8(dnn["static_data"][filt_data_idx])) * np.int32(np.int8(in_data[in_data_idx]))

      # Calculate the full constant
      bias += za*zb*in_features - za*sum_qb
      # Write this value back to static data
      bias_bytes = val_to_uint8_array(bias, 4, endianness)
      for i in range(0, len(bias_bytes)):
        dnn["static_data"][bias_data_idx+i] = bias_bytes[i]    


  def __transpose_filter(self, dnn, op):
    """
    Transpose a filter's data in the static data array. If possible,
    this is done in place. If not, a new buffer is created.

    Parameters:
    dnn - the dnn dictionary
    op - the current operation's dictionary
    """

    # Get filter buffer
    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]

    filt_rows = filt_buf["dimensions"][DIMS_NUM_ROWS]
    filt_cols = filt_buf["dimensions"][DIMS_NUM_COLS]

    # Get the data
    start_idx = filt_buf["start_idx"]
    data_len = filt_rows * filt_cols
    in_data = dnn["static_data"][start_idx:(start_idx+data_len)]
    out_data = np.empty([data_len], dtype=np.uint8)

    # Transpose the data
    for i in range(0, filt_rows):
      for j in range(0, filt_cols):
        out_data[j*filt_rows + i] = in_data[i*filt_cols + j]

   # Save new data to static data array in place
    for i in range(0, data_len):
      dnn["static_data"][start_idx+i] = out_data[i]

    # Transpose buffer's dimensions
    filt_buf["dimensions"][DIMS_NUM_ROWS] = filt_cols
    filt_buf["dimensions"][DIMS_NUM_COLS] = filt_rows
    filt_buf["data_num_rows"] = filt_cols
    filt_buf["data_num_cols"] = filt_rows
