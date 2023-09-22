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

from tflite.BuiltinOperator import BuiltinOperator
from tflite.ConcatenationOptions import ConcatenationOptions

from internal.constants import *
from internal.utils import *

# Minimum, maximum, and default value of parameters
PARAMETER_AXIS_MIN = 1
PARAMETER_AXIS_MAX = 3
PARAMETER_AXIS_DEF = 1

class DNNOperationConcatenate:
  """
  Operation-specific tasks for CONCATENATE operations. See dnn_operation.py
  for details on what each function does.
  """
  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    if opcode == BuiltinOperator.CONCATENATION:
      options = ConcatenationOptions()
      options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
      op_dict["parameters"]["axis"] = options.Axis()
    else:
      # QUANTIZE operation, which is implemented as a 1-input row concatenation
      op_dict["parameters"]["axis"] = PARAMETER_AXIS_DEF

    op_dict["parameters"]["scale_ratios"] = []
    op_dict["parameters"]["offsets"] = []
    op_dict["parameters"]["input_format"] = "tflite"

  def get_op_params_pytorch(self, op_dict, module, node):
    # Quantization scale and zeropoint for output are given as separate inputs
    inputs = list(node.inputs())
    op_dict["quant_scale"] = inputs[2].toIValue()
    op_dict["quant_zero"] = inputs[3].toIValue()
    op_dict["parameters"]["axis"] = -1
    op_dict["parameters"]["scale_ratios"] = []
    op_dict["parameters"]["offsets"] = []
    op_dict["parameters"]["input_format"] = "pytorch"

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      op_dict["parameters"]["q_format"] = DNN_QUANT_SCALE_Q_FORMAT
    else:
      op_dict["parameters"]["q_format"] = DNN_QUANT_OFFSET_Q_FORMAT
    if op_dict["parameters"]["input_format"] == "tflite":
      # If any dimensions were rearranged by the parser, update the axis field
      # accordingly
      in_idx = op_dict["inputs"][0]
      in_buf = dnn["buffers"][in_idx]
      if len(in_buf["dim_reorder"]) > 0:
        axis = op_dict["parameters"]["axis"]
        axis = in_buf["dim_reorder"][axis]
        op_dict["parameters"]["axis"] = axis
  
      axis = op_dict["parameters"]["axis"]
      check_param_range("axis", axis, PARAMETER_AXIS_MIN, PARAMETER_AXIS_MAX)
    else:
      # Get input buffer data dimensions
      in_idx = op_dict["inputs"][0]
      in_buf = dnn["buffers"][in_idx]
      in_dims = [in_buf["dimensions"][DIMS_NUM_BATCHES], in_buf["data_num_rows"], 
                 in_buf["data_num_cols"], in_buf["dimensions"][DIMS_NUM_CHANNELS]]
      # Get output buffer data dimensions
      out_idx = op_dict["outputs"][0]
      out_buf = dnn["buffers"][out_idx]
      out_dims = [out_buf["dimensions"][DIMS_NUM_BATCHES], out_buf["data_num_rows"], 
                  out_buf["data_num_cols"], out_buf["dimensions"][DIMS_NUM_CHANNELS]]
      # Default to axis = channels
      axis = DIMS_NUM_CHANNELS
      # Find the dimension where the output size is larger than the input size, and
      # assume that is the axis
      for i in range(0, DNN_BUFFER_MAX_DIMENSIONS):
        if in_dims[i] != out_dims[i]:
          axis = i
          break
      op_dict["parameters"]["axis"] = axis

    out_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_idx]

    quant_start_idx = out_buf["quant_start_idx"]
    sb = dnn["quant_params"][quant_start_idx][0]
    zb = dnn["quant_params"][quant_start_idx][1]

    # Calculate scale ratios and offsets for each input
    for in_idx in op_dict["inputs"]:
      in_buf = dnn["buffers"][in_idx]

      quant_start_idx = in_buf["quant_start_idx"]
      sa = dnn["quant_params"][quant_start_idx][0]
      za = dnn["quant_params"][quant_start_idx][1]

      scale_ratio = sa/sb
      offset = zb - sa/sb*za

      op_dict["parameters"]["scale_ratios"].append(scale_ratio)
      op_dict["parameters"]["offsets"].append(offset)

 

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
   # Create byte array and add to misc data
    bytes = []
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["axis"], 4, endianness))
    for sr in op_dict["parameters"]["scale_ratios"]:
      fp = float_to_fp_uint32(sr, op_dict["parameters"]["q_format"])
      bytes.extend(val_to_uint8_array(fp, 4, endianness))
    for off in op_dict["parameters"]["offsets"]:
      fp = float_to_fp_uint32(off, op_dict["parameters"]["q_format"])
      bytes.extend(val_to_uint8_array(fp, 4, endianness))
    return add_to_misc_data(dnn["misc_data"], bytes)

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    return 0

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "axis=%d, scales=%s, offsets=%s" % (
      params["axis"],
      ", ".join(["%.4f" % scale for scale in params["scale_ratios"]]),
      ", ".join(["%.4f" % offset for offset in params["offsets"]]))
    return string

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    axis = op_dict["parameters"]["axis"]
    cycles = 0
    for in_idx in op_dict["inputs"]:
      in_buf = dnn["buffers"][in_idx]
      in_size = in_buf["data_num_cols"]*in_buf["data_num_rows"]
      if (axis == BufferAxis.ROW.value):
        in_size *= in_buf["dimensions"][DIMS_NUM_CHANNELS]
        cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]
      elif (axis == BufferAxis.COLUMN.value):
        in_size *= in_buf["dimensions"][DIMS_NUM_CHANNELS]
        cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]
      elif (axis == BufferAxis.CHANNEL.value):
        cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]*in_buf["dimensions"][DIMS_NUM_CHANNELS]
      else:
        raise RuntimeError("Unsupported axis for concatenate operation: %d" % axis)

      cycles += record_instruction("VMADD_C (Scalar 1,2)", cnt, in_size, instr_cnts, instr_cycles)


    return 1000 * cycles * VPU_CLOCK_PERIOD_S

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      # If the offset is too large, require patch that changes q format from s7.24 to s9.22
      for off in op_dict["parameters"]["offsets"]:
        if (off > DNN_QUANT_SCALE_MAX) or (off < DNN_QUANT_SCALE_MIN):
          patches.append("170940_177827")
          break
    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      # Patch stores parameters in s9.22 format instead of s7.24 format
      if "170940_177827" in patches:
        op_dict["parameters"]["q_format"] = DNN_QUANT_OFFSET_Q_FORMAT

  # ============================================================================
  # Private methods
  # ============================================================================
