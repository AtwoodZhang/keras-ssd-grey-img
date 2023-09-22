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


from internal.constants import *
from internal.utils import *


class DNNOperationRelu:
  """
  Operation-specific tasks for RELU operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    if opcode == BuiltinOperator.RELU6:
      # clip_max will be calculated later once we have quantization parameters
      # available
      op_dict["parameters"]["clip_max"] = 6
    elif opcode == BuiltinOperator.RELU:
      op_dict["parameters"]["clip_max"] = RELU_NO_MAX_CLIP
    else:
      raise RuntimeError("Unexpected BuiltinCode on RELU operation")

  def get_op_params_pytorch(self, op_dict, module, node):
    op_dict["parameters"]["clip_max"] = RELU_NO_MAX_CLIP

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    quant_start_idx = out_buf["quant_start_idx"]
  
    quant_scale = dnn["quant_params"][quant_start_idx][0]
    quant_zero  = dnn["quant_params"][quant_start_idx][1]

    # Quantize the clip max
    relu_clip_max = op_dict["parameters"]["clip_max"]
    if (relu_clip_max != RELU_NONE) and (relu_clip_max != RELU_NO_MAX_CLIP ):
        relu_clip_max = min(127, max(-128, int(round(relu_clip_max / quant_scale + quant_zero))))
        op_dict["parameters"]["clip_max"] = relu_clip_max 

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    return op_dict["parameters"]["clip_max"]

  def op_params_to_string(self, dnn, op_dict):
    clip_max = op_dict["parameters"]["clip_max"]
    if clip_max == RELU_NONE:
      clip_max_string = "NO_RELU"
    elif clip_max == RELU_NO_MAX_CLIP:
      clip_max_string = "NO_MAX"
    else:
      clip_max_string = str(clip_max)
    return "clip_max=%s" % clip_max_string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    return 0

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    out_size = out_buf["data_num_cols"]*out_buf["data_num_rows"]
    out_size *= out_buf["dimensions"][DIMS_NUM_CHANNELS] 
    out_size *= out_buf["dimensions"][DIMS_NUM_BATCHES]

    cycles = 0
    if op_dict["parameters"]["clip_max"] == RELU_NO_MAX_CLIP:
      cycles += record_instruction("VMAX_C", 1, out_size, instr_cnts, instr_cycles)
    elif op_dict["parameters"]["clip_max"] != RELU_NONE:
      cycles += record_instruction("VCLAMP_C", 1, out_size, instr_cnts, instr_cycles)

    return 1000*VPU_CLOCK_PERIOD_S*cycles

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      in_buf = dnn["buffers"][op_dict["inputs"][0]]
      out_buf = dnn["buffers"][op_dict["outputs"][0]]
      # If this operation has multiple batches AND there is vertical padding on
      # either the input or output buffer, a patch is necessary
      if (in_buf["dimensions"][DIMS_NUM_BATCHES] > 1):
        if (in_buf["dimensions"][DIMS_NUM_ROWS] != in_buf["data_num_rows"]) or (
          out_buf["dimensions"][DIMS_NUM_ROWS] != out_buf["data_num_rows"]):
          patches.append("170940_177827")

    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
