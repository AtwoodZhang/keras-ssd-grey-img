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
from tflite.Conv2DOptions import Conv2DOptions
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding

from internal.constants import *
from internal.utils import *


class DNNOperationTranspose:
  """
  Operation-specific tasks for TRANSPOSE operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    pass

  def get_op_params_pytorch(self, op_dict, module, node):
    pass

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    if (in_buf["dimensions"][DIMS_NUM_CHANNELS] > 1) or (in_buf["dimensions"][DIMS_NUM_BATCHES] > 1):
      raise RuntimeError("TRANSPOSE layer only supports 2D input data. Invalid dimensions: %s" % (
        "x".join([str(d) for d in in_buf["dimensions"]])))
    
  def op_params_to_byte_array(self, op_dict, misc_data, endianness):
    return 0

  def op_params_to_string(self, dnn, op_dict):
    return ""

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    return 0

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_size = in_buf["data_num_cols"]*in_buf["data_num_rows"]

    cycles = record_instruction("VMT_C", 1, in_size, instr_cnts, instr_cycles)    

    return 1000*cycles*VPU_CLOCK_PERIOD_S

  def check_for_patches(self, dnn, op_dict):
    return []

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
