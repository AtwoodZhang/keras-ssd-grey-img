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
from tflite.SoftmaxOptions import SoftmaxOptions

from internal.constants import *
from internal.utils import *


class DNNOperationSoftmax:
  """
  Operation-specific tasks for SOFTMAX operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    # Axis will be filled in later
    op_dict["parameters"]["axis"] = -1
    options = SoftmaxOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    if options.Beta() != 1.0:
      raise RuntimeError("Unsupported parameter: Beta=%.4f. Only supported value is 1.0" % (
        options.Beta()))

  def get_op_params_pytorch(self, op_dict, module, node):
    inputs = list(node.inputs())
    dim = inputs[1].toIValue()
    op_dict["parameters"]["axis"] = dim

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    # If the axis is still set to its default value, make it the last dimension
    # with a value greater than 1
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    if op_dict["parameters"]["axis"] == -1:
      op_dict["parameters"]["axis"] = DNN_BUFFER_MAX_DIMENSIONS-1
      for i in range(DNN_BUFFER_MAX_DIMENSIONS-1, -1, -1):
        if in_buf["dimensions"][i] > 1:
          op_dict["parameters"]["axis"] = i
          break
    else:
      # If any dimensions were re-arranged or prepended by the parser, update 
      # the axis field accordingly
      if len(in_buf["dim_reorder"]) > 0:
        axis = op_dict["parameters"]["axis"]
        axis = in_buf["dim_reorder"][axis]
        op_dict["parameters"]["axis"] = axis

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    return np.uint32(op_dict["parameters"]["axis"])

  def op_params_to_string(self, dnn, op_dict):
    return "axis=%d" % op_dict["parameters"]["axis"]

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    rows = in_buf["dimensions"][DIMS_NUM_ROWS]
    cols = in_buf["dimensions"][DIMS_NUM_COLS]
    chan = in_buf["dimensions"][DIMS_NUM_CHANNELS]
    axis = op_dict["parameters"]["axis"]
    if axis == 1:
      # int32 matrix that is the full input size
      mem_size = 4 * rows * cols * chan
      # 3x int32 row vectors 1 x (W*C)
      mem_size += 3 * 4 * cols * chan
    elif axis == 2:
      # int32 matrix that is (H*W)
      mem_size = 4 * rows * cols
      # 3x int32 col vectors H x 1
      mem_size += 3 * 4 * rows
    else:
      # int32 matrix that is the full input size
      mem_size = 4 * rows * cols * chan
      # 3x int32 col vectors (H*W) x 1
      mem_size += 3 * 4 * rows * cols

    return int(mem_size)


  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    batch = in_buf["dimensions"][DIMS_NUM_BATCHES]
    rows = in_buf["dimensions"][DIMS_NUM_ROWS]
    cols = in_buf["dimensions"][DIMS_NUM_COLS]
    chan = in_buf["dimensions"][DIMS_NUM_CHANNELS]
    axis = op_dict["parameters"]["axis"]
    if axis == 1:
      mat_size = rows*cols*chan
      vec_size = cols*chan
      cnt = batch
    elif axis == 2:
      mat_size = rows*cols
      vec_size = rows*1
      cnt = batch*chan
    else:
      mat_size = rows*cols*chan
      vec_size = rows*cols
      cnt = batch

    cycles = 0
    cycles += record_instruction("VSMULL_C (Scalar 1,2)", cnt, mat_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VCLAMP_L", cnt, mat_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMUL_L (Scalar 1)", cnt, mat_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VEXP2_L", cnt, mat_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VCACC_L", cnt, mat_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMUL_L (Scalar 1)", cnt, vec_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VESTRECIP_L", cnt, vec_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VSHA_L (Scalar 1)", cnt, vec_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMSUBR_L (Scalar 2)", 3*cnt, vec_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMUL_L", 3*cnt, vec_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VMUL_L", cnt, mat_size, instr_cnts, instr_cycles)
    cycles += record_instruction("VADDC_L (Scalar 1)", cnt, mat_size, instr_cnts, instr_cycles)

    return 1000 * cycles * VPU_CLOCK_PERIOD_S

  def check_for_patches(self, dnn, op_dict):
    return []

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
