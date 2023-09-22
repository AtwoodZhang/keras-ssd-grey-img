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


class DNNOperationReshape:
  """
  Operation-specific tasks for RESHAPE operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    op_dict["parameters"] = dict()
    op_dict["parameters"]["source"] = "tflite"

  def get_op_params_pytorch(self, op_dict, module, node):
    op_dict["parameters"] = dict()
    op_dict["parameters"]["source"] = "pytorch"

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]

    source = op_dict["parameters"]["source"]
    if (source == "pytorch") and (mem_order == MemoryOrder.CHANNEL_FIRST):
      transpose = True
    elif (source == "tflite") and (mem_order == MemoryOrder .CHANNEL_LAST):
      transpose = True
    else:
      transpose = False
    op_dict["parameters"]["transpose"] = transpose

    # If the output buffer has padding, then the operation can no longer
    # be done in place.
    if (
        out_buf["dimensions"][DIMS_NUM_ROWS] != out_buf["data_num_rows"]) or (
        out_buf["dimensions"][DIMS_NUM_COLS] != out_buf["data_num_cols"]):
      unpadded = False
    else:
      unpadded = True
    op_dict["parameters"]["unpadded"] = unpadded

    if (not unpadded) or transpose:
      # Undo making this buffer a parent of the input buffer, since it can no
      # longer share the same memory location
      dnn["buffers"][out_buf["parent"]]["num_connections"] -= out_buf["num_connections"]
      out_buf["parent"] = -1
    

  def op_params_to_byte_array(self, op_dict, misc_data, endianness):
    bytes = []
    # ES1 and ES2 FW expects 16-bit values and can not be changed.
    # This originally set values in uint32_t: 
    #   val =(transpose? 1: 0) << 16
    #   val =(unpadded? 1: 0) << 16

    # That worked for big endian FW as transpose got picked up first, it
    # got it reversed for little endian.

    # This needs to add endian appropriate values, in order:
    #  * 16-bit transpose 0 or 1 
    #  * 16-bit unpadded 0 or 1 

    bytes.extend(val_to_uint8_array(op_dict["parameters"]["transpose"], 2, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["unpadded"], 2, endianness))
    return uint8_array_to_val(bytes, endianness)


  def op_params_to_string(self, dnn, op_dict):
    string = "unpadded=" + str(op_dict["parameters"]["unpadded"])
    string += ", transpose=" + str(op_dict["parameters"]["transpose"]) 
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    working_memory_size = 0
    # In order to transpose, one int8 buffer the same size as a batch of the
    # input is required
    if op_dict["parameters"]["transpose"]:
      in_buf_idx = op_dict["inputs"][0]
      in_buf = dnn["buffers"][in_buf_idx]
      working_memory_size = in_buf["dimensions"][DIMS_NUM_ROWS]
      working_memory_size *= in_buf["dimensions"][DIMS_NUM_COLS]
      working_memory_size *= in_buf["dimensions"][DIMS_NUM_CHANNELS]
      working_memory_size *= in_buf["dimensions"][DIMS_NUM_BATCHES]
      if not op_dict["parameters"]["unpadded"]:
        working_memory_size *= 2
    return working_memory_size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    cycles = 0
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    if op_dict["parameters"]["transpose"]:
      size = in_buf["data_num_cols"] * in_buf["data_num_rows"]
      size *= in_buf["dimensions"][DIMS_NUM_CHANNELS]
      cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]
      cycles += record_instruction("VMT_C", cnt, size, instr_cnts, instr_cycles)

      size = out_buf["data_num_cols"] * out_buf["data_num_rows"]
      size *= out_buf["dimensions"][DIMS_NUM_CHANNELS]
      cnt = out_buf["dimensions"][DIMS_NUM_BATCHES]
      cycles += record_instruction("VMT_C", cnt, size, instr_cnts, instr_cycles)

    elif not op_dict["parameters"]["unpadded"]:
      cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]
      size = in_buf["data_num_cols"] * in_buf["data_num_rows"]
      size *= in_buf["dimensions"][DIMS_NUM_CHANNELS]
      cycles += record_instruction("VMOV_C (Scalar 0)", cnt, size, instr_cnts, instr_cycles)
    return 1000*VPU_CLOCK_PERIOD_S*cycles

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      if op_dict["parameters"]["transpose"]:
        patches.append("170940_177827")
    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
