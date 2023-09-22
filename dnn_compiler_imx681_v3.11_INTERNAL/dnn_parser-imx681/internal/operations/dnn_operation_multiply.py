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

class DNNOperationMultiply:
  """
  Operation-specific tasks for MULTIPLY operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    pass

  def get_op_params_pytorch(self, op_dict, module, node):
    # Quantization scale and zeropoint for output are given as separate inputs
    inputs = list(node.inputs())
    if node.kind() == "quantized::mul":
      op_dict["quant_scale"] = inputs[2].toIValue()
      op_dict["quant_zero"] = inputs[3].toIValue()

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):

    # Get input and output buffers
    in0_buf_idx = op_dict["inputs"][0]
    in0_buf = dnn["buffers"][in0_buf_idx]
    in1_buf_idx = op_dict["inputs"][1]
    in1_buf = dnn["buffers"][in1_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]

    # Determine the mode based on sizes of buffers. If there is enough memory
    # to flatten the buffers, they will be flatted later (in get_working_mem_size)
    in1_buf_size = in1_buf["data_num_cols"]*in1_buf["data_num_rows"]
    if in1_buf_size == 1:
      mode = ArithmeticMode.SCALAR
    else:
      mode = ArithmeticMode.MATRIX

    # Calculate scale ratios and offset
    sa = dnn["quant_params"][in0_buf["quant_start_idx"]][0]
    sb = dnn["quant_params"][in1_buf["quant_start_idx"]][0]  
    sc = dnn["quant_params"][out_buf["quant_start_idx"]][0]

    scale_ratio = sa * sb / sc

    # Store values in parameter fields
    op_dict["parameters"]["mode"] = mode
    op_dict["parameters"]["scale_ratio"] = scale_ratio

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    params = op_dict["parameters"]
    scale_ratio = float_to_fp_uint32(params["scale_ratio"], DNN_QUANT_SCALE_Q_FORMAT)
    bytes = []
    bytes.extend(val_to_uint8_array(params["mode"].value, 4, endianness))
    bytes.extend(val_to_uint8_array(scale_ratio, 4, endianness))
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "mode=%s, scale_ratio=%.4f" % (
      params["mode"].name, params["scale_ratio"])
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    # Get input buffer size
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_buf_size = in_buf["data_num_cols"]*in_buf["data_num_rows"]
    in_num_chan = in_buf["dimensions"][DIMS_NUM_CHANNELS]

    in2_buf_idx = op_dict["inputs"][1]
    in2_buf = dnn["buffers"][in2_buf_idx]
    in2_num_batch = in2_buf["dimensions"][DIMS_NUM_BATCHES]

    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]

    # Check if any of the buffers require vertical padding. If they do, they
    # cannot be flattened in CHANNEL_LAST memory order
    can_be_flattened = True
    if config["MEMORY_ORDER"] == MemoryOrder.CHANNEL_LAST:
      if (
        in_buf["dimensions"][DIMS_NUM_ROWS] != in_buf["data_num_rows"]) or (
        in2_buf["dimensions"][DIMS_NUM_ROWS] != in2_buf["data_num_rows"]) or (
        out_buf["dimensions"][DIMS_NUM_ROWS] != out_buf["data_num_rows"]):
        can_be_flattened = False

    mode = op_dict["parameters"]["mode"]
    # Get the memory size based on the mode, and see if we can flatten in0 and
    # perform an add on all input channels in a single operation
    if mode in [ArithmeticMode.SCALAR]:
      # Scalar multiplication requires one temporary buffer of int32s
      mem_size = 4 * in_buf_size
      if can_be_flattened and in_num_chan > 1 and mem_size*in_num_chan <= ram_available:
        mem_size *= in_num_chan
        mode = ArithmeticMode.SCALAR_FLATTENED
    else:
      # Matrix multiplication requires one temporary buffer of int32s
      mem_size = 2 * 4 * in_buf_size
      if can_be_flattened and in_num_chan > 1 and mem_size*in_num_chan <= ram_available:
        mem_size *= in_num_chan
        mode = ArithmeticMode.MATRIX_FLATTENED
    op_dict["parameters"]["mode"] = mode
    return mem_size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    mode = op_dict["parameters"]["mode"]
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]

    in_size = in_buf["data_num_cols"]*in_buf["data_num_rows"]
    in_chan = in_buf["dimensions"][DIMS_NUM_CHANNELS]

    cnt = in_buf["dimensions"][DIMS_NUM_BATCHES]
    cycles = 0
    if mode == ArithmeticMode.SCALAR_FLATTENED:
      cycles += record_instruction("VSMULL_C (Scalar 1,2)", cnt, in_size*in_chan, instr_cnts, instr_cycles)    
      cycles += record_instruction("VADDC_L (Scalar 1)", cnt, in_size*in_chan, instr_cnts, instr_cycles)    
    elif mode == ArithmeticMode.MATRIX:
      cnt *= in_chan
      cycles += record_instruction("VSMULL_C (Scalar 1,2)", cnt, in_size, instr_cnts, instr_cycles)    
      cycles += record_instruction("VADDL_C (Scalar 1)", cnt, in_size, instr_cnts, instr_cycles)    
      cycles += record_instruction("VMADDC_L (Scalar 2)", cnt, in_size, instr_cnts, instr_cycles)    
    elif mode == ArithmeticMode.MATRIX_FLATTENED:
      cycles += record_instruction("VSMULL_C (Scalar 1,2)", cnt, in_chan*in_size, instr_cnts, instr_cycles)    
      cycles += record_instruction("VADDL_C (Scalar 1)", cnt, in_chan*in_size, instr_cnts, instr_cycles)    
      cycles += record_instruction("VMADDC_L (Scalar 2)", cnt, in_chan*in_size, instr_cnts, instr_cycles)    

    return 1000*cycles*VPU_CLOCK_PERIOD_S

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if op_dict["parameters"]["mode"] in [ArithmeticMode.SCALAR, ArithmeticMode.SCALAR_FLATTENED]:
      if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
        patches.append("170940_177827")
      else:
        patches.append("189270")
    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
