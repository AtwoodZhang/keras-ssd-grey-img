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

from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions

# Minimum and maximum values of parameters
PARAMETER_SCALE_MIN = 1
PARAMETER_SCALE_MAX = 255
PARAMETER_DROPPED_MIN = 0
PARAMETER_DROPPED_MAX = 255

class DNNOperationInterpolate:
  """
  Operation-specific tasks for INTERPOLATE operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    options = ResizeNearestNeighborOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    # Check that invalid parameters are not provided
    if options.AlignCorners():
      raise RuntimeError("Unsupported parameter: AlignCorners=True. Only supported value is False.")
    if options.HalfPixelCenters():
      raise RuntimeError("Unsupported parameter: HalfPixelCenters=True. Only supported value is False.")

  def get_op_params_pytorch(self, op_dict, module, node):
    pass

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    # Get the sizes of the input and output buffers
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    in_width = in_buf["data_num_cols"]
    in_height = in_buf["data_num_rows"]
    out_width = out_buf["data_num_cols"]
    out_height = out_buf["data_num_rows"]
    # Determine the values of the parameters based on sizing
    h_scale = np.ceil(out_height/in_height)
    w_scale = np.ceil(out_width/in_width)
    num_cols_dropped = in_width*w_scale - out_width
    num_rows_dropped = in_height*h_scale - out_height
    # Check ranges of values
    check_param_range("h_scale", h_scale, PARAMETER_SCALE_MIN, PARAMETER_SCALE_MAX)
    check_param_range("w_scale", w_scale, PARAMETER_SCALE_MIN, PARAMETER_SCALE_MAX)
    check_param_range("num_cols_dropped", num_cols_dropped, PARAMETER_DROPPED_MIN, PARAMETER_DROPPED_MAX)
    check_param_range("num_rows_dropped", num_rows_dropped, PARAMETER_DROPPED_MIN, PARAMETER_DROPPED_MAX)
    # Store values in parameter fields
    op_dict["parameters"]["h_scale"] = h_scale
    op_dict["parameters"]["w_scale"] = w_scale
    op_dict["parameters"]["num_cols_dropped"] = num_cols_dropped
    op_dict["parameters"]["num_rows_dropped"] = num_rows_dropped

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    bytes = []
    bytes.append(np.uint8(op_dict["parameters"]["h_scale"]))
    bytes.append(np.uint8(op_dict["parameters"]["w_scale"]))
    bytes.append(np.uint8(op_dict["parameters"]["num_cols_dropped"]))
    bytes.append(np.uint8(op_dict["parameters"]["num_rows_dropped"]))
    return uint8_array_to_val(bytes, endianness)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "h_scale=%d, w_scale=%d, cols_dropped=%d, rows_dropped=%d" % (
      params["h_scale"], params["w_scale"],
      params["num_cols_dropped"], params["num_rows_dropped"])
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    return 0

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]

    in_size = in_buf["data_num_cols"]*in_buf["data_num_rows"]
    w_scale = op_dict["parameters"]["w_scale"]
    h_scale = op_dict["parameters"]["h_scale"]
    num_batches = in_buf["dimensions"][DIMS_NUM_BATCHES]
    num_channels = in_buf["dimensions"][DIMS_NUM_CHANNELS]

    cycles = record_instruction("VMOV_C", num_batches*num_channels*h_scale*w_scale, in_size, instr_cnts, instr_cycles)

    return 1000 * cycles * VPU_CLOCK_PERIOD_S

  def check_for_patches(self, dnn, op_dict):
    return []

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
