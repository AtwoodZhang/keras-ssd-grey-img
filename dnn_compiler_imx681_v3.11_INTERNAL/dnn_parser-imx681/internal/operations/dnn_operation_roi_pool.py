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

# Approximate number of milliseconds it takes to process one ROI on one input channel
MS_PER_ROI = 0.1125

class DNNOperationROIPool:
  """
  Operation-specific tasks for ROI_POOL operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    # Tflite does not have an ROI_POOL layer type, so this function will never
    # get called. If it did, we would want to extract the spatial_scale parameter
    # here and store it in "scale_b"
    op_dict["parameters"]["mode"] = ROIPoolMode.HW_ASSIST
    op_dict["parameters"]["scale_b"] = 1.0

  def get_op_params_pytorch(self, op_dict, module, node):
    op_dict["parameters"]["mode"] = ROIPoolMode.HW_ASSIST
    op_dict["parameters"]["scale_b"] = module.spatial_scale

  def add_postprocessing(self, dnn, config, in_idx0, in_idx1, postproc_params):
    if len(postproc_params) != 11:
      error_msg = "Invalid value for POSTPROCESSING field of configuration file. "
      error_msg += "For ROI pool, must be: POSTPROCESSING ROI_POOL <spatial_scale> <out_w> <out_h> <scale_a> <scale_b> <scale_c> <zero_a> <zero_b> <zero_c> <num_chan>"
    # Read parameters
    spatial_scale = float(postproc_params[1])
    out_w = int(postproc_params[2])
    out_h = int(postproc_params[3])
    scale_a = float(postproc_params[4])
    scale_b = float(postproc_params[5])
    scale_c = float(postproc_params[6])
    zero_a = int(postproc_params[7])
    zero_b = int(postproc_params[8])
    zero_c = int(postproc_params[9])
    num_chan = int(postproc_params[10])

    op_dict = dict()
    op_dict["op_id"] = DNNOperationID.ROI_POOL
    op_dict["parameters"] = dict()
    op_dict["parameters"]["mode"] = config["ROI_POOL_MODE"]
    op_dict["parameters"]["scale_b"] = spatial_scale

    op_dict["inputs"] = [in_idx0, in_idx1]
    out_idx = self.__create_output_buffer(dnn, num_chan, out_w, out_h, in_idx0, in_idx1)
    op_dict["outputs"] = [out_idx]
    op_dict["working_mem_addr"] = 0

    # Add quantization parameters for each buffer
    dnn["buffers"][in_idx0]["quant_start_idx"] = len(dnn["quant_params"])
    dnn["quant_params"].append([scale_a, zero_a])
    dnn["buffers"][in_idx1]["quant_start_idx"] = len(dnn["quant_params"])
    dnn["quant_params"].append([scale_b, zero_b])
    dnn["buffers"][out_idx]["quant_start_idx"] = len(dnn["quant_params"])
    dnn["quant_params"].append([scale_c, zero_c])

    dnn["operations"].append(op_dict)

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):   
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    
    roi_buf_idx = op_dict["inputs"][1]
    roi_buf = dnn["buffers"][roi_buf_idx]

    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
  
    sa = dnn["quant_params"][in_buf["quant_start_idx"]][0]
    za  = dnn["quant_params"][in_buf["quant_start_idx"]][1]

    sb = dnn["quant_params"][roi_buf["quant_start_idx"]][0]
    zb  = dnn["quant_params"][roi_buf["quant_start_idx"]][1]

    sc = dnn["quant_params"][out_buf["quant_start_idx"]][0]
    zc  = dnn["quant_params"][out_buf["quant_start_idx"]][1]

    # Pre-calculate constants for converting input a from input quantization
    # to output c quantization:
    #   Sc (qc - zc) = Sa (qa - za)
    #   qc = Sa/Sc * (qa - za) + zc
    #   qc = Sa/Sc * qa + (zc - za * Sa/Sc)
    scale_a = sa / sc
    offset_a = zc - za * sa / sc
    # Pre-calculate constant to dequantize and apply spatial scale to roi input, b:
    #   B = Sb (qb - zb) * spatial_scale
    #   B = (Sb * spatial_scale) (qb - zb)
    #   B = (Sb * spatial_scale) * qb + (-zb * Sb * spatial_scale)
    scale_b = op_dict["parameters"]["scale_b"] * sb
    offset_b = scale_b * -1 * zb

    # Store values
    op_dict["parameters"]["scale_a"] = scale_a
    op_dict["parameters"]["scale_b"] = scale_b
    op_dict["parameters"]["offset_a"] = offset_a
    op_dict["parameters"]["offset_b"] = offset_b
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      op_dict["parameters"]["q_format_a"] = DNN_QUANT_SCALE_Q_FORMAT
    else:
      op_dict["parameters"]["q_format_a"] = DNN_QUANT_OFFSET_Q_FORMAT

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    scale_a = float_to_fp_uint32(op_dict["parameters"]["scale_a"], op_dict["parameters"]["q_format_a"])
    offset_a = float_to_fp_uint32(op_dict["parameters"]["offset_a"], op_dict["parameters"]["q_format_a"])

    scale_b = float_to_fp_uint32(op_dict["parameters"]["scale_b"], DNN_ROI_SCALE_Q_FORMAT)
    offset_b = float_to_fp_uint32(op_dict["parameters"]["offset_b"], DNN_ROI_SCALE_Q_FORMAT)

    bytes = []
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["mode"].value, 4, endianness))
    bytes.extend(val_to_uint8_array(scale_a, 4, endianness))
    bytes.extend(val_to_uint8_array(scale_b, 4, endianness))
    bytes.extend(val_to_uint8_array(offset_a, 4, endianness))
    bytes.extend(val_to_uint8_array(offset_b, 4, endianness))
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    return "mode=%s, scale_a=%.4f, offset_a=%.4f, scale_b=%.4f, offset_b=%.4f, " % (
      params["mode"].name, params["scale_a"], params["offset_a"], params["scale_b"], params["offset_b"])

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    # This operation requires one int16 buffer that is the same size as the ROI
    # input

    # Hack: fill in the config mode here, since it cannot be done earlier in the
    # pytorch flow
    op_dict["mode"] = config["ROI_POOL_MODE"]
    in_buf_idx = op_dict["inputs"][1]
    in_buf = dnn["buffers"][in_buf_idx]
    mem_size = 2 * in_buf["data_num_rows"] * in_buf["data_num_cols"]
    if op_dict["parameters"]["mode"] == ROIPoolMode.HW_ASSIST:
      out_buf_idx = op_dict["outputs"][0]
      out_buf = dnn["buffers"][out_buf_idx]
      mem_size += out_buf["data_num_rows"] * out_buf["data_num_cols"]       
    return mem_size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    roi_buf_idx = op_dict["inputs"][1]
    roi_buf = dnn["buffers"][roi_buf_idx]

    num_roi = roi_buf["data_num_rows"]-1
    image_size = out_buf["data_num_rows"] * out_buf["data_num_cols"]
    num_chan = out_buf["dimensions"][DIMS_NUM_CHANNELS]

    # Calculate VPU processing time
    cycles = 0
    # W & H columns
    cycles += record_instruction("VMADDS_C (Scalar 1,2)", 1, 4*num_roi, instr_cnts, instr_cycles)
    cycles += record_instruction("VCLAMP_S", 4, num_roi, instr_cnts, instr_cycles)
    cycles += record_instruction("VMADD_C (Scalar 1,2)", 1, num_roi*num_chan*image_size, instr_cnts, instr_cycles)

    vpu_time = 1000*cycles*VPU_CLOCK_PERIOD_S
    cpu_time = MS_PER_ROI*num_roi*num_chan
    return vpu_time + cpu_time

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      params = op_dict["parameters"]
      # If the offset is too large, require patch that changes q format from s7.24 to s9.22
      if (params["offset_a"] > DNN_QUANT_SCALE_MAX) or (params["offset_a"] < DNN_QUANT_SCALE_MIN):
        patches.append("170940_177827")
    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      # Patch stores parameters in s9.22 format instead of s7.24 format
      if "170940_177827" in patches:
        op_dict["parameters"]["q_format_a"] = DNN_QUANT_OFFSET_Q_FORMAT

  # ============================================================================
  # Private methods
  # ============================================================================
  def __create_output_buffer(self, dnn, num_chan, out_w, out_h, in_idx0, in_idx1):
    """
    Create a buffer to store the output in

    Parameters:
    dnn - the dnn dictionary
    num_chan - number of channels
    out_w,h - the output width and height size
    buf_start_idx - index in scratch RAM where this buffer will start
    in_idx0,1 - buffer indexes of the inputs to this operation

    Returns:
    output buffer index
    """
    # Get the input buffers
    in0_buf = dnn["buffers"][in_idx0]
    in1_buf = dnn["buffers"][in_idx1]

    # Hack: instead of input buffers being children of the DNN's outputs, they
    # are completely new input buffers
    in0_buf["buffer_type"] = BufferType.MODEL_INPUT
    dnn["buffers"][in0_buf["parent"]]["num_connections"] -= 1
    in0_buf["parent"] = -1
    in0_buf["num_connections"] = 1
    in1_buf["buffer_type"] = BufferType.MODEL_INPUT
    dnn["buffers"][in1_buf["parent"]]["num_connections"] -= 1
    in1_buf["parent"] = -1
    in1_buf["num_connections"] = 1

    # Hack input buffers to have dims = datasize
    in0_buf["dimensions"][DIMS_NUM_ROWS] = in0_buf["data_num_rows"]
    in0_buf["dimensions"][DIMS_NUM_COLS] = in0_buf["data_num_cols"]
    in1_buf["dimensions"][DIMS_NUM_ROWS] = in1_buf["data_num_rows"]
    in1_buf["dimensions"][DIMS_NUM_COLS] = in1_buf["data_num_cols"]
    in0_buf["dimensions"][DIMS_NUM_CHANNELS] = num_chan
    in1_buf["dimensions"][DIMS_NUM_CHANNELS] = 1
    
    out_batches = in1_buf["dimensions"][DIMS_NUM_ROWS]-1
    out_channels = in0_buf["dimensions"][DIMS_NUM_CHANNELS]
    dims = [out_batches, out_h, out_w, out_channels]

    buf_dict = dict()
    buf_dict["data_signed"] = True
    buf_dict["data_type"] = DataType.CHAR
    buf_dict["bpp_shift"] = 0
    buf_dict["quant_type"] = QuantType.PER_TENSOR
    buf_dict["dimensions"] = dims
    buf_dict["data_start_row"] = 0
    buf_dict["data_start_col"] = 0
    buf_dict["data_num_rows"] = dims[DIMS_NUM_ROWS]
    buf_dict["data_num_cols"] = dims[DIMS_NUM_COLS]
    buf_dict["quant_start_idx"] = in0_buf["quant_start_idx"]

    buf_dict["buffer_id"] = 0   
    buf_dict["num_connections"] = 1
    buf_dict["parent"] = -1
    buf_dict["dim_reorder"] = []
    buf_dict["buffer_type"] = BufferType.SCRATCH_RAM
    buf_dict["start_idx"] = 0 # filled in later
    out_idx = len(dnn["buffers"])
    dnn["buffers"].append(buf_dict)
    return out_idx
