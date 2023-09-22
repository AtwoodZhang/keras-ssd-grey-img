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
import os

from internal.constants import *
from internal.utils import *


class DNNOperationGenerateProposals:
  """
  Operation-specific tasks for GENERATE_PROPOSALS operations. See dnn_operation.py
  for details on what each function does.
  """
  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    params = []
    # TODO: Read these from model
    params["scale"] = [1.0, 1.0, 1.0, 1.0]
    params["data_addr"] = 0
    params["width_thresh"] = 0
    params["height_thresh"] = 0
    params["iou_thresh"] = 30
    params["conf_thresh"] = 0
    op_dict["parameters"] = params

  def get_op_params_pytorch(self, op_dict, module, node):
    params = []
    # TODO: Read these from model
    params["scale"] = [1.0, 1.0, 1.0, 1.0]
    params["data_addr"] = 0
    params["width_thresh"] = 0
    params["height_thresh"] = 0
    params["iou_thresh"] = 30
    params["conf_thresh"] = 0
    op_dict["parameters"] = params

  def add_postprocessing(self, dnn, config, in_idx0, in_idx1, postproc_params):
    if len(postproc_params) != 7:
      error_msg = "Invalid value for POSTPROCESSING field of configuration file. "
      error_msg += "For GENERATE_PROPOSALS, must be: POSTPROCESSING GENERATE_PROPOSALS "
      error_msg += "<data file name> <outboxes> <iouthresh> <widththresh> <heightthresh> "
      error_msg += "<confthresh>"
      raise RuntimeError(error_msg)
    # Extract anchor box data and store it in static data array
    filename = os.path.normpath(os.path.join(config["INPUT_DIRECTORY"], postproc_params[1]))
    data_addr, scales = self.__postproc_read_anchorbox_data(dnn, filename)
    # Create anchor box operation and add to the DNN
    op_dict = dict()
    op_dict["op_id"] = DNNOperationID.GENERATE_PROPOSALS
    op_dict["parameters"] = dict()
    op_dict["parameters"]["data_addr"] = data_addr
    op_dict["parameters"]["scale"] = scales
    op_dict["parameters"]["iou_thresh"] = int(postproc_params[3])
    op_dict["parameters"]["width_thresh"] = int(postproc_params[4])
    op_dict["parameters"]["height_thresh"] = int(postproc_params[5])
    op_dict["parameters"]["conf_thresh"] = int(postproc_params[6])
    op_dict["inputs"] = [in_idx0, in_idx1]

    out_idx = self.__create_output_buffer(dnn, int(postproc_params[2]))
    op_dict["outputs"] = [out_idx]
    op_dict["working_mem_addr"] = 0
    dnn["operations"].append(op_dict)

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    # Get input buffers
    box_buf_idx = op_dict["inputs"][0]
    box_buf = dnn["buffers"][box_buf_idx]
    box_quant_start_idx = box_buf["quant_start_idx"]

    conf_buf_idx = op_dict["inputs"][1]
    conf_buf = dnn["buffers"][conf_buf_idx]
    conf_quant_start_idx = conf_buf["quant_start_idx"]

    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    out_quant_start_idx = out_buf["quant_start_idx"]

    # Make sure there is only one batch of input data
    if box_buf["dimensions"][DIMS_NUM_BATCHES] > 1:
      err_msg = "GENERATE_PROPOSALS only supports 1 input batch, but "
      err_msg += "%d batches found." % box_buf["dimensions"][DIMS_NUM_BATCHES]
      raise RuntimeError(err_msg)

    # Add a row to the output buffer for the number of boxes
    out_buf["dimensions"][DIMS_NUM_ROWS] += 1
    out_buf["data_num_rows"] += 1

    # Calculate output scale factor
    quant_scale = dnn["quant_params"][out_quant_start_idx][0]
    quant_zero  = dnn["quant_params"][out_quant_start_idx][1]
    op_dict["parameters"]["out_scale"] = 1.0/quant_scale
    op_dict["parameters"]["out_offset"] = quant_zero

    quant_scale = dnn["quant_params"][box_quant_start_idx][0]
    quant_zero  = dnn["quant_params"][box_quant_start_idx][1]

    # Calculate scale factors and offsets for each anchor box coordinate
    offsets = []
    for i in range(0, AnchorBoxInput.SIZE.value):
      # assuming QUANT_SCALE_Q_FORMAT = 24, ANCHOR_SCALE_Q_FORMAT = 17,
      # scale = (quant_scale / anchor_scale) >> 7
      #       = (s0.24       / u4          ) >> 7
      #       = (s0.24) >> 7
      #       = (s0.17)
      anchor_scale = op_dict["parameters"]["scale"][i]
      op_dict["parameters"]["scale"][i] = quant_scale / anchor_scale
      offset = -1.0 * quant_zero * quant_scale / anchor_scale
      if i == AnchorBoxInput.W.value or i == AnchorBoxInput.H.value:
        offset += 1.0
      offsets.append(offset)

    # Calculate scale factors and offsets for confidence values
    #  scale = 100  * quant_scale
    #          s7.0 * s0.24
    #          s7.24 --> stored as s7.15
    #  offset = scale * quant_zero * -100
    #         = s0.24 * s7.0       * s7.0
    #         = s14.24 --> stored as s14.15
    quant_scale = dnn["quant_params"][conf_quant_start_idx][0]
    quant_zero  = dnn["quant_params"][conf_quant_start_idx][1]
    op_dict["parameters"]["scale"].append(quant_scale * 100)
    offsets.append(quant_scale * -100 * quant_zero)

    op_dict["parameters"]["offset"] = offsets

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    # Write parameters to misc data array
    bytes = []
    for i in range(0, len(op_dict["parameters"]["scale"])):
      if i == (len(op_dict["parameters"]["scale"])-1):
        scale = float_to_fp_uint32(op_dict["parameters"]["scale"][i], DNN_ANCHOR_DECODE_Q_FORMAT)
      else:
        scale = float_to_fp_uint32(op_dict["parameters"]["scale"][i], DNN_QUANT_SCALE_Q_FORMAT)
      bytes.extend(val_to_uint8_array(scale, 4, endianness))
    for i in range(0, len(op_dict["parameters"]["offset"])):
      if i == (len(op_dict["parameters"]["offset"])-1):
        offset = float_to_fp_uint32(op_dict["parameters"]["offset"][i], DNN_ANCHOR_DECODE_Q_FORMAT)
      else:
        offset = float_to_fp_uint32(op_dict["parameters"]["offset"][i], DNN_QUANT_SCALE_Q_FORMAT)
      bytes.extend(val_to_uint8_array(offset, 4, endianness))
    out_scale = float_to_fp_uint32(op_dict["parameters"]["out_scale"], DNN_ROI_SCALE_Q_FORMAT)
    bytes.extend(val_to_uint8_array(out_scale, 4, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["out_offset"], 4, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["data_addr"], 4, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["width_thresh"], 1, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["height_thresh"], 1, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["iou_thresh"], 1, endianness))
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["conf_thresh"], 1, endianness))
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "scales=%s, offsets=%s, out_scale=%.4f, out_offset=%d, data_addr=%8x, wthresh=%d, hthresh=%d, iouthresh=%d, confthresh=%d" % (
      ",".join(["%.4f" % scale for scale in params["scale"]]), 
      ",".join(["%.4f" % offset for offset in params["offset"]]),
      params["out_scale"], params["out_offset"],
      params["data_addr"], params["width_thresh"], params["height_thresh"], 
      params["iou_thresh"], params["conf_thresh"])
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    # POSTPROC_ANCHOR_BOXES requires the following int32 buffers:
    #   <num_boxes> x <box_size>
    #   <num_boxes> x <num_classes>
    #   <bum_boxes> x 1
    #   <bum_boxes> x 1
    in_buf0 = dnn["buffers"][op_dict["inputs"][0]]
    in_buf1 = dnn["buffers"][op_dict["inputs"][1]]
    size = 0

    box_size = in_buf0["data_num_cols"]
    num_classes = in_buf1["data_num_cols"]
    num_boxes = in_buf1["data_num_rows"]
    size += 4 * num_boxes * box_size
    size += 4 * num_boxes * (num_classes + 2)
    return size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    box_buf_idx = op_dict["inputs"][0]
    box_buf = dnn["buffers"][box_buf_idx]
    conf_buf_idx = op_dict["inputs"][1]
    conf_buf = dnn["buffers"][conf_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]

    box_size = box_buf["data_num_cols"]
    num_boxes = box_buf["data_num_rows"]
    num_classes = conf_buf["data_num_cols"]
    out_size = out_buf["data_num_cols"]*out_buf["data_num_rows"]

    # CPU processing time for anchor boxes varies based on how many boxes meet
    # various thresholds. Use these values to get a reasonable estimate of the
    # typical case
    #   Number of boxes with width > threshold && height > threshold
    BOXES_WITH_LARGE_ENOUGH_SIZE = num_boxes/2
    #   Number of boxes with confidence > threshold, which are considered for
    #   the final output
    BOXES_WITH_OBJECTS = 12
    #  Number of unique objects in final output
    FINAL_OUTPUT_BOXES = 4

    # Calculate VPU processing time
    cycles = 0
    # W & H columns
    cycles += record_instruction("VMADDL_C (Scalar 1)", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VMADDL_C (Scalar 1,2)", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VMUL_L", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VCLAMP_L", 2, num_boxes, instr_cnts, instr_cycles)

    # X & Y columns
    cycles += record_instruction("VMADDL_C (Scalar 1)", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VMADDL_C (Scalar 1,2)", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VMADD_L", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VSUB_L", 2, num_boxes, instr_cnts, instr_cycles)
    cycles += record_instruction("VCLAMP_L", 2, num_boxes, instr_cnts, instr_cycles)

    # Confidence decode
    cycles += record_instruction("VMADDL_C (Scalar 1,2)", 1, num_boxes * num_classes, instr_cnts, instr_cycles)

    vpu_time = 1000*cycles*VPU_CLOCK_PERIOD_S

    # Calculate CPU processing time
    cpu_time_ns = 2200  # baseline time to enter/exit processing
    output_boxes = 0
    for i in range(0, num_boxes):
      cpu_time_ns += 200 # baseline time for loop
      if (i < BOXES_WITH_LARGE_ENOUGH_SIZE):
        cpu_time_ns += 250*num_classes # time to get maximum confidence
        if (i < BOXES_WITH_OBJECTS):
          for j in range(0, output_boxes):
            cpu_time_ns += 800 # time to compare box to previous
          if (output_boxes < FINAL_OUTPUT_BOXES):
            output_boxes += 1
            cpu_time_ns += 750 # time to try to write box

    return vpu_time + cpu_time_ns * 1e3/1e9

  def check_for_patches(self, dnn, op_dict):
    return []

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
  def __postproc_read_anchorbox_data(self, dnn, filename):
    """
    Read anchor box data from a text file and store it in DNN's static_data array

    Parameters:
    dnn - the dnn dictionary, which will have data added to it
    filename - the file to parse for values

    Returns:
    data_addr - the start address of the data in the static_data array
    scales - scale values for the anchor boxes
    """
    first_data = True
    data_addr = len(dnn["static_data"])
    scales = []
    with open(filename, "r") as file:
      for line in file:
        # Remove whitespace
        line = line.strip()
        # Ignore comment lines
        if line.startswith("#"):
          continue
        # Split by commas, remove empty entry from the end
        vals = line.split(",")
        vals = vals[:-1]
        # Make sure there are 4 values, and convert them all to ints
        if len(vals) != 4:
          raise RuntimeError("Invalid line in anchor box file: " + line)
        # If this is the first line of data, interpret values as scales. Otherwise,
        # interpret it as the scales. Otherwise, add it to the static data array
        if first_data:
          vals = [float(v) for v in vals]
          scales = vals
          first_data = False
        else:
          vals = [np.uint8(v) for v in vals]
          dnn["static_data"].extend(vals)
    return data_addr, scales

  def __create_output_buffer(self, dnn, max_boxes):
    """
    Create a buffer to store the output in

    Parameters:
    dnn - the dnn dictionary
    max_boxes - maximum number of output boxes

    Returns:
    output buffer index
    """
    
    dims = [1, max_boxes, 4, 1]

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
    buf_dict["quant_start_idx"] = len(dnn["quant_params"])
    dnn["quant_params"].append([1.0, -128])

    buf_dict["buffer_id"] = 0   
    buf_dict["num_connections"] = 1
    buf_dict["parent"] = -1
    buf_dict["dim_reorder"] = []
    buf_dict["buffer_type"] = BufferType.SCRATCH_RAM
    buf_dict["start_idx"] = 0 # filled in later
    out_idx = len(dnn["buffers"])
    dnn["buffers"].append(buf_dict)
    return out_idx
