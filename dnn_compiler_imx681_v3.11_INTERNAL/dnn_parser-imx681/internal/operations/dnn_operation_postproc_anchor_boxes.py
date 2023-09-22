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


class DNNOperationPostProcAnchorBoxes:
  """
  Operation-specific tasks for POSTPROC_ANCHOR_BOXES operations. See dnn_operation.py
  for details on what each function does.
  """
  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    pass

  def get_op_params_pytorch(self, op_dict, module, node):
    pass

  def add_postprocessing(self, dnn, config, in_idx0, in_idx1, postproc_params):
    if len(postproc_params) != 2:
      error_msg = "Invalid value for POSTPROCESSING field of configuration file. "
      error_msg += "For anchor boxes, must be: POSTPROCESSING ANCHOR_BOXES <data file name>"
      raise RuntimeError(error_msg)
    # Extract anchor box data and store it in static data array
    data_file = os.path.join(config["DATA_DIRECTORY"], postproc_params[1])
    data_addr, scales = self.__postproc_read_anchorbox_data(dnn, data_file)
    # Create anchor box operation and add to the DNN
    self.__postproc_create_op_anchorbox(dnn, config, in_idx0, in_idx1, data_addr, scales)


  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    # Get input buffers
    box_buf_idx = op_dict["inputs"][0]
    box_buf = dnn["buffers"][box_buf_idx]
    box_quant_start_idx = box_buf["quant_start_idx"]

    conf_buf_idx = op_dict["inputs"][1]
    conf_buf = dnn["buffers"][conf_buf_idx]
    conf_quant_start_idx = conf_buf["quant_start_idx"]

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
    bytes.extend(val_to_uint8_array(op_dict["parameters"]["data_addr"], 4, endianness))
    bytes.extend([np.uint8(v) for v in op_dict["parameters"]["col_to_object_type"]])
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "scales=%s, offsets=%s, data_addr=%8x" % (
      ",".join(["%.4f" % scale for scale in params["scale"]]), 
      ",".join(["%.4f" % offset for offset in params["offset"]]), 
      params["data_addr"])
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    # POSTPROC_ANCHOR_BOXES requires the following int32 buffers:
    #   <num_boxes> x <box_size>
    #   <num_boxes> x max(2, <num_classes>)
    #   <bum_boxes> x 1
    #   <bum_boxes> x 1
    in_buf0 = dnn["buffers"][op_dict["inputs"][0]]
    in_buf1 = dnn["buffers"][op_dict["inputs"][1]]
    size = 0

    box_size = in_buf0["data_num_cols"]
    num_classes = in_buf1["data_num_cols"]
    num_boxes = in_buf1["data_num_rows"]
    size += 4 * num_boxes * box_size
    size += 4 * num_boxes * (max(2, num_classes) + 2)
    return size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    box_buf_idx = op_dict["inputs"][0]
    box_buf = dnn["buffers"][box_buf_idx]
    conf_buf_idx = op_dict["inputs"][1]
    conf_buf = dnn["buffers"][conf_buf_idx]

    box_size = box_buf["data_num_cols"]
    num_boxes = box_buf["data_num_rows"]
    num_classes = conf_buf["data_num_cols"]

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


  def __postproc_create_op_anchorbox(self, dnn, config, in_idx0, in_idx1, data_addr, scales):
    """
    Create Anchor Box post processing operation and end it to the end of the DNN

    Inputs:
    dnn - the dnn dictionary
    config - the configuration dictionary
    in_idx0, in_idx1 - the input buffer indices
    data_addr - the address of this operation's static data (anchor boxes)
    scales - scale values for the anchor boxes
    """
    # Create operation dictionary
    op_dict = dict()
    op_dict["op_id"] = DNNOperationID.POSTPROC_ANCHOR_BOXES
    op_dict["parameters"] = dict()
    op_dict["parameters"]["data_addr"] = data_addr
    op_dict["parameters"]["scale"] = scales
    # Convert string from config file to a list by:
    #  Removing [, ], and space characters
    #  Tokenizing using , as the separator
    col_to_object_type = config["POSTPROC_COL_TO_OBJECT_TYPE"]
    if len(col_to_object_type) != config["POSTPROC_COMPARE_VALS_NUM_COLS"]:
      err = "POSTPROC_COLS_TO_OBJECT_TYPE must have POSTPROC_COMPARE_VALS_NUM_COLS (%d) values." % (
        config["POSTPROC_COMPARE_VALS_NUM_COLS"])
      err += " However, it currently has %d values." % len(col_to_object_type)
      raise RuntimeError(err)
    for val in col_to_object_type:
      if int(val) > DNN_NUM_OBJECT_TYPES:
        raise RuntimeError("Object type %d from POSTPROC_COLS_TO_OBJECT_TYPE is outside valid range of [0, %d]" % (
          int(val), DNN_NUM_OBJECT_TYPES))
    op_dict["parameters"]["col_to_object_type"] = col_to_object_type
    in_buf = dnn["buffers"][in_idx0]

    op_dict["inputs"] = [in_idx0, in_idx1]
    op_dict["outputs"] = []
    dnn["operations"].append(op_dict)

