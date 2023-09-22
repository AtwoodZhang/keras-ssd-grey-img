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

class DNNOperationPostProcThreshold:
  """
  Operation-specific tasks for POSTPROC_THRESHOLD operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    pass

  def get_op_params_pytorch(self, op_dict, module, node):
    pass

  def add_postprocessing(self, dnn, config, in_idx0, in_idx1, postproc_params):
    # Create operation dictionary
    op_dict = dict()
    op_dict["op_id"] = DNNOperationID.POSTPROC_THRESHOLD
    op_dict["working_mem_addr"] = 0
    op_dict["inputs"] = [in_idx0, in_idx1]
    op_dict["outputs"] = []
    self.__parse_postproc_params(op_dict, config, postproc_params)
    dnn["operations"].append(op_dict)
 
  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    in_buf_idx = op_dict["inputs"][1]
    in_buf = dnn["buffers"][in_buf_idx]

    if op_dict["parameters"]["data_mode"] == ThresholdDataMode.DEQUANT:
      quant_scale = dnn["quant_params"][in_buf["quant_start_idx"]][0]
      quant_zero = float(dnn["quant_params"][in_buf["quant_start_idx"]][1])

      scale = quant_scale * 100
      offset = quant_scale * -100 * quant_zero
    else:
      scale = 1.0
      offset = 0.0

    op_dict["parameters"]["compare_scale"] = scale
    op_dict["parameters"]["compare_offset"] = offset

    pass

  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    bytes = []
    bytes.append(np.uint8(op_dict["parameters"]["compare_mode"].value))
    bytes.append(np.uint8(op_dict["parameters"]["report_mode"].value))
    report_fields = 0
    for field in op_dict["parameters"]["report_fields"]:
      report_fields |= field.value
    bytes.extend(val_to_uint8_array(report_fields, 2, endianness))
    compare_scale = float_to_fp_uint32(op_dict["parameters"]["compare_scale"], DNN_POSTPROC_THRESHOLD_Q_FORMAT)
    compare_offset = float_to_fp_uint32(op_dict["parameters"]["compare_offset"], DNN_POSTPROC_THRESHOLD_Q_FORMAT)
    bytes.extend(val_to_uint8_array(compare_scale, 4, endianness))
    bytes.extend(val_to_uint8_array(compare_offset, 4, endianness))
    bytes.extend([np.uint8(v) for v in op_dict["parameters"]["col_to_object_type"]])
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "compare=%s, report=%s, fields=%s, scale=%.4f, offset=%.4f" % (
      params["compare_mode"].name, params["report_mode"].name, 
      ",".join([p.name for p in params["report_fields"]]),
      params["compare_scale"], params["compare_offset"])
    return string

  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    # This operation requires working memory that is the same size as the input
    # "compare values" buffer
    in_buf_idx = op_dict["inputs"][1]
    in_buf = dnn["buffers"][in_buf_idx]
    return in_buf["data_num_cols"] * in_buf["data_num_rows"]

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    in_buf_idx = op_dict["inputs"][1]
    in_buf = dnn["buffers"][in_buf_idx]
    in_size = in_buf["data_num_cols"] * in_buf["data_num_rows"]
    cycles = record_instruction("VMADD_C (Scalar 1,2)", 1, in_size, instr_cnts, instr_cycles)
    return 1000*VPU_CLOCK_PERIOD_S*cycles

  def check_for_patches(self, dnn, op_dict):
    return []

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
  def __parse_postproc_params(self, op_dict, config, postproc_params):
    """
    Parse params from configuration file and store in dictionary
    """
    if len(postproc_params) < 5:
      error_msg = "Invalid value for POSTPROCESSING field of configuration file. "
      error_msg += "For threshold, must be: POSTPROCESSING THRESHOLD <data_mode> <compare_mode> <report mode> <report fields...>"
      raise RuntimeError(error_msg)

    if not postproc_params[1] in ThresholdDataMode.__members__.keys():
      error_msg = "Invalid <data_mode> for POSTPROC_THRESHOLD: %s. " % postproc_params[1]
      error_msg += "Valid options are: [%s]" % ",".join(ThresholdDataMode.__members__.keys())
      raise RuntimeError(error_msg)      
    else:
      data_mode = ThresholdDataMode[postproc_params[1]]


    if not postproc_params[2] in ThresholdCompareMode.__members__.keys():
      error_msg = "Invalid <compare_mode> for POSTPROC_THRESHOLD: %s. " % postproc_params[2]
      error_msg += "Valid options are: [%s]" % ",".join(ThresholdCompareMode.__members__.keys())
      raise RuntimeError(error_msg)      
    else:
      compare_mode = ThresholdCompareMode[postproc_params[2]]

    if not postproc_params[3] in ThresholdReportMode.__members__.keys():
      error_msg = "Invalid <report_mode> for POSTPROC_THRESHOLD: %s. " % postproc_params[3]
      error_msg += "Valid options are: [%s]" % ",".join(ThresholdReportMode.__members__.keys())
      raise RuntimeError(error_msg)      
    else:
      report_mode = ThresholdReportMode[postproc_params[3]]

    report_fields = []
    for i in range(4, len(postproc_params)):
      if not postproc_params[i] in ThresholdReportField.__members__.keys():
        error_msg = "Invalid <report_mode> for POSTPROC_THRESHOLD: %s. " % postproc_params[i]
        error_msg += "Valid options are: [%s]" % ",".join(ThresholdReportField.__members__.keys())
        raise RuntimeError(error_msg)
      else:
        report_fields.append(ThresholdReportField[postproc_params[i]])

    op_dict["parameters"] = dict()
    op_dict["parameters"]["data_mode"] = data_mode
    op_dict["parameters"]["compare_mode"] = compare_mode
    op_dict["parameters"]["report_mode"] = report_mode
    op_dict["parameters"]["report_fields"] = report_fields

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
