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

from .dnn_operation_addsub import DNNOperationAddSub
from .dnn_operation_concatenate import DNNOperationConcatenate
from .dnn_operation_conv2d import DNNOperationConv2D
from .dnn_operation_fully_connected import DNNOperationFullyConnected
from .dnn_operation_interpolate import DNNOperationInterpolate
from .dnn_operation_max_pool import DNNOperationMaxPool
from .dnn_operation_multiply import DNNOperationMultiply
from .dnn_operation_postproc_anchor_boxes import DNNOperationPostProcAnchorBoxes
from .dnn_operation_postproc_threshold import DNNOperationPostProcThreshold
from .dnn_operation_relu import DNNOperationRelu
from .dnn_operation_reshape import DNNOperationReshape
from .dnn_operation_sigmoid import DNNOperationSigmoid
from .dnn_operation_softmax import DNNOperationSoftmax
from .dnn_operation_transpose import DNNOperationTranspose
from .dnn_operation_roi_pool import DNNOperationROIPool
from .dnn_operation_generate_proposals import DNNOperationGenerateProposals

# List of operations that do not support cropping around their input buffers
OPERATIONS_WITHOUT_CROP_SUPPORT = [
  DNNOperationID.SOFTMAX,  # SOFTMAX flattens a buffer from H x W x C to 
                           # (H*W) x C, so crop is not supported.
  DNNOperationID.RESHAPE   # RESHAPE changes the dimensions of the data without
                           # moving any data around, so crop is not supported.
]

# List of DNN operations that support padding around input buffers
OPERATIONS_WITH_PADDING = [
  DNNOperationID.CONV_2D, 
  DNNOperationID.DEPTHWISE_CONV_2D, 
  DNNOperationID.MAX_POOL
]

class DNNOperation:
  """
  This top-level class contains functions that perform operation-specific tasks.
  Each method checks the operation type, and calls into that operation's class
  to perform the task.
  """

  @staticmethod
  def get_op_params_tflite(op_dict, graph, op, opcode):
    """
    Populate "parameters" field in an operation dictionary with parameters that
    are extracted directly from a tflite structure

    Parameters:
    op_dict - the dictionary to add parameters to
    graph - the tflite model's subgraph
    op - the tensorflow operation object
    opcode - the tensorflow BuiltinOperator code

    Raises:
    RuntimeError if unsupported layer type or parameter is found
    """
    op_dict["parameters"] = dict()
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    operation.get_op_params_tflite(op_dict, graph, op, opcode)

  @staticmethod
  def get_op_params_pytorch(op_dict, module, node):
    """
    Populate "parameters" field in an operation dictionary with parameters that
    are extracted directly from a pytorch module

    Parameters:
    op_dict - the dictionary to add parameters to
    module - the pytorch module class for this layer
    node - node from the trace graph for this layer

    Raises:
    RuntimeError if unsupported layer type or parameter is found
    """
    op_dict["parameters"] = dict()
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    return operation.get_op_params_pytorch(op_dict, module, node)

  @staticmethod
  def finalize_op_params(dnn, op_dict, endianness, mem_order):
    """
    Finalize operation parameters that require the entire DNN to be loaded before
    they can be calculated
    
    Parameters:
    dnn - the dnn's top-level dictionary
    op_dict - the dictionary to finalize parameters in
    endianness - the data endianness (big or little)
    mem_order - the data memory order (channel last vs channel first)

    Raises:
    RuntimeError if parameter value is out of range
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    operation.finalize_op_params(dnn, op_dict, endianness, mem_order)

  @staticmethod
  def supports_cropping(op_dict):
    """
    Returns true if this operation supports an input buffer that is a cropped
    portion of a larger buffer (e.g. buffer_size > data_size)
    
    Parameters:
    op_dict - the operation's dictionary

    Returns:
    TRUE if cropping is supported
    """

    if op_dict["op_id"] in OPERATIONS_WITHOUT_CROP_SUPPORT:
      return False
    else:
      return True

  @staticmethod
  def set_padding(dnn, op_dict):
    """
    Set the values of the pad_left, pad_right, pad_top, and pad_bottom parameters
    of an operation that supports padding based on buffer sizes.

    Parameters:
    dnn - the dnn's top-level dictionary
    op_dict - the operation's dictionary
    """
    if op_dict["op_id"] in OPERATIONS_WITH_PADDING:
      operation = DNNOperation.__get_operation_class(op_dict["op_id"])
      operation.set_padding(dnn, op_dict)

  @staticmethod
  def op_params_to_byte_array(op_dict, dnn, endianness):
    """
    Convert operation parameters to a raw byte array and append it to the misc
    data array.
    
    Parameters:
    op_dict - the dictionary to get parameters from
    dnn - the dnn dictionary with misc data array, which parameter values may be added to
    endianness - "big" for MSB first, "little" for LSB first

    Returns:
      uint32_t parameter value to write to the operation's structure. This value
      is either the misc_data start address, or bitfields if misc data is not
      used.
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    return operation.op_params_to_byte_array(op_dict, dnn, endianness)

  @staticmethod
  def op_params_to_string(dnn, op_dict):
    """
    Get a string describing all operation parameter values

    Parameters:
    dnn - the dnn dictionary
    op_dict - the operation dictionary

    Returns:
    string of parameters (or empty for none)
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    return operation.op_params_to_string(dnn, op_dict)

  @staticmethod
  def get_working_memory_size(dnn, op_dict, config, ram_available):
    """
    Determine how many bytes of working memory for temporary reuslts are
    required for a given operation based on the opcode and the buffer sizes.

    Parameters:
    op_dict - the operation dictionary
    buffers - buffer dictionary
    config - the configuration dictionary
    ram_available - number of bytes of ram available

    Returns:
    the working memory size for this operation
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    size = operation.get_working_memory_size(dnn, op_dict, config, ram_available)
    while (size % 4 != 0):
      size += 1
    return size

  @staticmethod
  def get_processing_time(dnn, op_dict, instr_cnts, instr_cycles):
    """
    Determine the processing time, in ms, for a given operation

    Parameters:
    dnn - the dnn dictionary
    op_dict - this operation's dictionary
    instr_cnt - dictionary that tracks how many times each instruction is called
    instr_cycles - dictionary that tracks how many cycles run for each instruction

    Returns:
    the working memory size for this operation
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    return operation.get_processing_time(dnn, op_dict, instr_cnts, instr_cycles)

  @staticmethod
  def add_postprocessing(dnn, config, in_idx0, in_idx1):
    """
    Add a post-processing operation to the end of the DNN

    Parameters:
    dnn - the DNN to add post-processing to
    config - configuration parameters
    in_idx0, in_idx1 - indices of the postprocessing inputs
    """
    postproc_params = config["POSTPROCESSING"].split(' ')
    if postproc_params[0] == "ANCHOR_BOXES":
      operation = DNNOperationPostProcAnchorBoxes()
      operation.add_postprocessing(dnn, config, in_idx0, in_idx1, postproc_params)
    elif postproc_params[0] == "THRESHOLD":
      operation = DNNOperationPostProcThreshold()
      operation.add_postprocessing(dnn, config, in_idx0, in_idx1, postproc_params)
    elif postproc_params[0] == "ROI_POOL":
      # NOTE: ROI_POOL isn't actually a post processing layer, but this mode is
      # supported to allow an ROI_POOL layer to be appended to the end of a
      # DNN for testing purposes.
      operation = DNNOperationROIPool()
      operation.add_postprocessing(dnn, config, in_idx0, in_idx1, postproc_params)
    elif postproc_params[0] == "GENERATE_PROPOSALS":
      # NOTE: GENERATE_PROPOSALS isn't actually a post processing layer, but this mode is
      # supported to allow an GENERATE_PROPOSALS layer to be appended to the end of a
      # DNN for testing purposes.
      operation = DNNOperationGenerateProposals()
      operation.add_postprocessing(dnn, config, in_idx0, in_idx1, postproc_params)
    else:
      error_msg = "Invalid value for POSTPROCESSING field of configuration file. "
      error_msg += "Value must start with one of: ANCHOR_BOXES, THRESHOLD, ROI_POOL, GENERATE_PROPOSALS"
      raise RuntimeError(error_msg)

  @staticmethod
  def check_for_patches(dnn, op_dict):
    """
    Check if a given operation requires a firmware patch to run

    Parameters:
    dnn - the DNN dictionary
    op_dict - the operation dictionary
  
    Returns:
    list of patches that are needed
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    return operation.check_for_patches(dnn, op_dict)

  @staticmethod
  def adjust_for_patches(dnn, op_dict, patches):
    """
    Adjusts operation to account for any patches that will be loaded

    Parameters:
    dnn - the DNN dictionary
    op_dict - the operation dictionary
    patches - list of patches that will be loaded
    """
    operation = DNNOperation.__get_operation_class(op_dict["op_id"])
    operation.adjust_for_patches(dnn, op_dict, patches)

  # ============================================================================
  # Private methods
  # ============================================================================
  def __get_operation_class(op_id):
    if op_id in [DNNOperationID.ADDSUB]:
      operation = DNNOperationAddSub()
    elif op_id in [DNNOperationID.CONV_2D, DNNOperationID.DEPTHWISE_CONV_2D]:
      operation = DNNOperationConv2D()
    elif op_id in [DNNOperationID.CONCATENATE]:
      operation = DNNOperationConcatenate()
    elif op_id in [DNNOperationID.FULLY_CONNECTED]:
      operation = DNNOperationFullyConnected()
    elif op_id in [DNNOperationID.INTERPOLATE]:
      operation = DNNOperationInterpolate()
    elif op_id in [DNNOperationID.MAX_POOL]:
      operation = DNNOperationMaxPool()
    elif op_id in [DNNOperationID.MULTIPLY]:
      operation = DNNOperationMultiply()
    elif op_id in [DNNOperationID.POSTPROC_ANCHOR_BOXES]:
      operation = DNNOperationPostProcAnchorBoxes()
    elif op_id in [DNNOperationID.POSTPROC_THRESHOLD]:
      operation = DNNOperationPostProcThreshold()
    elif op_id in [DNNOperationID.RELU]:
      operation = DNNOperationRelu()
    elif op_id in [DNNOperationID.RESHAPE]:
      operation = DNNOperationReshape()
    elif op_id in [DNNOperationID.SIGMOID]:
      operation = DNNOperationSigmoid()
    elif op_id in [DNNOperationID.SOFTMAX]:
      operation = DNNOperationSoftmax()
    elif op_id in [DNNOperationID.TRANSPOSE]:
      operation = DNNOperationTranspose()
    elif op_id in [DNNOperationID.ROI_POOL]:
      operation = DNNOperationROIPool()
    elif op_id in [DNNOperationID.GENERATE_PROPOSALS]:
      operation = DNNOperationGenerateProposals()
    else:
      raise RuntimeError("Unsupported layer type: %s" % op_id.name)

    return operation
