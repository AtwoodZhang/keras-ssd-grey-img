# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import copy
import logging

from internal.constants import *
from internal.utils import *
from internal.operations.dnn_operation import DNNOperation
from internal.operations.dnn_operation import DNNOperationConv2D
from internal.operations.dnn_operation import OPERATIONS_WITH_PADDING
from internal.operations.dnn_operation import OPERATIONS_WITHOUT_CROP_SUPPORT

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

class DNNFinalizer:
  """
  Class that handles finalizing the structure and parameter values in a DNN
  to make it compatible with IMX firmware.
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self):
    pass

  def add_postprocessing(self, dnn, config):
    """
    Add a post processing operation to the end of the DNN based on the config
    file parameters provided

    Parameters:
    dnn - the DNN compiler output dictionary
    config - config dictionary containing all configuration parameters

    Raises:
    ValueError if invalid configuration is provided
    """
    # Check if we are in a test mode. If we are, ignore invalid cropping
    postproc_params = config["POSTPROCESSING"].split(' ')
    if postproc_params[0] in ["ROI_POOL", "GENERATE_PROPOSALS"]:
      check_cropping = False
    else:
      check_cropping = True


    # Create buffers for each post processing input
    in_idx0, in_idx1 = self.__postproc_create_input_buffers(dnn, config, check_cropping)

    # Add post processing operation
    DNNOperation.add_postprocessing(dnn, config, in_idx0, in_idx1)

    if len(config["POSTPROC_THRESHOLD_VALS"]) > config["NUM_DNN_POSTPROC_THRESHOLD_REGS"]:
      err_msg = "Length of POSTPROC_THRESHOLD_VALS array (%d) exceeds " % len(config["POSTPROC_THRESHOLD_VALS"])
      err_msg += "number of DNN_POSTPROC_THRESHOLD registers (%d)" % config["NUM_DNN_POSTPROC_THRESHOLD_REGS"]
      raise ValueError(err_msg)



  def pad_buffers(self, dnn):
    """
    Increase the size of buffers in the DNN to include space for padding around
    the image, when applicable

    Parameters:
    dnn - the DNN compiler output dictionary to modify

    Raises:
    RuntimeError if padding could not be added to a buffer
    """
    op_idx = 0
    for op in dnn["operations"]:
      if op["op_id"] in OPERATIONS_WITH_PADDING:
        # Calculate how much padding is needed on each side of the buffer
        DNNOperation.set_padding(dnn, op)
        # Pad the input buffer
        out_buf_idx = op["outputs"][0]
        in_buf_idx = op["inputs"][CONV_INPUT_IMAGE]
        in_buf = dnn["buffers"][in_buf_idx]
        pad_left = op["parameters"]["pad_left"]
        pad_right = op["parameters"]["pad_right"]
        pad_top = op["parameters"]["pad_top"]
        pad_bottom = op["parameters"]["pad_bottom"]
        if (pad_left + pad_right + pad_top + pad_bottom) > 0:
          try:
            self.__pad_buffer(dnn, in_buf, in_buf_idx, pad_left, pad_right, pad_top, pad_bottom)
          except RuntimeError as err:
            raise RuntimeError(("Operation %d (%s) requires padding on its input, " +
              "but padding could not be added: %s") % (op_idx, op["op_idx"].name, err))
          # Make sure that if any other operation is connected to this operation's
          # input buffer and qualifies for the optimized 1x1 filter mode, we add
          # the same amount of horizontal padding to its other input/output buffer
          # to ensure they have the same number of columns.
          other_op_idx = 0
          for other_op in dnn["operations"]:
            if DNNOperationConv2D.check_opt1x1_criteria(other_op, dnn):
              for other_out_idx in other_op["outputs"]:
                if other_out_idx == in_buf_idx:
                  other_in_idx = other_op["inputs"][CONV_INPUT_IMAGE]
                  other_in_buf = dnn["buffers"][other_in_idx]
                  try:
                    self.__pad_buffer(dnn, other_in_buf, other_in_idx, pad_left, pad_right, 0, 0)
                    logging.debug(("Added horizontal padding around operation %d to allow " +
                      "optimized 1x1 CONV2D implementation to be used") % other_op_idx)
                  except RuntimeError as err:
                    logging.debug(("Added horizontal padding around operation %d caused operation" +
                      "%d to no longer be compatible with the optimized 1x1 CONV2D implementation") % (
                        op_idx, other_op_idx))
              for other_in_idx in other_op["inputs"]:
                if other_in_idx == in_buf_idx:
                  other_out_idx = other_op["outputs"][0]
                  other_out_buf = dnn["buffers"][other_out_idx]
                  try:
                    self.__pad_buffer(dnn, other_out_buf, other_out_idx, pad_left, pad_right, 0, 0)
                    logging.debug(("Added horizontal padding around operation %d to allow " +
                      "optimized 1x1 CONV2D implementation to be used") % other_op_idx)
                  except RuntimeError as err:
                    logging.debug(("Added horizontal padding around operation %d caused operation" +
                      "%d to no longer be compatible with the optimized 1x1 CONV2D implementation") % (
                        op_idx, other_op_idx))
            other_op_idx += 1
        op_idx += 1

  def finalize_buffer_descriptors(self, dnn, config):
    """
    Adds calculated fields to a buffer's dictionary that will be written out to
    the buffer descriptor object. These include:

    data_start_offset
    row_size
    batch_size

    start_idx may also be modified if reverse_alloc is being used

    Parameters:
    dnn - the dnn dictionary
    config - configuration dictionary
    """
    reverse_alloc = config["REVERSE_ALLOC_SCRATCH_RAM"]
    max_scratch_mem_size = config["DNN_RAM_MAX_SIZE"]
    for buf in dnn["buffers"]:
      dims = buf["dimensions"]
      if config["MEMORY_ORDER"] == MemoryOrder.CHANNEL_LAST:
        col_size = 1 << buf["bpp_shift"]
        buf["row_size"] = col_size * dims[DIMS_NUM_COLS]
        # Use channel size as batch size
        buf["batch_size"] = buf["row_size"] * dims[DIMS_NUM_ROWS]
        total_size = buf["batch_size"] * dims[DIMS_NUM_CHANNELS] * dims[DIMS_NUM_BATCHES]
      else:
        col_size = dims[DIMS_NUM_CHANNELS] << buf["bpp_shift"]
        buf["row_size"] = col_size * dims[DIMS_NUM_COLS]
        buf["batch_size"] = buf["row_size"] * dims[DIMS_NUM_ROWS]
        total_size = buf["batch_size"] * dims[DIMS_NUM_BATCHES]
      if (reverse_alloc == 1) and (buf["buffer_type"] in [BufferType.SCRATCH_RAM]):
        start_idx = max(0, max_scratch_mem_size - (buf["start_idx"] + total_size))
        buf["start_idx"] = align_value(start_idx, 4, False)
      if buf["parent"] >= 0:
        buf["start_idx"] = dnn["buffers"][buf["parent"]]["start_idx"]
      buf["data_start_offset"] = buf["start_idx"] + buf["data_start_row"] * buf["row_size"]
      buf["data_start_offset"] += buf["data_start_col"] * col_size
    # If reverse alloc, reverse working memory addresses too
    if reverse_alloc == 1:
      for op in dnn["operations"]:
        if op["working_mem_size"] > 0:
          start_idx =  max_scratch_mem_size - (op["working_mem_addr"] + op["working_mem_size"])
          op["working_mem_addr"] = align_value(start_idx, 4, False)

  def finalize_op_params(self, dnn, endianness, mem_order):
    """
    Calculate the values of any operation-specific parameters that could not be
    calculated when the input file was read. This is generally because their value
    is based on other information, such as buffer sizes or quantization parameters
    that was not available when the operation dictionary was created.

    Parameters:
    dnn - the dnn dictionary
    endianness - "big" or "little"
    mem_order - "channel first" or "channel last"

    Raises:
    RuntimeError if parameter value is out of range
    """
    for op_idx in range(0, len(dnn["operations"])):
      op = dnn["operations"][op_idx]
      try:
        DNNOperation.finalize_op_params(dnn, op, endianness, mem_order)
      except RuntimeError as e:
        err = "Unsupported layer parameter at Operation %d (%s): %s)" % (
          op_idx, op["op_id"].name, e)
        raise RuntimeError(err)


  def split_operations(self, dnn, cfg):
    """
    If needed, split operations into multiple separate operations in order to
    ensure they are compatible with firmware.

    Parameters:
    dnn - the dnn dictionary
    """
    op_idx = 0
    while op_idx < len(dnn["operations"]):
      op = dnn["operations"][op_idx]
      if (op["op_id"] == DNNOperationID.RESHAPE) and (op["parameters"]["transpose"]):
        in_buf_idx = op["inputs"][0]
        in_buf = dnn["buffers"][in_buf_idx]
        if buffer_has_stride(in_buf):
           # create temporary buffer that we can copy the input to to remove stride
           # this buffer is the same as the input, except that it has no stride
           # and is only used between the copy and reshape.
           temp_buf = copy.deepcopy(in_buf)
           buffer_remove_stride(temp_buf)
           temp_buf["num_connections"] = 2
           temp_buf["parent"] = -1
           temp_buf_idx = len(dnn["buffers"])
           dnn["buffers"].append(temp_buf)
           # update the reshape to use the temporary buffer as an input
           dnn["operations"][op_idx]["inputs"] = [temp_buf_idx]
           # insert a copy operation that copies from in_buf to temp_buf before
           # the reshape
           self.__insert_copy_operation(dnn, cfg, op_idx, in_buf_idx, temp_buf_idx)
           op_idx += 1
        out_buf_idx = op["outputs"][0]
        out_buf = dnn["buffers"][out_buf_idx]
        if buffer_has_stride(out_buf):
           # create temporary buffer that we can store the reshape output in
           # before copying to out_buf.
           # this buffer is the same as out_buf, except that it has no stride
           # and is only used between the copy and reshape.
          temp_buf = copy.deepcopy(out_buf)
          buffer_remove_stride(temp_buf)
          temp_buf["num_connections"] = 2
          temp_buf["parent"] = -1
          temp_buf_idx = len(dnn["buffers"])
          dnn["buffers"].append(temp_buf)
           # update the reshape to use the temporary buffer as an output
          dnn["operations"][op_idx]["outputs"] = [temp_buf_idx]
          # insert a copy operation that copies from temp_buf to out_buf after
          # the reshape
          self.__insert_copy_operation(dnn, cfg, op_idx+1, temp_buf_idx, out_buf_idx)
          op_idx += 1
      op_idx += 1



  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __pad_buffer(self, dnn, buf, buf_idx, pad_left, pad_right, pad_top, pad_bottom):
    """
    Increase a buffer's dimension to allow space for padding.

    Parameters:
    dnn - the dnn dictionary
    buf - the buffer to add padding to
    buf_idx - the buffer's index
    pad_left, pad_right, pad_top, pad_bottom - the desired padding

    Raises:
    RuntimeError if padding cannot be added
    """
    # Make sure this buffer is not connected to the input of any operations that
    # do not support padding
    for op in dnn["operations"]:
      for idx in op["inputs"]:
        if (idx == buf_idx) and not DNNOperation.supports_cropping(op):
          raise RuntimeError(("Input buffer is also connected to a layer that " +
            "does not support padding on its input (%s). The following layer " + 
            "types don't support padding: %s") % (op["op_id"].name,
            ", ".join([op_id.name for op_id in OPERATIONS_WITHOUT_CROP_SUPPORT])))

    # Get the current padding
    curr_pad_left = buf["data_start_col"]
    curr_pad_right = buf["dimensions"][DIMS_NUM_COLS] - buf["data_num_cols"] - curr_pad_left
    curr_pad_top = buf["data_start_row"]
    curr_pad_bottom = buf["dimensions"][DIMS_NUM_ROWS] - buf["data_num_rows"] - curr_pad_top

    if (pad_left > curr_pad_left):
      buf["dimensions"][DIMS_NUM_COLS] += (pad_left - curr_pad_left)
      buf["data_start_col"] += (pad_left - curr_pad_left)

    if (pad_right > curr_pad_right):
      buf["dimensions"][DIMS_NUM_COLS] += (pad_right - curr_pad_right)

    if (pad_top > curr_pad_top):
      buf["dimensions"][DIMS_NUM_ROWS] += (pad_top - curr_pad_top)
      buf["data_start_row"] += (pad_top - curr_pad_top)

    if (pad_bottom > curr_pad_bottom):
      buf["dimensions"][DIMS_NUM_ROWS] += (pad_bottom - curr_pad_bottom)

  def __postproc_create_input_buffers(self, dnn, config, check_cropping):
    """
    Create buffers for each of the inputs to the post processing operation that
    are each a cropped region of the model's output buffers.

    Parameters:
    dnn - the dnn dictionary, which will have buffers added to it
    config - config dictionary containing input mapping paramaters
    check_cropping - if True, throw an error for invalid cropping

    Returns:
    buffer index for each of the 2 inputs that were created
    
    Raises:
    ValueError if invalid parameters are provided
    """
    # Find and collect the model outputs (outputs with only one connection)
    model_outputs = []
    for op in dnn["operations"]:
      for buf_idx in op["outputs"]:
        buf = dnn["buffers"][buf_idx]
        if buf["num_connections"] == 1:
          model_outputs.append(buf_idx)
    # Create buffer for DNN data
    out_idx = config["POSTPROC_DNN_DATA_OUT_IDX"]
    start_row = config["POSTPROC_DNN_DATA_START_ROW"]
    start_col = config["POSTPROC_DNN_DATA_START_COL"]
    num_rows = config["POSTPROC_DNN_DATA_NUM_ROWS"]
    num_cols = config["POSTPROC_DNN_DATA_NUM_COLS"]
    try:
      in0_idx = self.__postproc_create_buffer(dnn, model_outputs, out_idx, start_row,
                                              start_col, num_rows, num_cols, check_cropping)
    except ValueError as err:
      raise ValueError("Invalid configuration for POSTPROC_DNN_DATA buffer: %s" % err)

    # Create buffer for Compare Values
    out_idx = config["POSTPROC_COMPARE_VALS_OUT_IDX"]
    start_row = config["POSTPROC_COMPARE_VALS_START_ROW"]
    start_col = config["POSTPROC_COMPARE_VALS_START_COL"]
    num_rows = config["POSTPROC_COMPARE_VALS_NUM_ROWS"]
    num_cols = config["POSTPROC_COMPARE_VALS_NUM_COLS"]
    try:
      in1_idx = self.__postproc_create_buffer(dnn, model_outputs, out_idx, start_row,
                                              start_col, num_rows, num_cols, check_cropping)
    except ValueError as err:
      raise ValueError("Invalid configuration for POSTPROC_COMPARE_VALS buffer: %s" % err)
    return in0_idx, in1_idx

  def __postproc_create_buffer(self, dnn, model_outputs, out_idx, start_row,
                               start_col, num_rows, num_cols, check_cropping):
    """
    Create a single buffer for post processing operation that
    is a cropped region of the model's output buffers.

    Parameters:
    dnn - the dnn dictionary, which will have buffers added to it
    model_outputs - list of buffer indices of each model output
    out_idx, start_row, start_col, num_rows, num_cols - configuration parameters
      for this buffer
    check_cropping - if True, throw an error for invalid cropping

    Returns:
    buffer index for the buffer that was created

    Raises:
    ValueError if invalid parameters are provided
    """
    buffer_idx = model_outputs[out_idx]
    buffer = dnn["buffers"][buffer_idx]
    # Check for invalid setup
    if check_cropping:
      if out_idx >= len(model_outputs):
        raise ValueError("Output index is %d, but DNN only has %d outputs!" % (
          out_idx, len(model_outputs)))
      elif (start_row + num_rows) > buffer["data_num_rows"]:
        raise ValueError("Invalid vertical cropping (start row = " +
          "%d, num rows = %d). Output buffer only has %d rows!" % (
            start_row, num_rows, buffer["data_num_rows"]))
      elif (start_col + num_cols) > buffer["data_num_cols"]:
        raise ValueError("Invalid horizontal cropping (start col = " +
          "%d, num cols = %d). Output buffer only has %d columns!" % (
            start_col, num_cols, buffer["data_num_cols"]))

    if (start_row == buffer["data_start_row"]) and (
        start_col == buffer["data_start_col"]) and (
        num_rows == buffer["data_num_rows"]) and (
        num_cols == buffer["data_num_cols"]):
      # If this is a full buffer that already exists, just use that buffer
      in_idx = buffer_idx
    else:
      # Create a new buffer descriptor for the cropped portion of the full buffer
      new_buffer = copy.deepcopy(buffer)
      # If the buffer is already a child of a larger buffer, then this one is also
      # a child of it. Otherwise, this one is a child of the original buffer
      if buffer["parent"] < 0:
        new_buffer["parent"] = buffer_idx
      new_buffer["data_start_row"] = start_row
      new_buffer["data_start_col"] = start_col
      new_buffer["data_num_rows"] = num_rows
      new_buffer["data_num_cols"] = num_cols
      new_buffer["num_connections"] = 1
      in_idx = len(dnn["buffers"])
      dnn["buffers"].append(new_buffer)
    # Add a connection to this buffer
    parent_idx = dnn["buffers"][in_idx]["parent"]
    if parent_idx >= 0:
      dnn["buffers"][parent_idx]["num_connections"] += 1
    dnn["buffers"][in_idx]["num_connections"] += 1
    return in_idx


  def __insert_copy_operation(self, dnn, cfg, op_idx, in_buf_idx, out_buf_idx):
    """
    Insert an operation into the DNN that copies a buffer from one buffer to
    another. This is done using a CONCATENATE layer with only one input.

    Parameters:
    dnn - the dnn dictionary to modify
    cfg - config dictionary
    op_idx - the index of where to insert this operation in dnn["operations"]
    in_buf_idx - buffer index of the copy input
    out_buf_idx - buffer index of the copy output
    """
    op = dict()
    # create a CONCATENATE layer that operates on the channel axis. This effectively
    # copies the buffer over one channel at a time
    op["op_id"] = DNNOperationID.CONCATENATE
    op["parameters"] = dict()
    op["parameters"]["axis"] = -1
    op["parameters"]["scale_ratios"] = []
    op["parameters"]["offsets"] = []
    if cfg["DNN_MODEL"] == "pytorch":
      op["parameters"]["input_format"] = "pytorch"
    else:
      op["parameters"]["input_format"] = "tflite"
    op["inputs"] = [in_buf_idx]
    op["outputs"] = [out_buf_idx]
    # working memory address is filled in later when buffers are allocated
    op["working_mem_addr"] = 0
    # finalize the parameters to fill in scale_ratios, offsets, etc
    DNNOperation.finalize_op_params(dnn, op, cfg["OUTPUT_ENDIANNESS"], cfg["MEMORY_ORDER"])
    # insert this operation at the specified index
    dnn["operations"].insert(op_idx, op)
