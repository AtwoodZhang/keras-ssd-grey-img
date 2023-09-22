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
import math

from internal.constants import *
from internal.utils import *
from internal.operations.dnn_operation import DNNOperation
from internal.operations.dnn_operation import DNNOperationConv2D
from tflite.Padding import Padding

class ConvOptimizer:
  """
  Class that handles optimizing the structure of convolution layers in the
  DNN for IMX firmware.
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self):
    pass

  def split_multilayer_convolutions(self, dnn, config):
    """
    Reduce the amount of RAM needed for convolutions in this DNN by
    partitioning large buffers into smaller buffers and operating on them one
    at a time.

    If a convolution has an output buffer larger than <max_out_size> bytes, it
    is combined with other convolutions after it to create a multi-layer
    convolution. Then, the image is partitioned into <num_partitions> sections
    and each section is processed one at a time. This reduces the maximum amount
    of scratch RAM that is needed for intermediate results at any given time.

    Parameters:
    dnn - the DNN compiler output dictionary to modify
    config - the configuration dictionary

    Raises:
    ValueError if configuration file settings are not valid
    """
    multilayer_start_idx = -1
    multilayer_end_idx = -1
    op_idx = 0
    num_partitions = config["ML_CONV_NUM_PARTITIONS"]
    max_out_size = config["ML_CONV_MAX_OUT_SIZE"]
    # Make sure <num_partitions>  is set to a valid value
    if not num_partitions in VALID_ML_CONV_NUM_PARTITIONS:
      raise ValueError("ML_CONV_NUM_PARTITIONS set to invalid value: " +
        "%d. Supported values are: %s" % (
          num_partitions, ", ".join(VALID_ML_CONV_NUM_PARTITIONS)))

    # Initialize prev_buffer_idx to the main input to the entire network
    inputs = dnn["operations"][0]["inputs"]
    prev_buffer_idx = inputs[CONV_INPUT_IMAGE]
    while op_idx < len(dnn["operations"]):
      op = dnn["operations"][op_idx]
      if multilayer_start_idx < 0:
        # A multi-layer convolution is not currently in progress. See if we
        # should start a new one
        if op["op_id"] in [DNNOperationID.CONV_2D, DNNOperationID.DEPTHWISE_CONV_2D]:
          out_buffer_idx = op["outputs"][0]
          out_buf = dnn["buffers"][out_buffer_idx]
          out_size = out_buf["data_num_rows"]*out_buf["data_num_cols"]
          if out_size > max_out_size:
            multilayer_start_idx = op_idx
      else:
        # A multi-layer convolution is in progress. Check if we've reached a
        # point where we should end it
        if not (op["op_id"] in [DNNOperationID.CONV_2D, DNNOperationID.DEPTHWISE_CONV_2D]):
          # Reached a non-convolution instruction, which can't be part of a
          # multi-layer convolution.
          multilayer_end_idx = op_idx - 1
        elif not (prev_buffer_idx in op["inputs"]):
          # The previous output isn't connected to this input, so we must have
          # reached a different branch of the network that cannot be part of the
          # multi-layer convolution.
          multilayer_end_idx = op_idx - 1
        else:
          out_buffer_idx = op["outputs"][0]
          out_buf = dnn["buffers"][out_buffer_idx]
          out_size = out_buf["data_num_rows"]*out_buf["data_num_cols"]
          if out_size <= max_out_size:
            # The output of this convolution is small enough, end the multi-layer
            # convolution here
            multilayer_end_idx = op_idx
      if op_idx < len(dnn["operations"])-1:
        prev_buffer_idx = op["outputs"][0]
      # If we reached the end of the multi-layer convolution, do the conversion
      # now
      if multilayer_end_idx >= 0:
        if multilayer_end_idx > multilayer_start_idx:
          op_idx = self.__convert_to_multilayer_conv(
            dnn, multilayer_start_idx, multilayer_end_idx, num_partitions)
        else:
          logging.warning(("Convolution at operation index %d" % multilayer_start_idx) +
            " exceeds maximum buffer size, but cannot be converted to multilayer.")
        multilayer_start_idx = -1
        multilayer_end_idx = -1
      else:
        op_idx += 1

  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __convert_to_multilayer_conv(self, dnn, start_idx, end_idx, num_partitions):
    """
    Replace a series of convolution operations with multilayer convolutions

    Parameters:
    dnn - the dnn's dictionary that will be modified
    start_idx - index of the first operation to replace
    end_idx - index of the last operation to replace
    num_partitions - number of partitions to break input buffer into

    Returns:
    index of the next operation after the multilayer convolutions
    """
    num_layers = end_idx - start_idx + 1
    num_partitions_per_axis = int(math.sqrt(num_partitions))
    logging.info(("Converting %d convolutions starting at operation %d" % (num_layers, start_idx)) +
      " to multilayer convolution")

    # Get the input buffer for the first operation
    inputs = dnn["operations"][start_idx]["inputs"]
    in_buffer_idx = inputs[CONV_INPUT_IMAGE]
    in_buf_full = dnn["buffers"][in_buffer_idx]
    # Get the output buffer for the final operation
    out_buffer_idx = dnn["operations"][end_idx]["outputs"][0]
    out_buf_full = dnn["buffers"][out_buffer_idx]
    # Loop through partitions
    new_ops = []
    for y in range(0, num_partitions_per_axis):
      for x in range(0, num_partitions_per_axis):
        # Create new output buffer that is a portion of the full output buffer
        out_buf = self.__create_partition(dnn, out_buffer_idx, x, y, num_partitions_per_axis)
        prev_buffer_idx = len(dnn["buffers"])
        dnn["buffers"].append(out_buf)
        # Create a new operation for each layer, working backwards
        for layer in range(num_layers-1, -1, -1):
          full_op = dnn["operations"][start_idx + layer]
          op = copy.deepcopy(full_op)
          # Replace the output with the last buffer that was created,
          # which will be the previous operation's input
          op["outputs"][0] = prev_buffer_idx
          # Create a new buffer for the input. The first layer's input is a
          # partition of the full input buffer, while all other layers have
          # their own standalone input buffers.
          if layer == 0:
            in_buf = self.__create_partition(
              dnn, in_buffer_idx, x, y, num_partitions_per_axis)
          else:
            in_idx = op["inputs"][0]
            in_buf = copy.deepcopy(dnn["buffers"][in_idx])
          # Increase the size of the buffer to ensure that there is enough data
          # processed to produced the necessary output size
          in_width, in_height = self.__calc_input_data_size(dnn, op, full_op, x, y, num_partitions_per_axis)
          if layer == 0:
            # If this is a partition of a larger buffer, move/resize the data window
            width_delta = in_width - in_buf["data_num_cols"]
            height_delta = in_height - in_buf["data_num_rows"]
            self.__adjust_data_region(in_buf, width_delta, height_delta, x, y, num_partitions_per_axis)
          else:
            # Otherwise, just increase the buffer size to the necessary dimensions
            in_buf["data_num_cols"] = in_width
            in_buf["data_num_rows"] = in_height
          # Recalculate how much padding is needed
          DNNOperation.set_padding(dnn, op)
          if layer != 0:
            # If this is a stand-alone buffer, add padding around the data region
            pad_top = op["parameters"]["pad_top"]
            pad_bottom = op["parameters"]["pad_bottom"]
            pad_left = op["parameters"]["pad_left"]
            pad_right = op["parameters"]["pad_right"]
            in_buf["data_start_row"] = pad_top
            in_buf["data_start_col"] = pad_left
            in_buf["dimensions"][DIMS_NUM_ROWS] = in_buf["data_num_rows"] + pad_top + pad_bottom
            in_buf["dimensions"][DIMS_NUM_COLS] = in_buf["data_num_cols"] + pad_left + pad_right
          # On all inner boundaries (the boundaries between partitions), remove
          # padding and replace it with overlapping image content
          self.__replace_padding_with_overlap(op, in_buf, x, y, num_partitions_per_axis)
          # Add the new buffer and operation to the DNN
          prev_buffer_idx = len(dnn["buffers"])
          dnn["buffers"].append(in_buf)
          op["inputs"][CONV_INPUT_IMAGE] = prev_buffer_idx
          new_ops.insert(0, op)
    # Replace a single connection on each buffer with one for every partition
    in_buf_full["num_connections"] += num_partitions - 1
    out_buf_full["num_connections"] += num_partitions - 1
    # Replace the original operations with the new operations
    del dnn["operations"][start_idx:end_idx+1]
    dnn["operations"][start_idx:start_idx] = new_ops
    return start_idx + len(new_ops)
 
  def __create_partition(self, dnn, full_buf_idx, x, y, num_per_axis):
    """
    Create a new buffer that is a partition of an existing full buffer

    Parameters:
    dnn - the dnn dictionary
    full_buf_idx - the full buffer's index
    x - the x index of this partition
    y - the y index of this partition
    num_per_axis - number of partitions per axis

    Returns:
    The new buffer dictionary
    """
    full_buf = dnn["buffers"][full_buf_idx]
    buf = copy.deepcopy(full_buf)
    buf["parent"] = full_buf_idx
    buf["data_num_rows"] = int(math.ceil(full_buf["data_num_rows"] / num_per_axis))
    buf["data_num_cols"] = int(math.ceil(full_buf["data_num_cols"] / num_per_axis))
    buf["data_start_row"] += int(math.floor(full_buf["data_num_rows"] / num_per_axis * y))
    buf["data_start_col"] += int(math.floor(full_buf["data_num_cols"] / num_per_axis * x))
    return buf

  def __calc_input_data_size(self, dnn, op, full_op, x, y, num_partitions_per_axis):
    """
    Calculate how much input data needs to be processed for a convolution based
    on the filter and output buffer sizes.  Padding and ignored data will be added to the
    buffer's size

    Parameters:
    dnn - the dnn dictionary
    op - the convolution operation dictionary
    full_op - the original full convolution operation dictionary (before spliting up)
    x - x coordinate of this partition of the multilayer convolution
    y - y coordinate of this partition of the multilayer convolution
    num_partitions_per_axis - number of partitions on each axis of multilayer convolution

    Returns:
    in_width, in_height - data size of the input
    """
    # Get buffer sizes
    out_buf = dnn["buffers"][op["outputs"][0]]
    filt_buf = dnn["buffers"][op["inputs"][CONV_INPUT_WEIGHTS]]
    out_width = out_buf["data_num_cols"]
    out_height = out_buf["data_num_rows"]
    filt_width = filt_buf["data_num_cols"]
    filt_height = filt_buf["data_num_rows"]
    # Determine how much data needs to be processed to create the necessary output
    # buffer size
    in_width, in_height = DNNOperationConv2D.get_input_size(
      op, out_width, out_height, filt_width, filt_height)
    # Count how many rows/columns of data are ignored in the full operation
    full_in_buf = dnn["buffers"][full_op["inputs"][CONV_INPUT_IMAGE]]
    [ignore_l, ignore_r, ignore_t, ignore_b] = self.__get_ignored_data(
      dnn, full_op, full_in_buf["data_num_cols"], full_in_buf["data_num_rows"])
    # Add any ignored rows/columns to the input buffer
    if ignore_l > 0 and x == 0:
      in_width += ignore_l
    if ignore_r > 0 and x == num_partitions_per_axis-1:
      in_width += ignore_r
    if ignore_t > 0 and y == 0:
      in_height += ignore_t
    if ignore_b > 0 and y == num_partitions_per_axis-1:
      in_height += ignore_b
    return in_width, in_height

  def __adjust_data_region(self, in_buf, width_delta, height_delta, x, y, num_partitions_per_axis):
    """
    Change the size of the data region in a partition buffer, and move the data
    window accordingly

    Parameters:
    in_buf - the buffer to resize
    width_delta - the amount to change the width
    height_delta - the amount to change the height
    x - x coordinate of this partition of the multilayer convolution
    y - y coordinate of this partition of the multilayer convolution
    num_partitions_per_axis - number of partitions on each axis of multilayer convolution
    """
    if width_delta > 0:
      in_buf["data_num_cols"] += width_delta
      if x == 0:
        # Add all overlap on the RIGHT side
        in_buf["data_start_col"] -= 0
      elif x == num_partitions_per_axis-1:
        # Add all overlap on the LEFT side
        in_buf["data_start_col"] -= width_delta
      else:
        # Add half of the overlap on each side
        in_buf["data_start_col"] -= math.floor(width_delta/2)
    if height_delta > 0:
      in_buf["data_num_rows"] += height_delta
      if y == 0:
        # Add all overlap on the BOTTOM
        in_buf["data_start_row"] -= 0
      elif y == num_partitions_per_axis-1:
        # Add all overlap on the TOP
        in_buf["data_start_row"] -= height_delta
      else:
        # Add half of the overlap on each side
        in_buf["data_start_row"] -= math.floor(height_delta/2)

  def __get_ignored_data(self, dnn, op, in_width, in_height):
    """
    Count how many rows and columns of data should be ignored for a given
    convolution operation

    Parameters:
    dnn - the full dnn dictionary
    op - the operation
    in_width - input data width
    in_height - input data height

    Returns:
    [ignored_left, ignored_right, ignored_top, ignored_bottom]
    """
    if (op["parameters"]["padding_type"] == Padding.SAME):
      return 0, 0, 0, 0

    in_buf = dnn["buffers"][op["inputs"][CONV_INPUT_IMAGE]]
    filt_buf = dnn["buffers"][op["inputs"][CONV_INPUT_WEIGHTS]]
    out_buf = dnn["buffers"][op["outputs"][0]]

    # Get buffer sizes (including input padding)
    filt_width = filt_buf["data_num_cols"]
    filt_height = filt_buf["data_num_rows"]
    out_width = out_buf["data_num_cols"]
    out_height = out_buf["data_num_rows"]

    # Determine how many input pixels are actually processed, and how many are ignored
    proc_in_width, proc_in_height = DNNOperationConv2D.get_input_size(op, out_width, out_height, filt_width, filt_height)
    ignored_cols = max(0, in_width - proc_in_width)
    ignored_rows = max(0, in_height - proc_in_height)

    # Split ignored rows/columns evenly between left and right side
    ignored_left = math.floor(ignored_cols/2)
    ignored_right = ignored_cols - ignored_left
    ignored_top = math.floor(ignored_rows/2)
    ignored_bottom = ignored_rows - ignored_top
    return ignored_left, ignored_right, ignored_top, ignored_bottom

  def __replace_padding_with_overlap(self, op, in_buf, x, y, num_partitions_per_axis):
    """
    On all inner boundaries of a multilayer convolution partition, replace padding
    rows/columns with actual data rows/columns that overlap other partitions.

    Parameters:
    op - the convoltuion operation to update
    in_buf - the input buffer to update
    x - x coordinate of this partition of the multilayer convolution
    y - y coordinate of this partition of the multilayer convolution
    num_partitions_per_axis - number of partitions on each axis of multilayer convolution
    """
    if x != 0:
      in_buf["data_start_col"] -= op["parameters"]["pad_left"]
      in_buf["data_num_cols"] += op["parameters"]["pad_left"]
      op["parameters"]["pad_left"] = 0
    if x != num_partitions_per_axis-1:
      in_buf["data_num_cols"] += op["parameters"]["pad_right"]
      op["parameters"]["pad_right"] = 0
    if y != 0:
      in_buf["data_start_row"] -= op["parameters"]["pad_top"]
      in_buf["data_num_rows"] += op["parameters"]["pad_top"]
      op["parameters"]["pad_top"] = 0
    if y != num_partitions_per_axis-1:
      in_buf["data_num_rows"] += op["parameters"]["pad_bottom"]
      op["parameters"]["pad_bottom"] = 0
