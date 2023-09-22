# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import copy
import math
import numpy as np

from tflite.BuiltinOperator import BuiltinOperator
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding

import torch

from internal.constants import *
from internal.utils import *

# Minimum and maximum values of parameters
if DNN_COMPILER_INTERNAL_USE:
  PARAMETER_STRIDE_MIN = 1
  PARAMETER_STRIDE_MAX = 15
  PARAMETER_FILT_SIZE_MIN = 1
  PARAMETER_FILT_SIZE_MAX = 15
  PARAMETER_PADDING_MIN = 0
  PARAMETER_PADDING_MAX = 15
else:
  PARAMETER_STRIDE_MIN = 1
  PARAMETER_STRIDE_MAX = 3
  PARAMETER_FILT_SIZES = [1, 3, 5]
  PARAMETER_PADDING_MIN = 0
  PARAMETER_PADDING_MAX = 15

class DNNOperationConv2D:
  """
  Operation-specific tasks for CONV2D operations. See dnn_operation.py
  for details on what each function does.
  """

  def get_op_params_tflite(self, op_dict, graph, op, opcode):
    parameters = dict()
    # Initialize clip_max based on the activation type
    if opcode == BuiltinOperator.CONV_2D:
      options = Conv2DOptions()
    else:
      options = DepthwiseConv2DOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    if (options.FusedActivationFunction() == ActivationFunctionType.NONE):
      parameters["relu_clip_max"] = RELU_NONE
    elif (options.FusedActivationFunction() == ActivationFunctionType.RELU):
      parameters["relu_clip_max"] = RELU_NO_MAX_CLIP
    elif (options.FusedActivationFunction() == ActivationFunctionType.RELU6):
      parameters["relu_clip_max"] = 6
    else:
      err = "Unsupported activation function: %s." % (
        flatbuf_enum_to_str(option.FusedActivationFunction(), ActivationFunctionType))
      err += "Supported values are [NONE, RELU, RELU6]."
      raise RuntimeError(err)

    parameters["stride_w"] = options.StrideW()
    parameters["stride_h"] = options.StrideH()
    parameters["mode"] = ConvMode.DEFAULT
    parameters["padding_type"] = options.Padding()
    parameters["source"] = "tflite"
    op_dict["parameters"] = parameters

    # Check that invalid parameters are not provided
    if options.DilationWFactor() != 1:
      raise RuntimeError("Unsupported parameter: DilationWFactor=%d. Only supported value is 1." % (
        options.DilationWFactor()))
    if options.DilationHFactor() != 1:
      raise RuntimeError("Unsupported parameter: DilationHFactor=%d. Only supported value is 1." % (
        options.DilationHFactor()))    


  def get_op_params_pytorch(self, op_dict, module, node):
    parameters = dict()
    if node.kind() == "quantized::conv2d_relu":
      parameters["relu_clip_max"] = RELU_NO_MAX_CLIP
    else:
      parameters["relu_clip_max"] = RELU_NONE

    parameters["stride_w"] = module.stride[1]
    parameters["stride_h"] = module.stride[0]
    parameters["mode"] = ConvMode.DEFAULT
    parameters["source"] = "pytorch"
    if module.padding[0] == 0 and module.padding[1] == 0:
      parameters["padding_type"] = Padding.VALID
    else:
      parameters["padding_type"] = Padding.SAME

    if module.groups == module.in_channels:
      op_dict["op_id"] = DNNOperationID.DEPTHWISE_CONV_2D
    elif module.groups == 1:
        op_dict["op_id"] = DNNOperationID.CONV_2D
    else:
      raise RuntimeError("Unsupported parameter: groups = %d. Must be either 1 (standard convolution) or equal to "
                         "in_channels (depthwise convolution)." % module.groups)

    # Check that invalid parameters are not provided
    if module.dilation[0] != 1:
      raise RuntimeError("Unsupported parameter: dilation (H). Only supported value is 1." % (
        module.dilation[0]))
    if module.dilation[1] != 1:
      raise RuntimeError("Unsupported parameter: dilation (W). Only supported value is 1." % (
        module.dilation[1]))
    if module.padding_mode != "zeros":
      raise RuntimeError("Unsupported parameter: padding_mode. Only supported value is \'zeros\'." % (
        module.padding_mode))

    op_dict["parameters"] = parameters

  def finalize_op_params(self, dnn, op_dict, endianness, mem_order):
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    quant_start_idx = out_buf["quant_start_idx"]
  
    quant_scale = dnn["quant_params"][quant_start_idx][0]
    quant_zero  = dnn["quant_params"][quant_start_idx][1]

    filt_buf_idx = op_dict["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    
    # CONV2D actually uses a set of quantization parameters per filter BATCH,
    # not channel.
    if op_dict["op_id"] == DNNOperationID.CONV_2D:

      if filt_buf["quant_type"] == QuantType.PER_CHANNEL:
        filt_buf["quant_type"] = QuantType.PER_BATCH

    # Quantize the relu max
    relu_clip_max = op_dict["parameters"]["relu_clip_max"]
    if (relu_clip_max != RELU_NONE) and (relu_clip_max != RELU_NO_MAX_CLIP ):
        relu_clip_max = min(127, max(-128, int(round(relu_clip_max / quant_scale + quant_zero))))
        op_dict["parameters"]["relu_clip_max"] = relu_clip_max
    if op_dict["op_id"] == DNNOperationID.DEPTHWISE_CONV_2D:
      self.__update_bias_terms(op_dict, dnn, endianness, mem_order, True)
    else:
      self.__update_bias_terms(op_dict, dnn, endianness, mem_order, False)

    # Check that parameters are valid
    filt_dims = filt_buf["dimensions"]
    check_param_range("stride_w", op_dict["parameters"]["stride_w"], PARAMETER_STRIDE_MIN, PARAMETER_STRIDE_MAX)
    check_param_range("stride_h", op_dict["parameters"]["stride_h"], PARAMETER_STRIDE_MIN, PARAMETER_STRIDE_MAX)
    if DNN_COMPILER_INTERNAL_USE:
      check_param_range("filter_width", filt_dims[DIMS_NUM_COLS], PARAMETER_FILT_SIZE_MIN, PARAMETER_FILT_SIZE_MAX)
      check_param_range("filter_height", filt_dims[DIMS_NUM_ROWS], PARAMETER_FILT_SIZE_MIN, PARAMETER_FILT_SIZE_MAX)
    else:
      check_param_valid("filter_width", filt_dims[DIMS_NUM_COLS], PARAMETER_FILT_SIZES)
      check_param_valid("filter_height", filt_dims[DIMS_NUM_ROWS], PARAMETER_FILT_SIZES)
      if op_dict["parameters"]["stride_w"] != op_dict["parameters"]["stride_h"]:
        raise RuntimeError("stride_w (%d) and stride_h (%d) must be equal." % (
          op_dict["parameters"]["stride_w"], op_dict["parameters"]["stride_h"]))
      if filt_dims[DIMS_NUM_COLS] != filt_dims[DIMS_NUM_ROWS]:
        raise RuntimeError("filter_width (%d) and filter_height (%d) must be equal." % (
          filt_dims[DIMS_NUM_COLS], filt_dims[DIMS_NUM_ROWS]))

    check_param_range("pad_left", op_dict["parameters"]["pad_left"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("pad_right", op_dict["parameters"]["pad_right"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("pad_top", op_dict["parameters"]["pad_top"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    check_param_range("pad_bottom", op_dict["parameters"]["pad_bottom"], PARAMETER_PADDING_MIN, PARAMETER_PADDING_MAX)
    
  def op_params_to_byte_array(self, op_dict, dnn, endianness):
    bytes = []
    if endianness == Endianness.BIG:
      val = (op_dict["parameters"]["pad_right"] & 0xF)
      val |= (op_dict["parameters"]["pad_left"] & 0xF) << 4
      bytes.extend(val_to_uint8_array(val, 1, endianness))

      val = (op_dict["parameters"]["pad_bottom"] & 0xF)
      val |= (op_dict["parameters"]["pad_top"] & 0xF) << 4
      bytes.extend(val_to_uint8_array(val, 1, endianness))

      val = (op_dict["parameters"]["mode"].value & 0xF)
      val |= (op_dict["parameters"]["stride_h"] & 0x3) << 4
      val |= (op_dict["parameters"]["stride_w"] & 0x3) << 6
      bytes.extend(val_to_uint8_array(val, 1, endianness))

      val = op_dict["parameters"]["relu_clip_max"] & 0xFF
      bytes.extend(val_to_uint8_array(val, 1, endianness))
    else:
      val = (op_dict["parameters"]["pad_right"] & 0xF) << 4
      val |= (op_dict["parameters"]["pad_left"] & 0xF)
      bytes.extend(val_to_uint8_array(val, 1, endianness))

      val = (op_dict["parameters"]["pad_bottom"] & 0xF) << 4
      val |= (op_dict["parameters"]["pad_top"] & 0xF)
      bytes.extend(val_to_uint8_array(val, 1, endianness))

      val = (op_dict["parameters"]["mode"].value & 0xF) << 4
      val |= (op_dict["parameters"]["stride_h"] & 0x3) << 2
      val |= (op_dict["parameters"]["stride_w"] & 0x3)
      bytes.extend(val_to_uint8_array(val, 1, endianness))

      val = op_dict["parameters"]["relu_clip_max"] & 0xFF
      bytes.extend(val_to_uint8_array(val, 1, endianness))


    if op_dict["op_id"] == DNNOperationID.DEPTHWISE_CONV_2D:
      self.__calc_and_store_chan_mult(bytes, op_dict, dnn, endianness)
    elif op_dict["parameters"]["mode"] == ConvMode.OPTIMIZED_1X1_FILTER:
      self.__calc_and_store_dummy_cols(bytes, op_dict, dnn, endianness)
    self.__calc_and_store_scale_ratios(bytes, op_dict, dnn, endianness)
    return add_to_misc_data(dnn["misc_data"], bytes)

  def op_params_to_string(self, dnn, op_dict):
    params = op_dict["parameters"]
    string = "pad=(L:%d, R:%d, T:%d, B:%d), stride=(H:%d, W:%d), mode=%s, clip_max=%d" % ( 
      params["pad_left"], params["pad_right"], params["pad_top"], params["pad_bottom"],
      params["stride_h"], params["stride_w"], 
      params["mode"].name, params["relu_clip_max"])
    return string


  def get_working_memory_size(self, dnn, op_dict, config, ram_available):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    filt_buf_idx = op_dict["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]

    # Get buffer sizes
    in_cols = in_buf["data_num_cols"] + op_dict["parameters"]["pad_left"] + op_dict["parameters"]["pad_right"]
    in_rows = in_buf["data_num_rows"] + op_dict["parameters"]["pad_top"] + op_dict["parameters"]["pad_bottom"]
    in_size = in_cols * in_rows
    filt_size = filt_buf["data_num_rows"]*filt_buf["data_num_cols"]
    out_size = out_buf["data_num_cols"] * out_buf["data_num_rows"]
    if op_dict["op_id"] == DNNOperationID.DEPTHWISE_CONV_2D:
      in_chan = 1
    else:
      in_chan = in_buf["dimensions"][DIMS_NUM_CHANNELS]
    out_chan = out_buf["dimensions"][DIMS_NUM_CHANNELS]

    # Check if we meet criteria to use the optimized 1x1 filter implementation
    if (
      DNNOperationConv2D.check_opt1x1_criteria(op_dict, dnn)) and (
      in_buf["dimensions"][DIMS_NUM_COLS] == out_buf["dimensions"][DIMS_NUM_COLS]):
      # This implementation requires two int32 temporary buffers that are the full
      # output size, out_rows x out_cols x out_chan, including horizontal padding
      full_out_size = out_buf["data_num_rows"] * out_buf["dimensions"][DIMS_NUM_COLS]
      mem_size =  2 * 4 * full_out_size * out_chan
      if mem_size <= ram_available:
        if config["MEMORY_ORDER"] == MemoryOrder.CHANNEL_FIRST:
          # For this mode, we need the filter to be stored as <in_chan> x <out_chan>
          # instead of <out_chan> x <in_chan>.
          self.__transpose_filter(dnn, op_dict)
        if filt_buf["quant_type"] == QuantType.PER_BATCH:
          filt_buf["quant_type"] = QuantType.PER_CHANNEL
        op_dict["parameters"]["mode"] = ConvMode.OPTIMIZED_1X1_FILTER


    # If we were unable to use an optimized implementation, determine the memory
    # needed for the default implementation
    if op_dict["parameters"]["mode"] == ConvMode.DEFAULT:
      # The default implemenation requires two temporary buffers of int32 values.
      # For a depthwise convolution, both need to be the size of the output, while
      # the standard convolution requires one to be the size of the input and one
      # to be the size of the output.
      if op_dict["op_id"] == DNNOperationID.DEPTHWISE_CONV_2D:
        mem_size = 2 * 4 * out_size
      else:
        mem_size = 4 * in_size + 4 * out_size
      
    return mem_size

  def get_processing_time(self, dnn, op_dict, instr_cnts, instr_cycles):
    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    filt_buf_idx = op_dict["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]

    in_cols = in_buf["data_num_cols"] + op_dict["parameters"]["pad_left"] + op_dict["parameters"]["pad_right"]
    in_rows = in_buf["data_num_rows"] + op_dict["parameters"]["pad_top"] + op_dict["parameters"]["pad_bottom"]
    in_size = in_cols * in_rows
    in_batches = in_buf["dimensions"][DIMS_NUM_BATCHES]
    out_size = out_buf["data_num_cols"]*out_buf["data_num_rows"]
    filt_size = filt_buf["data_num_cols"]*filt_buf["data_num_rows"]

    if op_dict["op_id"] == DNNOperationID.CONV_2D:
      in_channels = filt_buf["dimensions"][DIMS_NUM_CHANNELS]
      out_channels = filt_buf["dimensions"][DIMS_NUM_BATCHES]
    else:  # depthwise conv2d
      out_channels = filt_buf["dimensions"][DIMS_NUM_CHANNELS]
      in_channels = 1

    cycles = 0
    if op_dict["parameters"]["mode"] == ConvMode.DEFAULT:
      # VMOVs
      pad_left = op_dict["parameters"]["pad_left"] * in_buf["dimensions"][DIMS_NUM_CHANNELS]
      pad_right = op_dict["parameters"]["pad_right"] * in_buf["dimensions"][DIMS_NUM_CHANNELS]
      pad_top = op_dict["parameters"]["pad_top"]
      pad_bottom = op_dict["parameters"]["pad_bottom"]
      total_rows = in_buf["data_num_rows"] + pad_top + pad_right
      total_cols = in_buf["data_num_cols"] * in_buf["dimensions"][DIMS_NUM_CHANNELS] + pad_left + pad_right
      num_rows = total_rows - (pad_top + pad_bottom + 1)
      num_cols = pad_left + pad_right
      if pad_top > 0:
        cycles += record_instruction("VMOV_C (Scalar 0)", in_batches, pad_top*total_cols, instr_cnts, instr_cycles)
      if pad_bottom > 0:
        cycles += record_instruction("VMOV_C (Scalar 0)", in_batches, pad_bottom*total_cols, instr_cnts, instr_cycles)
      if pad_left > 0:
        cycles += record_instruction("VMOV_C (Scalar 0)", in_batches, pad_left*total_rows, instr_cnts, instr_cycles)
      if pad_right > 0:
        cycles += record_instruction("VMOV_C (Scalar 0)", in_batches, pad_right*total_rows, instr_cnts, instr_cycles)

      for i in range(0, out_channels):
        if op_dict["op_id"] == DNNOperationID.CONV_2D:
          cycles += record_instruction("VCACCL_C", in_batches, in_size*in_channels, instr_cnts, instr_cycles)
          if filt_size > 1:
            cycles += record_instruction("VCACC_L", in_batches, out_size*filt_size, instr_cnts, instr_cycles)
          else:
            cycles += record_instruction("VCONVBIAS_L (Scalar 1)", in_batches, out_size, instr_cnts, instr_cycles)
        else:
          cycles += record_instruction("VCACCL_C", in_batches, out_size*filt_size, instr_cnts, instr_cycles)
        if filt_size > 1:
          cycles += record_instruction("VCONVBIASL_C", in_batches, out_size*filt_size, instr_cnts, instr_cycles)
          cycles += record_instruction("VMADD_L (Scalar 1)", in_batches, out_size, instr_cnts, instr_cycles)
        else:
          cycles += record_instruction("VCONVL_C", in_batches, out_size*filt_size, instr_cnts, instr_cycles)
          cycles += record_instruction("VADD_L", in_batches, out_size, instr_cnts, instr_cycles)         
        if in_channels > 1:
          cycles += record_instruction("VCONVL_C", in_batches*(in_channels-1), out_size*filt_size, instr_cnts, instr_cycles)
          cycles += record_instruction("VADD_L", in_batches*(in_channels-1), out_size, instr_cnts, instr_cycles)
        cycles += record_instruction("VMADDC_L (Scalar 1,2)", in_batches, out_size, instr_cnts, instr_cycles)

        if op_dict["parameters"]["relu_clip_max"] == RELU_NO_MAX_CLIP:
          cycles += record_instruction("VMAX_C", in_batches, out_size, instr_cnts, instr_cycles)
        elif op_dict["parameters"]["relu_clip_max"] != RELU_NONE:
          cycles += record_instruction("VCLAMP_C", in_batches, out_size, instr_cnts, instr_cycles)
    else:  # op_dict["parameters"]["mode"] == ConvMode.OPTIMIZED_1X1_FILTER
      full_out_size = out_buf["data_num_rows"] * out_buf["dimensions"][DIMS_NUM_COLS]
      cycles += record_instruction("VMML_C (Scalar 1)", in_batches, (full_out_size*in_channels*out_channels), instr_cnts, instr_cycles)
      if full_out_size > 20:
        cycles += record_instruction("VMMBIASL_C", in_batches*out_channels, (full_out_size*in_channels*1), instr_cnts, instr_cycles)
      else:
        cycles += record_instruction("VMML_C", in_batches, (full_out_size*in_channels*out_channels), instr_cnts, instr_cycles)
        cycles += record_instruction("VADD_L", in_batches, (full_out_size*out_channels), instr_cnts, instr_cycles)
      cycles += record_instruction("VMADD_L (Scalar 1)", in_batches, (full_out_size*out_channels), instr_cnts, instr_cycles)
      cycles += record_instruction("VMADDC_L (Scalar 1,2)", in_batches, (full_out_size*out_channels), instr_cnts, instr_cycles)
      if op_dict["parameters"]["relu_clip_max"] == RELU_NO_MAX_CLIP:
        cycles += record_instruction("VMAX_C", in_batches, (full_out_size*out_channels), instr_cnts, instr_cycles)
      elif op_dict["parameters"]["relu_clip_max"] != RELU_NONE:
        cycles += record_instruction("VCLAMP_C", in_batches, (full_out_size*out_channels), instr_cnts, instr_cycles)
    return 1000*VPU_CLOCK_PERIOD_S*cycles
  
  @staticmethod
  def set_padding(dnn, op_dict):
      # Get the input image's size
      in_idx = op_dict["inputs"][CONV_INPUT_IMAGE]
      in_buf = dnn["buffers"][in_idx]
      in_width = in_buf["data_num_cols"]
      in_height = in_buf["data_num_rows"]

      # Get the filter's size
      filt_idx = op_dict["inputs"][CONV_INPUT_WEIGHTS]
      filt_buf = dnn["buffers"][filt_idx]
      filter_width = filt_buf["data_num_cols"]
      filter_height = filt_buf["data_num_rows"]

      if (op_dict["parameters"]["padding_type"] == Padding.SAME):
        stride_w = op_dict["parameters"]["stride_w"]
        stride_h = op_dict["parameters"]["stride_h"]
        # If the stride is wider than the filter, treat the stride as the filter
        # size for all padding calculations
        if stride_w > filter_width:
          filter_width = stride_w
        if stride_h > filter_height:
          filter_height = stride_h
        # Determine the total padding needed in each direction
        horizontal_padding = filter_width - stride_w
        vertical_padding = filter_height - stride_h
        # Add padding to ensure the output size is an integer
        out_width = in_width - filter_width + horizontal_padding
        if (out_width % stride_w) != 0:
          horizontal_padding += stride_w - (out_width % stride_w)
        out_height = in_height - filter_height + vertical_padding
        if (out_height % stride_h) != 0:
          vertical_padding += stride_h - (out_height % stride_h)

        # Determine padding on each side of the image
        pad_min = math.floor(horizontal_padding/2)
        pad_max = horizontal_padding - pad_min
        if op_dict["parameters"]["source"] == "tflite":
          op_dict["parameters"]["pad_left"] = pad_min
          op_dict["parameters"]["pad_right"] = pad_max
        else:
          op_dict["parameters"]["pad_left"] = pad_max
          op_dict["parameters"]["pad_right"] = pad_min

        pad_min = math.floor(vertical_padding/2)
        pad_max = vertical_padding - pad_min
        if op_dict["parameters"]["source"] == "tflite":
          op_dict["parameters"]["pad_top"] = pad_min
          op_dict["parameters"]["pad_bottom"] = pad_max
        else:
          op_dict["parameters"]["pad_top"] = pad_max
          op_dict["parameters"]["pad_bottom"] = pad_min

      elif (op_dict["parameters"]["padding_type"] == Padding.VALID):
        op_dict["parameters"]["pad_left"] = 0
        op_dict["parameters"]["pad_right"] = 0
        op_dict["parameters"]["pad_top"] = 0
        op_dict["parameters"]["pad_bottom"] = 0
      else:
        raise RuntimeError("Unrecognized padding: " + str(op_dict["parameters"]["padding_type"]))


  @staticmethod
  def __internal_check_opt1x1_criteria(op_dict, dnn):
    """
    Internal implementation to check that an operation could use the 
    1x1 filter optimization.
    
    This explicitly does not check the "do_not_1x1_optimize" op field 
    that explicitly disables 1x1 optimizations so that this can be used
    to set that field during initialization.
    """
    if op_dict["op_id"] != DNNOperationID.CONV_2D:
      return False

    in_buf_idx = op_dict["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    filt_buf_idx = op_dict["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    out_buf_idx = op_dict["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]

    filt_size = filt_buf["data_num_cols"]*filt_buf["data_num_rows"]
    if (
        filt_size == 1) and (
        filt_buf["quant_type"] == QuantType.PER_TENSOR) and (
        op_dict["parameters"]["stride_w"] == 1) and (
        op_dict["parameters"]["stride_h"] == 1) and (
        in_buf["parent"] < 0) and (
        out_buf["parent"] < 0):
      return True
    else:
      return False


  @staticmethod
  def check_opt1x1_criteria(op_dict, dnn):
    """
    Check if a convolution may be able to use the optimized 1x1 filter implementation
    """
    return ( DNNOperationConv2D.__internal_check_opt1x1_criteria(op_dict, dnn) 
             and not "do_not_1x1_optimize" in op_dict )


  @staticmethod
  def pre_check_opt1x1_criteria(dnn: dict, skip_count: int) -> int:
    """
    Premark dnn operations so that the first `skip_count` operations that
    could use the 1x1 optimization are marked so that they don't get
    optimized later.

    Parameters:
    dnn        - dnn model to process
    skip_count - number of operations to skip 

    Return:
    The number of potential 1x1 optimizations that are still enabled.
    """
    still_enabled = 0

    for op in dnn["operations"]:
      if DNNOperationConv2D.__internal_check_opt1x1_criteria(op, dnn):
        if skip_count > 0:
          skip_count -= 1
          op["do_not_1x1_optimize"] = True
        else:
          still_enabled += 1
    
    return still_enabled


  @staticmethod
  def get_input_size(op_dict, out_width, out_height, filter_width, filter_height):
    """
    Get the input buffer's size based on the output buffer's size
    
    Parameters:
    op_dict - the operation dictionary
    out_width/height - dimensions for input buffer
    filter_width/height - dimensions for filter buffer

    Returns:
    in_width, in_height
    """
    if (op_dict["parameters"]["padding_type"] == Padding.SAME):
      in_width = out_width * op_dict["parameters"]["stride_w"]
      in_height = out_height * op_dict["parameters"]["stride_h"]
    elif (op_dict["parameters"]["padding_type"] == Padding.VALID):
      in_width = out_width * op_dict["parameters"]["stride_w"]
      in_height = out_height * op_dict["parameters"]["stride_h"]
      in_width += (filter_width - op_dict["parameters"]["stride_w"])
      in_height += (filter_height - op_dict["parameters"]["stride_h"])
    else:
      raise RuntimeError("Unrecognized padding: " + str(op_dict["parameters"]["padding_type"]))
    return in_width, in_height

  def check_for_patches(self, dnn, op_dict):
    patches = []
    if dnn["cfg"]["SENSOR_VERSION"].value == "ES1":
      in_buf = dnn["buffers"][op_dict["inputs"][0]]
      filt_buf = dnn["buffers"][op_dict["inputs"][1]]

      if in_buf["dimensions"][DIMS_NUM_BATCHES] > 1:
        # Patch 170940_177827 is necessary if there are multiple batches and
        # one of the following is true:
        #  - Layer type is CONV2D
        #  - Padding is required
        #  - Per-channel quantization is used
        if (
          op_dict["op_id"] == DNNOperationID.CONV_2D) or (
          op_dict["parameters"]["pad_left"] > 0) or (
          op_dict["parameters"]["pad_right"] > 0) or (
          op_dict["parameters"]["pad_top"] > 0) or (
          op_dict["parameters"]["pad_bottom"] > 0) or (
          filt_buf["quant_type"] != QuantType.PER_TENSOR):
            patches.append("170940_177827")

    return patches

  def adjust_for_patches(self, dnn, op_dict, patches):
    pass

  # ============================================================================
  # Private methods
  # ============================================================================
  def __transpose_filter(self, dnn, op):
    """
    Transpose a convolution's filter data in the static data array. If possible,
    this is done in place. If not, a new buffer is created.

    Parameters:
    dnn - the dnn dictionary
    op - the current operation's dictionary
    """

    # Get filter buffer
    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]

    # Assume the buffer is N x 1 x 1 x C and interpret it as a 2D matrix: N x C
    filt_rows = filt_buf["dimensions"][DIMS_NUM_BATCHES]
    filt_cols = filt_buf["dimensions"][DIMS_NUM_CHANNELS]

    # Get the data
    start_idx = filt_buf["start_idx"]
    data_len = filt_rows * filt_cols
    in_data = dnn["static_data"][start_idx:(start_idx+data_len)]
    out_data = np.empty([data_len], dtype=np.uint8)

    # Transpose the data
    for i in range(0, filt_rows):
      for j in range(0, filt_cols):
        out_data[j*filt_rows + i] = in_data[i*filt_cols + j]

    # See if another operation uses this data and expects it to not be transposed
    filt_refs = 0
    for other_op in dnn["operations"]:
      if other_op["op_id"] == DNNOperationID.CONV_2D and other_op["parameters"]["mode"] != ConvMode.OPTIMIZED_1X1_FILTER:
        if other_op["inputs"][1] == filt_buf_idx:
          filt_refs += 1

    # If this filter is in use by another operation, see if a transposed version
    # of it already exists
    if filt_refs > 1:
      transpose_found = False
      for other_op in dnn["operations"]:
        if other_op["op_id"] == DNNOperationID.CONV_2D:
          other_buf_idx = other_op["inputs"][1]
          other_buf = dnn["buffers"][other_buf_idx]
          other_start_idx = other_buf["start_idx"]
          other_len = other_buf["dimensions"][DIMS_NUM_BATCHES]*other_buf["dimensions"][DIMS_NUM_CHANNELS]
          other_data = dnn["static_data"][start_idx:(start_idx+other_len)]
          if np.array_equal(other_data, out_data):
            op["inputs"][1] = other_buf_idx
            transpose_found = True
            break

      # if the transposed filter was not found, create a new buffer for it
      if not transpose_found:
        new_buf = copy.deepcopy(filt_buf)
        new_buf["start_idx"] = len(dnn["static_data"])
        new_buf["dimensions"][DIMS_NUM_BATCHES] = filt_cols
        new_buf["dimensions"][DIMS_NUM_CHANNELS] = filt_rows
        dnn["static_data"].extend(out_data)
        op["inputs"][1] = len(dnn["buffers"])
        dnn["buffers"].append(new_buf)
    else:
      # Save new data to static data array in place
      for i in range(0, data_len):
        dnn["static_data"][start_idx+i] = out_data[i]

      # Transpose buffer's dimensions
      filt_buf["dimensions"][DIMS_NUM_BATCHES] = filt_cols
      filt_buf["dimensions"][DIMS_NUM_CHANNELS] = filt_rows

  def __update_bias_terms(self, op, dnn, endianness, mem_order, depthwise):
    """
    Update bias terms for convolution to equal all constant values:
    Za*Zb*N-Za*sum(qb)+bias

    Parameters:
    op - the current operation's dictionary (it will be modified)
    dnn - the dnn dictionary (its misc_data will be modified)
    endianness - "big" or "little"
    mem_order - "channel_first" or "channel_last" memory order
    depthwise - if True, this is a depthwise convolution
    """

    # Get input buffers
    in_buf_idx = op["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_quant_start_idx = in_buf["quant_start_idx"]

    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    filt_quant_start_idx = filt_buf["quant_start_idx"]

    out_buf_idx = op["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    
    bias_buf_idx = op["inputs"][2]
    bias_buf = dnn["buffers"][bias_buf_idx]
   
    if depthwise:
      num_in_channels = 1
    else:
      num_in_channels = filt_buf["dimensions"][DIMS_NUM_CHANNELS]

    num_out_channels = out_buf["dimensions"][DIMS_NUM_CHANNELS]
    n = filt_buf["data_num_cols"]*filt_buf["data_num_rows"]

    op["biases"] = []
    for out_channel in range(0, num_out_channels):
      # Get this output channel's int32 bias term from static data
      bias_data_idx = bias_buf["start_idx"] + out_channel * 4
      bias = np.int32(uint8_array_to_val(dnn["static_data"][bias_data_idx:(bias_data_idx+4)], endianness))
      op["biases"].append(bias)
      # Get quantization parameters
      if filt_buf["quant_type"] == QuantType.PER_TENSOR:
        filt_quant_idx = filt_quant_start_idx
      else:
        filt_quant_idx = filt_quant_start_idx + out_channel
      za = dnn["quant_params"][in_quant_start_idx][1]
      zb = dnn["quant_params"][filt_quant_idx][1]
      # Calculate the sum of all elements of all input channels in the filter
      if depthwise:
        if mem_order == MemoryOrder.CHANNEL_FIRST:
          filt_data_idx = filt_buf["start_idx"] + out_channel
        else:
          filt_data_idx = filt_buf["start_idx"] + n * out_channel
      else:
        filt_data_idx = filt_buf["start_idx"] + out_channel * (n * num_in_channels)

      sum_qb = 0
      for i in range(0, n*num_in_channels):
        sum_qb = sum_qb + np.int32(np.int8(dnn["static_data"][filt_data_idx]))
        if depthwise and (mem_order == MemoryOrder.CHANNEL_FIRST):
          filt_data_idx += num_out_channels
        else:
          filt_data_idx += 1
      # Calculate the full convolution constant
      bias += za*zb*n*num_in_channels - za*sum_qb
      # Write this value back to static data
      bias_bytes = val_to_uint8_array(bias, 4, endianness)
      for i in range(0, len(bias_bytes)):
        dnn["static_data"][bias_data_idx+i] = bias_bytes[i]    


  def __calc_and_store_chan_mult(self, bytes, op, dnn, endianness):
    """
    Calculate the value of the convolution's channel multiple: out_chan/in_chan

    Parameters:
    bytes - array to store values in
    op - the current operation's dictionary (it will be modified)
    dnn - the dnn dictionary (its misc_data will be modified)
    endianness - "big" or "little"
    """

    # Get input buffers
    in_buf_idx = op["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]

    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]

    chan_mult = filt_buf["dimensions"][DIMS_NUM_CHANNELS] / in_buf["dimensions"][DIMS_NUM_CHANNELS]

    bytes.extend(val_to_uint8_array(np.uint32(chan_mult), 4, endianness))


  def __calc_and_store_dummy_cols(self, bytes, op, dnn, endianness):
    """
    Calculate the number of dummy columns needed and store in byte array

    Parameters:
    bytes - array to store values in
    op - the current operation's dictionary (it will be modified)
    dnn - the dnn dictionary (its misc_data will be modified)
    endianness - "big" or "little"
    """

    # Get input/output buffers
    in_buf_idx = op["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]

    dummy_cols_left = in_buf["data_start_col"]
    dummy_cols_right = in_buf["dimensions"][DIMS_NUM_COLS] - in_buf["data_num_cols"] - dummy_cols_left

    bytes.extend(val_to_uint8_array(np.uint16(dummy_cols_left), 2, endianness))
    bytes.extend(val_to_uint8_array(np.uint16(dummy_cols_right), 2, endianness))


  def __calc_and_store_scale_ratios(self, bytes, op, dnn, endianness):
    """
    Calculate the value of the quantization scale ratio: Sa*Sb/Sc

    Parameters:
    bytes - array to store values in
    op - the current operation's dictionary (it will be modified)
    dnn - the dnn dictionary (its misc_data will be modified)
    endianness - "big" or "little"
    """

    # Get input & output buffers
    in_buf_idx = op["inputs"][0]
    in_buf = dnn["buffers"][in_buf_idx]
    in_quant_start_idx = in_buf["quant_start_idx"]

    filt_buf_idx = op["inputs"][1]
    filt_buf = dnn["buffers"][filt_buf_idx]
    filt_quant_start_idx = filt_buf["quant_start_idx"]

    out_buf_idx = op["outputs"][0]
    out_buf = dnn["buffers"][out_buf_idx]
    out_quant_start_idx = out_buf["quant_start_idx"]
    op["parameters"]["scale_ratios"] = []
    if filt_buf["quant_type"] == QuantType.PER_TENSOR:
      num_out_channels = 1
    else:
      num_out_channels = out_buf["dimensions"][DIMS_NUM_CHANNELS]
    for out_channel in range(0, num_out_channels):
      sa = dnn["quant_params"][in_quant_start_idx][0]
      sb = dnn["quant_params"][filt_quant_start_idx + out_channel][0]
      sc = dnn["quant_params"][out_quant_start_idx][0]
      scale_ratio = float_to_fp_uint32(sa * sb / sc, DNN_QUANT_SCALE_Q_FORMAT)
      op["parameters"]["scale_ratios"].append(scale_ratio)
      bytes.extend(val_to_uint8_array(scale_ratio, 4, endianness))
