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
import numpy as np

from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType
from tflite.AddOptions import AddOptions
from tflite.SubOptions import SubOptions
from tflite.MulOptions import MulOptions
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.Pool2DOptions import Pool2DOptions
from tflite.ActivationFunctionType import ActivationFunctionType

from internal.constants import *
from internal.utils import *
from internal.operations.dnn_operation import DNNOperation

# Dictionary mapping all supported operators from the .tflite file to an
# operation ID
SUPPORTED_OPERATIONS = {
  BuiltinOperator.CONV_2D                      : DNNOperationID.CONV_2D,
  BuiltinOperator.DEPTHWISE_CONV_2D            : DNNOperationID.DEPTHWISE_CONV_2D,
  BuiltinOperator.FULLY_CONNECTED              : DNNOperationID.FULLY_CONNECTED,
  BuiltinOperator.RELU                         : DNNOperationID.RELU,
  BuiltinOperator.RELU6                        : DNNOperationID.RELU,
  BuiltinOperator.ADD                          : DNNOperationID.ADDSUB,
  BuiltinOperator.SUB                          : DNNOperationID.ADDSUB,
  BuiltinOperator.MUL                          : DNNOperationID.MULTIPLY,
  BuiltinOperator.CONCATENATION                : DNNOperationID.CONCATENATE,
  # QUANTIZE just changes the quantization parameters, and this can be accomplished
  # by doing a CONCATENATE with only one input
  BuiltinOperator.QUANTIZE                     : DNNOperationID.CONCATENATE,
  BuiltinOperator.RESIZE_NEAREST_NEIGHBOR      : DNNOperationID.INTERPOLATE,
  BuiltinOperator.LOGISTIC                     : DNNOperationID.SIGMOID,
  BuiltinOperator.SOFTMAX                      : DNNOperationID.SOFTMAX,
  BuiltinOperator.MAX_POOL_2D                  : DNNOperationID.MAX_POOL,
  BuiltinOperator.RESHAPE                      : DNNOperationID.RESHAPE,
  BuiltinOperator.TRANSPOSE                    : DNNOperationID.TRANSPOSE,
}

# List of tensorflow operations that will be ignored by the reader and removed
# from the model, each mapped to a string
IGNORED_OPERATIONS = [
  # It's common for a tensorflow model to start with a QUANTIZE operation to
  # convert a float32 image to an uint8 image and end with a DEQUANTIZE operation
  # to convert back. But, in the generated C code we do not support float32's and
  # take in an image that is already uint8 data.
  BuiltinOperator.DEQUANTIZE,
  # When a reshape is performed in tensorflow, a path of
  # Shape -> Strided Slice -> Pack is used to get an image's dimensions and this
  # is used as an input to Reshape. But, in the generated C code we can look at
  # the input and output buffer dimensions directly and do not need these
  # operations
  BuiltinOperator.SHAPE,
  BuiltinOperator.STRIDED_SLICE,
  BuiltinOperator.PACK,
  # Custom operations are not supported by the firmware. Some models use a custom
  # operator to implement post processing, but the parser will ignore this layer
  # and add its own post processing based on the configuration file parameters
  BuiltinOperator.CUSTOM
]

# List of DNN operations that can potentially be combined with later operations,
# mapped to the maximum number of operations that can be combined:
OPERATIONS_WITH_FUSED_ACTIVATION_SUPPORT = [
  BuiltinOperator.CONV_2D,
  BuiltinOperator.DEPTHWISE_CONV_2D,
  BuiltinOperator.FULLY_CONNECTED
]

# List of DNN operations that have extra input buffers that can be ignored.
# For these operations, Tensorflow uses a 2nd input buffer to represent some
# parameters that are not needed by the firmware.
OPERATIONS_WITH_EXTRA_INPUT = [
  DNNOperationID.INTERPOLATE, 
  DNNOperationID.RESHAPE,
  DNNOperationID.SOFTMAX,
  DNNOperationID.TRANSPOSE
]

class TFLiteReader:
  """
  Class responsible for reading a .tflite file and extracting information about
  the DNN's buffers, operations, and static data (weights/biases)
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self, filename, cfg_dict):
    """
    Constructor

    Parameters:
    filename - path and filename of input .tflite file
    endianness - the endianness for writing static data (Endianness enum)
    mem_order - memory order for 3D buffers (MemoryOrder enum)
    """
    self.__filename = filename
    self.__endianness = cfg_dict["OUTPUT_ENDIANNESS"]
    self.__mem_order = cfg_dict["MEMORY_ORDER"]
    if "INPUT_SCALE" in cfg_dict:
      self.__input_scale = cfg_dict["INPUT_SCALE"]
    else:
      self.__input_scale = None
    if "INPUT_ZEROPOINT" in cfg_dict:
      self.__input_zeropoint = cfg_dict["INPUT_ZEROPOINT"]
    else:
      self.__input_zeropoint = None

    self.__model = []
    self.__graph = []
    self.__model_inputs = []

    self.__next_scratch_ram_start_idx = 0

    self.__tensor_remap = dict()

  def load_dict(self):
    """
    Read the .tflite file and create a dictionary containing all static data
    and information about all buffers and operations in the DNN. For more
    information on its structure, see doc/dnn_compiler_design.docx

    Returns:
      output dictionary

    Raises:
      IOError if file cannot be opened
      RuntimeError if DNN contains unsupported features
    """

    # Load the model's graph using flatbuffers
    buf = open(self.__filename, 'rb').read()
    buf = bytearray(buf)
    self.__model = Model.GetRootAsModel(buf, 0)
    self.__graph = self.__model.Subgraphs(0)
    self.__model_inputs = list(self.__graph.InputsAsNumpy())

    out = dict()
    out["static_data"] = []
    out["quant_params"] = []
    out["misc_data"] = []
    out["buffers"] = []
    out["operations"] = []
    op_idx = 0
    while op_idx < self.__graph.OperatorsLength():
      try:
        op_dict, in_op, out_op, op_idx_inc = self.__get_next_op(op_idx)
      except RuntimeError as err:
        raise RuntimeError("Error encountered at operation %d of .tflite file %s" % (
          op_idx, err))
      if op_dict:
        # Check that the number of inputs and outputs are valid
        if (in_op.InputsLength() > DNN_OPERATION_MAX_INPUTS):
          raise RuntimeError("Error encountered at operation " +
            "%d (%s): Maximum number of inputs exceeded. Maximum: %d, Actual: %d" % (
              op_idx, op_dict["op_id"].name, DNN_OPERATION_MAX_INPUTS, in_op.InputsLength()))
        if (out_op.OutputsLength() > DNN_OPERATION_MAX_OUTPUTS):
          raise RuntimeError("Error encountered at operation " +
            "%d (%s): Maximum number of outputs exceeded. Maximum: %d, Actual: %d" % (
              op_idx, op_dict["op_id"].name, DNN_OPERATION_MAX_OUTPUTS, out_op.OutputsLength()))
        # Add all input connections to the DNN
        for i in range(0, in_op.InputsLength()):
          # Tensorflow gives some operations a 2nd input buffer that just 
          # contains the values of operation parameters. This
          # is not used by the firmware, so it can be removed.
          if op_dict["op_id"] in OPERATIONS_WITH_EXTRA_INPUT and i > 0:
            logging.debug("Ignoring input %d on operation %d (%s)" % (
              i, op_idx, op_dict["op_id"].name))
            continue
          else:
            tensor_id = in_op.Inputs(i)
            try:
              self.__parse_tensor_connection(tensor_id, out, op_dict, True)
            except RuntimeError as err:
              raise RuntimeError("Error encountered at input %d of operation %d (%s): %s" % (
                i, op_idx, op_dict["op_id"].name, err))
        # Add all output connections to the DNN
        for i in range(0, out_op.OutputsLength()):
          tensor_id = out_op.Outputs(i)
          try:
            self.__parse_tensor_connection(tensor_id, out, op_dict, False)
          except RuntimeError as err:
            raise RuntimeError("Error encountered at output %d of operation %d (%s): %s" % (
              i, op_idx, op_dict["op_id"].name, err))
        # Special case: if this is a RELU operation with different input and
        # output quantization parameters, split it into two operations:
        # a concatenate (to perform re-quantization) and a relu.
        if op_dict["op_id"] == DNNOperationID.RELU:
          in_buf = out["buffers"][op_dict["inputs"][0]]
          out_buf = out["buffers"][op_dict["outputs"][0]]
          in_params = out["quant_params"][in_buf["quant_start_idx"]]
          out_params = out["quant_params"][out_buf["quant_start_idx"]]
          if in_params != out_params:
            logging.debug("Splitting RELU into CONCATENATE -> RELU")
            concat_op = copy.deepcopy(op_dict)
            concat_op["op_id"] = DNNOperationID.CONCATENATE
            concat_op["outputs"][0]
            # Initialize concatenate parameters
            concat_op["parameters"]["axis"] = 1
            concat_op["parameters"]["scale_ratios"] = []
            concat_op["parameters"]["offsets"] = []
            concat_op["parameters"]["input_format"] = "tflite"
            # Update RELU to use the same buffer as input and output
            op_dict["inputs"][0] = op_dict["outputs"][0]
            # Add two connections to the original output buffer -- one for the
            # output of the CONCAT, one for the input of the RELU
            out_buf["num_connections"] += 2
            out["operations"].append(concat_op)
        out["operations"].append(op_dict)
        # If this operation has a fused activation function that needs to be
        # unfused, split it now
        activation = self.__get_fused_activation_to_unfuse(out_op)
        if activation != ActivationFunctionType.NONE:
          logging.debug("Unfusing RELU activation function from operation %d (%s)" % (
              op_idx, op_dict["op_id"].name))
          relu_op_dict = copy.deepcopy(op_dict)
          relu_op_dict["op_id"] = DNNOperationID.RELU
          relu_op_dict["inputs"] = [relu_op_dict["outputs"][0]]
          if activation == ActivationFunctionType.RELU6:
            relu_op_dict["parameters"]["clip_max"] = 6
          else:
            relu_op_dict["parameters"]["clip_max"] = RELU_NO_MAX_CLIP
          out["buffers"][relu_op_dict["outputs"][0]]["num_connections"] += 2
          out["operations"].append(relu_op_dict)
      op_idx += op_idx_inc
    # If quantization parameters were provided in configuration file, override the
    # first layer's input buffer's parameters
    input_buffer_idx = out["operations"][0]["inputs"][0]
    input_buffer = out["buffers"][input_buffer_idx]
    input_quant_idx = input_buffer["quant_start_idx"]
    input_quant = out["quant_params"][input_quant_idx]
    if self.__input_scale != None:
      logging.info("Overriding input scale found in model (%.3f) with input scale from configuration file (%.3f)" % (
        input_quant[0], self.__input_scale))
      input_quant[0] = self.__input_scale
    if self.__input_zeropoint != None:
      logging.info("Overriding input zeropoint found in model (%d) with input zeropoint from configuration file (%d)" % (
        input_quant[1], self.__input_zeropoint))
      input_quant[1] = self.__input_zeropoint - 128

    return out

  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __get_next_op(self, op_idx):
    """
    Read the next operation from the a tflite file. If multiple operations can 
    be combined into a single operation, they are combined here.

    Parameters:
    op_idx - the next operation's index in the graph
    
    Return:
    op_dict - initialized dictionary for this operation
    in_op - tflite operation to extract this operations inputs from (e.g.
            if operations are combined, the FIRST operation)
    out_op - tflite operation to extract this operations outputs from (e.g.
             if operations are combined, the LAST operation)
    num_ops - number of operations that were combined, or 1 if they were not

    Raises:
    RuntimeError if unsupported operation or parameters are encountered
    """
    ops = [];
    ops.append(self.__graph.Operators(op_idx))
    opcode = self.__model.OperatorCodes(ops[0].OpcodeIndex()).BuiltinCode()

    # Determine how many layers can be combined based on opcode
    if opcode in OPERATIONS_WITH_FUSED_ACTIVATION_SUPPORT:
      # These operations can be fused with an activation layer
      num_ops = 2
    elif opcode == BuiltinOperator.TRANSPOSE:
      # TRANSPOSE -> SOFTMAX -> TRANSPOSE can be converted to a single SOFTMAX
      num_ops = 3
    else:
      # Other operations cannot be combined
      num_ops = 1

    # Make sure there are enough operations left to combine them
    if (op_idx + num_ops) > self.__graph.OperatorsLength():
      num_ops = 1

    # Get all operations that could be combined, and make sure they are connected
    for i in range(1, num_ops):
      ops.append(self.__graph.Operators(op_idx+i))
      if (ops[i-1].Outputs(0) != ops[i].Inputs(0)):
        num_ops = 1
        break
    
    # Try to combine layers when possible
    op_dict = self.__init_op_dict(ops[0])
    if num_ops > 1:
      # Combine RELU with previous layer when possible
      if (opcode in OPERATIONS_WITH_FUSED_ACTIVATION_SUPPORT) and (
         op_dict["parameters"]["relu_clip_max"] == RELU_NONE):
        next_opcode = self.__model.OperatorCodes(ops[1].OpcodeIndex()).BuiltinCode()
        if next_opcode == BuiltinOperator.RELU:
          logging.debug("Fusing RELU activation function to operation %d (%s)" % (
              op_idx, op_dict["op_id"].name))
          op_dict["parameters"]["relu_clip_max"] = RELU_NO_MAX_CLIP
        elif next_opcode == BuiltinOperator.RELU6:
          # clip_max will be calculated later once we have quantization parameters
          logging.debug("Fusing RELU6 activation function to operation %d (%s)" % (
              op_idx, op_dict["op_id"].name))
          op_dict["parameters"]["relu_clip_max"] = 6
        else:
          # can't combine
          num_ops = 1
      elif (opcode in [BuiltinOperator.TRANSPOSE]):
        # Convert TRANSPOSE -> SOFTMAX -> TRANSPOSE to SOFTMAX with axis set based
        # on tranpose permutation
        second_opcode = self.__model.OperatorCodes(ops[1].OpcodeIndex()).BuiltinCode()
        third_opcode = self.__model.OperatorCodes(ops[2].OpcodeIndex()).BuiltinCode()
        if second_opcode == BuiltinOperator.SOFTMAX and third_opcode == BuiltinOperator.TRANSPOSE:
          logging.debug("Converting TRANSPOSE -> SOFTMAX -> TRANSPOSE to SOFTMAX at operation %d" % (
              op_idx))
          op_dict = self.__init_op_dict(ops[1])
          op_dict["parameters"]["axis"] = self.__get_softmax_axis(ops[0])
        else:
          # can't combine
          num_ops = 1
      else:
        num_ops = 1
    # Finalize outputs
    in_op = ops[0]
    out_op = ops[num_ops-1]
    return op_dict, in_op, out_op, num_ops

  def __init_op_dict(self, op):
    """
    Create operation dictionary from a tflite Operator object

    Parameters:
    op - the tflite Operator object

    Return:
    operation dictionary

    Raises:
    RuntimeError if unsupported operation or parameters are encountered
    """
    op_dict = dict()
    opcode = self.__model.OperatorCodes(op.OpcodeIndex())
    if (opcode.BuiltinCode() in IGNORED_OPERATIONS) or (
      (opcode.BuiltinCode() == BuiltinOperator.QUANTIZE) and (op.Inputs(0) in self.__model_inputs)):
      logging.debug("Ignoring operation: %s" % flatbuf_enum_to_str(opcode.BuiltinCode(), BuiltinOperator))
      # If we encounter this tensor's output buffer in the future, replace it
      # with this tensor's input. This will bypass this operation and remove it
      # from the DNN's flow.
      if op.Inputs(0) in self.__model_inputs:
        # If this was the first operation in the network, its output becomes the
        # new model input
        self.__model_inputs.append(op.Outputs(0))
      else:
        # For all other operations, when we see this tensor's output we will
        # replace it with this tensor's input to bypass the operation
        self.__tensor_remap[op.Outputs(0)] = op.Inputs(0);
      return None
    elif opcode.BuiltinCode() in SUPPORTED_OPERATIONS:
      op_dict["op_id"] = SUPPORTED_OPERATIONS[opcode.BuiltinCode()]
    else:
      raise RuntimeError("Unsupported layer type: %s" % (
        flatbuf_enum_to_str(opcode.BuiltinCode(), BuiltinOperator)))

    op_dict["inputs"] = []
    op_dict["outputs"] = []
    op_dict["working_mem_addr"] = 0

    # Extract operation-specific parameters
    try:
      DNNOperation.get_op_params_tflite(op_dict, self.__graph, op, opcode.BuiltinCode())
    except RuntimeError as e:
      raise RuntimeError("(%s): %s" % (op_dict["op_id"].name, e))
    return op_dict

  def __get_softmax_axis(self, transpose_op):
    """
    Determine a SOFTMAX operation's axis based on a transpose layer's permutation
    when a transpose is combined with softmax.

    Parameters:
    transpose_op - the tensorflow operation for the transpose

    Returns:
    axis value (0 - 3)
    """
    # Get the transpose operation's 2nd input, which is an int32 array of axis
    # permutations
    data = self.__get_tensor_data(transpose_op.Inputs(1), 4, False)
    # Convert uint8 array to uint32_t array
    perm = []
    for i in range(0, len(data), 4):
      perm.append(uint8_array_to_val(data[i:i+4], self.__endianness))
    # Read the LAST axis from the array
    axis = perm[-1]
    return axis


  def __parse_tensor_connection(self, tensor_id, out, op_dict, is_input):
    """
    Get information about a tensor connected to an operation in the tensorflow
    model and update the output dictionary

    Parameters:
    tensor_id - the tensor's id
    out       - the output dictionary. Its "buffers" and "static_data" fields
                may be updated.
    op_dict   - dictionary for the current operation. This buffer will be added
                to it's inputs/outputs
    is_input  - True if this tensor is an input to the operation, False if it is
                an output

    Raises:
    RuntimeError if invalid data type or quantization scheme is found
    """

    # Check if this tensor should be remapped to a different one (e.g. if
    # an operation was removed from the DNN)
    while tensor_id in self.__tensor_remap:
      logging.debug("Remapping tensor id: %d -> %d" % (tensor_id, self.__tensor_remap[tensor_id]))
      tensor_id = self.__tensor_remap[tensor_id]

    # Either create a new buffer for this tensor or update an existing one
    buf_dict, buf_idx = self.__find_buffer_by_id(out["buffers"], tensor_id)
    if not buf_dict:
      buf_dict = self.__init_buf_dict(tensor_id, is_input)

      # If data has been quantized to uint8s, we need to convert it to int8s
      # because the firmware only supports signed quantization
      if buf_dict["data_signed"]:
        buf_dict["convert_uint8_to_int8"] = False
      else:
        buf_dict["convert_uint8_to_int8"] = True
        buf_dict["data_signed"] = True

      self.__add_quant_params(tensor_id, buf_dict, out, buf_dict["convert_uint8_to_int8"])
      buf_idx = len(out["buffers"])
      # If the operation is in-place, the buffer descriptor may be different
      # (e.g. different dimensions) but the data we are using should be the same
      # memory. So, create a new buffer dictionary for the output buffer but make
      # make it a child of the input buffer to force the same memory section to be used.
      if not is_input and op_dict["op_id"] in INPLACE_OPERATIONS:
        buf_dict["parent"] = op_dict["inputs"][0]
        out["buffers"][buf_dict["parent"]]["num_connections"] += 1
      out["buffers"].append(buf_dict)
    else:
      if buf_dict["parent"] >= 0:
        out["buffers"][buf_dict["parent"]]["num_connections"] += 1
      buf_dict["num_connections"] += 1

    if is_input:
      # Avoid duplicate inputs. In some versions of tensorflow, the Reshape
      # operation takes in two inputs: a buffer and a shape. However, in C code
      # the Reshape just takes one input (the buffer).  So, the path to get the
      # shape is removed by the parser and as a result the Reshape will have two
      # identical inputs at this point.
      if buf_idx in op_dict["inputs"]:
        if buf_dict["parent"] >= 0:
          out["buffers"][buf_dict["parent"]]["num_connections"] -= 1
        buf_dict["num_connections"] -= 1
      else:
        op_dict["inputs"].append(buf_idx)
        # If this is static data, read the data and store it in the static data
        # array
        if buf_dict["buffer_type"] == BufferType.STATIC_DATA:
          # Make sure the data is aligned properly
          buf_dict["start_idx"] = len(out["static_data"])
          data = self.__get_tensor_data(tensor_id, 2**buf_dict["bpp_shift"],
            buf_dict["convert_uint8_to_int8"])
          out["static_data"].extend(data)
          align_byte_array(out["static_data"], 4)
    else:
      op_dict["outputs"].append(buf_idx)

  def __find_buffer_by_id(self, buffers, id):
    """
    Find a buffer in a buffer list based on its id

    Parameters:
    buffers - the buffer list
    id - the ID to search for

    Returns:
    the buffer dictionary if its found, or an empty dictionary if its not
    the buffer's index in the buffer list, or -1 if its not found
    """
    for i in range(0, len(buffers)):
      buf_dict = buffers[i]
      if buf_dict["buffer_id"] == id:
        return buf_dict, i
    return dict(), -1


  def __init_buf_dict(self, tensor_id, is_input):
    """
    Create buffer dictionary from a tensorflow tensor_id

    Parameters:
    tensor_id - the ID of this tensor in tensorflow
    is_input - if True, this tensor was first encountered as an input. If False,
               if was first encountered as an output

    Return:
    buffer dictionary

    Raises:
    RuntimeError if unsupported data type is encountered 
    """
    tensor = self.__graph.Tensors(tensor_id)
    buf_dict = dict()
    dtype = tensor.Type()
    if dtype == TensorType.UINT8:
      buf_dict["data_signed"] = False
      buf_dict["data_type"] = DataType.CHAR
      buf_dict["bpp_shift"] = 0
    elif dtype == TensorType.INT8:
      buf_dict["data_signed"] = True
      buf_dict["data_type"] = DataType.CHAR
      buf_dict["bpp_shift"] = 0
    elif dtype == TensorType.INT16:
      buf_dict["data_signed"] = True
      buf_dict["data_type"] = DataType.SHORT
      buf_dict["bpp_shift"] = 1
    elif dtype == TensorType.INT32:
      buf_dict["data_signed"] = True
      buf_dict["data_type"] = DataType.LONG
      buf_dict["bpp_shift"] = 2
    else:
      raise RuntimeError("Unsupported data type: %s" % (
        flatbuf_enum_to_str(dtype, TensorType)))
    # Ensure there are exactly DNN_BUFFER_MAX_DIMENSIONS dimensions
    dims = list(tensor.ShapeAsNumpy())
    if len(dims) > DNN_BUFFER_MAX_DIMENSIONS:
      raise RuntimeError("Too many dimensions listed for tensor " +
        "%d. Max: %dD, Actual: %dD" % (
          tensor_id, DNN_BUFFER_MAX_DIMENSIONS, len(dims)))
    # Special case: if a buffer is only 2D, treat it as rows x cols and set
    # batches and channels to 1.
    if len(dims) == 2:
      dims.insert(0, 1)
      dims.append(1)
      dim_reorder = [1, 2]
    else:
      dim_reorder = []

    while len(dims) < DNN_BUFFER_MAX_DIMENSIONS:
      dims.append(1)

    buf_dict["quant_type"] = QuantType.PER_TENSOR
    buf_dict["dimensions"] = dims
    buf_dict["data_start_row"] = 0
    buf_dict["data_start_col"] = 0
    buf_dict["data_num_rows"] = dims[DIMS_NUM_ROWS]
    buf_dict["data_num_cols"] = dims[DIMS_NUM_COLS]
    buf_dict["quant_start_idx"] = 0
    buf_dict["buffer_id"] = tensor_id   
    buf_dict["num_connections"] = 1
    buf_dict["parent"] = -1

    # If a buffer is less than 4D, track how the original dimensions were
    # rearranged or shifted. Some operations, like CONATENATE,
    # may need to adjust parameter values, like the "axis" based on this.
    buf_dict["dim_reorder"] = dim_reorder

    # If a tensor is first encountered as an input, that means it is an input
    # to the entire DNN and must either be the model's main input or a weight or
    # bias (static data). If it is an output, it needs to be stored in scratch
    # RAM
    if is_input:
      if not tensor_id in self.__model_inputs:
        buf_dict["buffer_type"] = BufferType.STATIC_DATA
        # Start index will be filled in when static data is read
      else:
        buf_dict["buffer_type"] = BufferType.MODEL_INPUT
        buf_dict["start_idx"] = self.__next_scratch_ram_start_idx
        self.__next_scratch_ram_start_idx += get_buffer_size(buf_dict)
    else:
      buf_dict["buffer_type"] = BufferType.SCRATCH_RAM
      buf_dict["start_idx"] = self.__next_scratch_ram_start_idx
      self.__next_scratch_ram_start_idx += get_buffer_size(buf_dict)
    return buf_dict

  def __add_quant_params(self, tensor_id, buf_dict, out, convert_uint8_to_int8):
    """
    Add quantization parameters for buffer to the output dictionary

    Parameters:
    tensor_id - the tensor ID associated with this buffer
    buf_dict - the buffer dictionary to add quantization data to
    out - the output dictionary
    convert_uint8_to_int8 - if true, quantization will be converted from uint8
                            to int8

    Return:
    The start index of the first parameters for this buffer
    The total number of sets of quantization parameters for this buffer

    Raises:
    RuntimeError if quantization scheme is not supported
    """
    start_idx =  len(out["quant_params"])
    tensor = self.__graph.Tensors(tensor_id)
    quant = tensor.Quantization()
    buf_dict["quant_start_idx"] = start_idx
    # Check if this buffer has more than one set of unique quantization parameters
    num_params = quant.ScaleLength()
    for i in range(1, num_params):
      if (quant.ZeroPoint(i) != quant.ZeroPoint(0)) or (quant.Scale(i) != quant.Scale(0)):
        if buf_dict["buffer_type"] == BufferType.STATIC_DATA:
          buf_dict["quant_type"] = QuantType.PER_CHANNEL
          break
        else:
          raise RuntimeError("Per-channel quantization only supported on weights & biases." +
            " Tensor %d cannot have more than one set of quantization parameters." % tensor_id)
    # Determine the number of quantization parameters to store for this buffer
    if buf_dict["quant_type"] != QuantType.PER_CHANNEL:
      num_params = 1
    # Store the quantization parameters for this buffer
    for i in range(0, num_params):
      zeropoint = quant.ZeroPoint(i)
      if convert_uint8_to_int8:
        zeropoint = zeropoint - 128
      out["quant_params"].append([quant.Scale(i), zeropoint])
   
  def __get_tensor_data(self, tensor_id, bytes_per_element, convert_uint8_to_int8):
    """
    Get a tensor's data as an array of uint8's

    Parameters:
    tensor_id - the tensor to get data for
    bytes_per_element - number of bytes per data element
    convert_uint8_to_int8 - if true, quantization will be converted from uint8
                            to int8
    """
    buffer = self.__graph.Tensors(tensor_id).Buffer()
    raw_data = self.__model.Buffers(buffer).DataAsNumpy().flatten()
    dims = (self.__graph.Tensors(tensor_id).ShapeAsNumpy())
    if (len(dims) > 2) and (self.__mem_order == MemoryOrder.CHANNEL_LAST):
      # If the data needs to be transposed, convert it from a vector to a 4D
      # matrix, transpose, then flatten back to a vector
      raw_data = np.reshape(raw_data, dims)
      if len(dims) == 3:
        raw_data = np.transpose(raw_data, (2, 0, 1))
      else:
        raw_data = np.transpose(raw_data, (0, 3, 1, 2))
      raw_data = raw_data.flatten()

    if convert_uint8_to_int8:
      # All data is stored as uint8's, so we need to convert the uint8 to an int8,
      # then cast it back to a uint8.
      data = [np.uint8(np.int8(d - 128)) for d in raw_data]
    else:
      data = copy.deepcopy(raw_data)
    # Data is little endian. If we need big endian, swap byte order
    if self.__endianness == Endianness.BIG and bytes_per_element > 1:
      for i in range(0, len(data), bytes_per_element):
        if bytes_per_element == 2:
          data[i] = raw_data[i+1]
          data[i+1] = raw_data[i]
        else:  # bytes_per_element == 4
          data[i] = raw_data[i+3]
          data[i+1] = raw_data[i+2]
          data[i+2] = raw_data[i+1]
          data[i+3] = raw_data[i]
    return data


  def __get_fused_activation_to_unfuse(self, op):
    """
    Check if this operation has an activation function that is fused in tensorflow
    but needs to be split in the firmware implementation

    Inputs:
    op - the tensorflow operator object

    Returns:
    the activation function, or ActivationFunctionType.NONE if there isn't one
    """
    opcode = self.__model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    if opcode == BuiltinOperator.ADD:
      options = AddOptions()
    elif opcode == BuiltinOperator.SUB:
      options = SubOptions()
    elif opcode == BuiltinOperator.MUL:
      options = MulOptions()
    elif opcode == BuiltinOperator.CONCATENATION:
      options = ConcatenationOptions()
    elif opcode == BuiltinOperator.MAX_POOL_2D:
      options = Pool2DOptions()
    else:
      return ActivationFunctionType.NONE
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return options.FusedActivationFunction()
