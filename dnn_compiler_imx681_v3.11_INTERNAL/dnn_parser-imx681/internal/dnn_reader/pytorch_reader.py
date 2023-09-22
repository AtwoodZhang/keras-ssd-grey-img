# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import copy
import torch
import numpy as np
import logging

from internal.constants import *
from internal.utils import *

from internal.operations.dnn_operation import DNNOperation

# Dictionary mapping all supported kinds of node in the model's graph to a
# DNN operation ID
SUPPORTED_NODES = {
  "aten::add":                  DNNOperationID.ADDSUB,
  "aten::flatten":              DNNOperationID.RESHAPE,
  "aten::max_pool2d":           DNNOperationID.MAX_POOL,
  "aten::mul":                  DNNOperationID.MULTIPLY,
  "aten::quantize_per_tensor":  DNNOperationID.CONCATENATE,
  "aten::relu":                 DNNOperationID.RELU,
  "aten::relu_":                DNNOperationID.RELU,
  "aten::reshape":              DNNOperationID.RESHAPE,
  "aten::sigmoid":              DNNOperationID.SIGMOID,
  "aten::softmax":              DNNOperationID.SOFTMAX,
  "aten::sub":                  DNNOperationID.ADDSUB,
  "aten::transpose":            DNNOperationID.TRANSPOSE,
  "aten::upsample_nearest2d":   DNNOperationID.INTERPOLATE,
  "aten::view":                 DNNOperationID.RESHAPE,

  "quantized::add":             DNNOperationID.ADDSUB,
  "quantized::add_relu":        DNNOperationID.ADDSUB,
  "quantized::cat":             DNNOperationID.CONCATENATE,
  # Note: Operation ID may be changed to DEPTHWISE_CONV_2D if a depthwise operation is detected
  "quantized::conv2d":          DNNOperationID.CONV_2D,
  "quantized::conv2d_relu":     DNNOperationID.CONV_2D,
  "quantized::linear":          DNNOperationID.FULLY_CONNECTED,
  "quantized::linear_relu":     DNNOperationID.FULLY_CONNECTED,
  "quantized::mul":             DNNOperationID.MULTIPLY,
}

# List of kinds of node in the model's graph that will be ignored by the reader
# and removed from the model without printing a warning
IGNORED_NODES = [
  # Dequantize layers can be ignored because they will always be either at the
  # end of the DNN or right before a quantize layer
  "aten::dequantize",
  # Reshape consists of permute -> contiguous -> view
  "aten::permute",
  "aten::contiguous",
  # This node just creates a list of nodes that are input to a future layer (e.g. concatenate layer)
  "prim::ListConstruct",
  "prim::GetAttr",
  # Dropout layers are ignored during inference
  "aten::dropout",
]

# List of kinds of nodes that perform quantization
QUANTIZE_NODES = [
  "aten::quantize_per_tensor",
]

# List of DNN operations that can potentially be combined with later operations,
# mapped to the maximum number of operations that can be combined:
OPERATIONS_WITH_FUSED_ACTIVATION_SUPPORT = [
  DNNOperationID.CONV_2D,
  DNNOperationID.DEPTHWISE_CONV_2D,
  DNNOperationID.FULLY_CONNECTED
]

# List of DNN operations where the output quantization parameters will always
# match the input quantization parameters
OPERATIONS_WITH_QUANTIZATION_PRESERVED = [
  DNNOperationID.INTERPOLATE,
  DNNOperationID.MAX_POOL,
  DNNOperationID.RESHAPE,
  DNNOperationID.RELU
]

# List of DNN operations where the quantization parameters can be inferred if
# none are found in the pytorch model
OPERATIONS_WITH_DEFAULT_QUANTIZATION = [
  DNNOperationID.SIGMOID,
  DNNOperationID.SOFTMAX
]

# Default scale and zeropoint for any operations that can use default quantization
# (see OPERATIONS_WITH_DEFAULT_QUANTIZATION).
#
# These values give an output range of 0.0 - 1.0
DEFAULT_QUANT_PARAMS = [np.float32(1.0/255.0), np.int8(-128)]

class PyTorchReader:
  """
  Class responsible for reading a PyTorch model and extracting information about
  the DNN's buffers, operations, and static data (weights/biases)
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self, model, cfg_dict):
    """
    Constructor

    Parameters:
    model - the pytorch model object
    endianness - the endianness for writing static data (Endianness enum)
    mem_order - memory order for 3D buffers (MemoryOrder enum)
    """
    self.__model = model
    self.__graph = None
    self.__endianness = cfg_dict["OUTPUT_ENDIANNESS"]
    self.__mem_order = cfg_dict["MEMORY_ORDER"]
    self.__cfg_dict = cfg_dict
    self.__next_scratch_ram_start_idx = 0
    self.__model_inputs = []
    self.__input_scale = 1.0
    self.__input_zeropoint = 0
    self.__tensor_remap = dict()

  def load_dict(self):
    """
    Read the PyTorch model and create a dictionary containing all static data
    and information about all buffers and operations in the DNN. For more
    information on its structure, see doc/dnn_compiler_design.docx

    Returns:
      output dictionary

    Raises:
      RuntimeError if DNN contains unsupported features
    """

    out = dict()
    out["static_data"] = []
    out["quant_params"] = []
    out["misc_data"] = []
    out["buffers"] = []
    out["operations"] = []

    input_tensor = torch.randn(1, 1, DNN_INPUT_ROWS, DNN_INPUT_COLS, device='cpu')

    # Trace the model to convert the model class to a graph
    try:
      self.__graph = torch.jit.trace(self.__model, input_tensor, check_trace=False).inlined_graph
    except RuntimeError as err:
      raise RuntimeError("Failed to trace model: %s" % err)

    # Parse through the trace's graph node-by-node and populate the output dictionary
    op_idx = 0
    for node in self.__graph.nodes():
      node_type = node.kind()
      inputs = self.__get_tensor_list(node.inputs())
      outputs = self.__get_tensor_list(node.outputs())
      if len(inputs) > 0 and len(outputs) > 0:
        if node_type in SUPPORTED_NODES:
          op_idx = self.__process_node(node, inputs, outputs, op_idx, out)
        elif node_type in IGNORED_NODES:
          self.__ignore_node(node)
        else:
          # If the layer is not in the SUPPORTED_NODES or IGNORED_NODES lists, that means we've never encountered it
          # before and do not know how to handle it.
          raise RuntimeError("Unsupported layer type: %s" % node_type)

    # Make sure that all outputs have valid quantization parameters
    op_idx = 0
    for op in out["operations"]:
      out_buf = out["buffers"][op["outputs"][0]]
      if out_buf["quant_start_idx"] == None:
        if op["op_id"] in OPERATIONS_WITH_DEFAULT_QUANTIZATION:
          logging.debug("Using default quantization parameters for output of layer %d (%s)" % (
            op_idx, op["op_id"]))
          out_buf["quant_start_idx"] = len(out["quant_params"])
          out_buf["quant_type"] = QuantType.PER_TENSOR
          out["quant_params"].append(DEFAULT_QUANT_PARAMS)
        else:
          raise RuntimeError("No quantization parameters found for layer %d (%s)" % (
            op_idx, op["op_id"]))
      op_idx += 1

    # Second pass to combine or split any operations where its required
    prev_op = out["operations"][0]
    op_idx = 1
    while op_idx < len(out["operations"]):
      op = out["operations"][op_idx]
      op_idx += 1
      # If possible, fuse RELU to previous operation
      ops_fused = False
      if (op["op_id"] == DNNOperationID.RELU) and (
        prev_op["op_id"] in OPERATIONS_WITH_FUSED_ACTIVATION_SUPPORT):
        mid_buf_idx = op["inputs"][0]
        mid_buf = out["buffers"][mid_buf_idx]
        # Make sure these operations are connected and the intermediate result
        # between them isn't connected to any other layers
        if prev_op["outputs"][0] == mid_buf_idx and mid_buf["num_connections"] == 2:
          prev_op["parameters"]["relu_clip_max"] = op["parameters"]["clip_max"]
          prev_op["outputs"][0] = op["outputs"][0]
          out["operations"].remove(op)
          self.__remove_buffer(out, mid_buf_idx)
          ops_fused = True
          op_idx -= 1
          logging.debug("Fusing RELU to %s", prev_op["op_id"])
      prev_op = op

      # If operation has a fused activation but firmware does not support a fused
      # activation, split it now
      if op["op_id"] == DNNOperationID.ADDSUB and op["parameters"]["clip_max"] != RELU_NONE:
          relu_op = copy.deepcopy(op)
          relu_op["op_id"] = DNNOperationID.RELU
          relu_op["inputs"] = [relu_op["outputs"][0]]
          out["buffers"][relu_op["outputs"][0]]["num_connections"] += 2
          out["operations"].insert(op_idx, relu_op)
          prev_op = relu_op
          op_idx += 1
          
      # Otherwise, if RELU has different input and output quantization parameters,
      # split it into two operations: a concatenate (to perform re-quantization) and a relu.
      if op["op_id"] == DNNOperationID.RELU and not ops_fused:
        in_buf = out["buffers"][op["inputs"][0]]
        out_buf = out["buffers"][op["outputs"][0]]
        in_params = out["quant_params"][in_buf["quant_start_idx"]]
        out_params = out["quant_params"][out_buf["quant_start_idx"]]
        if in_params != out_params:
          logging.debug("Splitting RELU into CONCATENATE -> RELU")
          concat_op = copy.deepcopy(op)
          concat_op["op_id"] = DNNOperationID.CONCATENATE
          # Initialize concatenate parameters
          concat_op["parameters"]["axis"] = 1
          concat_op["parameters"]["scale_ratios"] = []
          concat_op["parameters"]["offsets"] = []
          concat_op["parameters"]["input_format"] = "pytorch"
          # Update RELU to use the same buffer as input and output
          op["inputs"][0] = op["outputs"][0]
          # Add two connections to the original output buffer -- one for the
          # output of the CONCAT, one for the input of the RELU
          out_buf["num_connections"] += 2
          out["operations"].insert(op_idx, concat_op)
          prev_op = concat_op
          op_idx += 1
    return out

  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __process_node(self, node, inputs, outputs, op_idx, out):
    """
    Process a node from the pytorch graph and add details to the output dictionary

    Parameters:
    node - the next graph node
    op_idx - the next operation's index in the output dictionary
    inputs - list of input tensors
    outputs - list of output tensors
    out - the output dictionary
    
    Return:
    op_idx - the next operation's index in the output dictionary

    Raises:
    RuntimeError if a node cannot be parsed
    """
    op_dict = dict()
    op_dict["inputs"] = []
    op_dict["outputs"] = []
    op_dict["working_mem_addr"] = 0

    # Map the graph node back to a pytorch module that contains the layer's parameters
    module = self.__node_to_module(node)

    if node.kind() in QUANTIZE_NODES:
      if not self.__process_quantize_node(node, module, inputs, outputs, out, op_idx == 0):
        return op_idx

    # Initialize the op_dict with the operation's parameters
    op_id = SUPPORTED_NODES[node.kind()]
    try:
      op_dict["op_id"] = op_id
      DNNOperation.get_op_params_pytorch(op_dict, module, node)
    except RuntimeError as e:
      raise RuntimeError("(%s): %s" % (op_id.name, e))

    # Process this node's inputs and outputs and add buffers for them to the output dictionary
    if not self.__process_node_io(module, inputs, outputs, op_dict, op_idx, out):
      return op_idx

    # Make sure the inputs have valid quantization parameters
    for in_idx in op_dict["inputs"]:
      in_buf = out["buffers"][in_idx]
      if in_buf["quant_start_idx"] == None:
        if (in_buf["buffer_type"] == BufferType.SCRATCH_RAM) and (
            in_buf["origin_op_id"] in OPERATIONS_WITH_DEFAULT_QUANTIZATION):
          logging.debug("Using default quantization parameters for input to %s layer" % op_dict["op_id"])
          in_buf["quant_start_idx"] = len(out["quant_params"])
          in_buf["quant_type"] = QuantType.PER_TENSOR
          out["quant_params"].append(DEFAULT_QUANT_PARAMS)
        else:
          raise RuntimeError("No quantization parameters found for input to layer (%s)" % (
            op_dict["op_id"]))

    # Add weight & bias buffers if they exist
    if module:
      state_dict = module.state_dict()
      if "weight" in state_dict:
        # Read weights from the state dictionary
        try:
          if op_dict["op_id"] == DNNOperationID.DEPTHWISE_CONV_2D:
            swap_batch_and_chan = True
          else:
            swap_batch_and_chan = False
          self.__process_weights(state_dict["weight"], "weight%d" % op_idx, op_dict, out, swap_batch_and_chan)
        except RuntimeError as e:
          raise RuntimeError("(%s): Failed to load weights: %s" % (op_id.name, e))
      if "bias" in state_dict:
        if state_dict["bias"] == None:
          # Create a bias term of all 0's if no bias is included for this module
          out_buf_idx = op_dict["outputs"][0]
          num_out_chan = out["buffers"][out_buf_idx]["dimensions"][DIMS_NUM_CHANNELS]
          self.__process_biases(torch.zeros([num_out_chan], dtype=torch.float32), "bias%d" % op_idx, op_dict, out)
        else:
          # Read biases from the state dictionary
          try:
            self.__process_biases(state_dict["bias"], "bias%d" % op_idx, op_dict, out)
          except RuntimeError as e:
            raise RuntimeError("(%s): Failed to load biases: %s" % (op_id.name, e))
      if "_packed_params._packed_params" in state_dict:
        # Read weights and biases from a single packed params list
        self.__process_weights(state_dict["_packed_params._packed_params"][0], "weight%d" % op_idx, op_dict, out, False)
        self.__process_biases(state_dict["_packed_params._packed_params"][1], "bias%d" % op_idx, op_dict, out)

    # Add the final operation to the output dictionary
    out["operations"].append(op_dict)
    op_idx += 1
    return op_idx

  def __remove_buffer(self, out, buf_idx):
    """
    Delete a buffer from the output dictionary, along with its quantization
    parameters.

    out - the output dictionary
    buf_idx - the buffer to delete
    """
    quant_start_idx = out["buffers"][buf_idx]["quant_start_idx"]
    if out["buffers"][buf_idx]["quant_type"] == QuantType.PER_BATCH:
      num_quant_params = out["buffers"]["dimensions"][DIMS_NUM_BATCHES]
    elif out["buffers"][buf_idx]["quant_type"] == QuantType.PER_CHANNEL:
      num_quant_params = out["buffers"]["dimensions"][DIMS_NUM_CHANNELS]
    else:
      num_quant_params = 1

    del out["buffers"][buf_idx]
    del out["quant_params"][quant_start_idx:(quant_start_idx + num_quant_params)]
    for buf in out["buffers"]:
      if buf["quant_start_idx"] > quant_start_idx:
        buf["quant_start_idx"] -= num_quant_params
      if buf["parent"] > buf_idx:
        buf["parent"] -= 1
    for op in out["operations"]:
      for i in range(0, len(op["inputs"])):
        if op["inputs"][i] > buf_idx:
          op["inputs"][i] -= 1
      for i in range(0, len(op["outputs"])):
        if op["outputs"][i] > buf_idx:
          op["outputs"][i] -= 1


  def __ignore_node(self, node):
    """
    Ignore a node in the Pytorch graph. If it is a processing layer, create a remapping to replace this layer's output
    tensor with it's input tensor when it is encountered later in the DNN. This effectively removes the layer.

    Parameters:
    node - the Pytorch graph node
    """
    logging.debug("Ignoring layer type: %s" % node.kind())
    inputs = self.__get_tensor_list(node.inputs())
    outputs = self.__get_tensor_list(node.outputs())
    # If this layer has a tensor input and output, then it is a processing layer and needs to be removed
    if len(inputs) == 1 and len(outputs) == 1:
      self.__tensor_remap[outputs[0].debugName()] = inputs[0]

  def __node_to_module(self, node):
    # Get the PyTorch module associated with a given node in the model's trace graph
    #
    # Parameters:
    #   node - the node to lookup
    #
    # Returns:
    #   module - a pytorch module from the model

    # Get the node's scope name string
    #   Example: '__module.features/__module.features.in_conv/__module.features.in_conv.conv2d'
    scope_name = node.scopeName()
    if not scope_name:
      return None

    # Extract the full variable name path from scopeName
    #  Example:  'features.in_conv.conv2d'
    var_path = scope_name.split("/")[-1].replace("__module.", "")

    # Convert path to a list
    #  Example: ['features', 'in_conv', 'conv2d']
    var_list = var_path.split(".")

    # Starting at the top level of the model, recursively get modules by name
    #  Example:
    #    module = model.features
    #    module = module.in_conv
    #    module = module.conv2d
    module = self.__model
    for level in var_list:
      module = getattr(module, level)

    return module

  def __process_quantize_node(self, node, module, inputs, outputs, out, is_first_layer):
    """
    Special handling for a quantize node type: check if the layer should be ignored (e.g. its the first in the DNN or
    the input and output quantization parameters are equal) or if it should be included as a layer in the DNN

    Parameters:
      node - the PyTorch graph node
      module - the PyTorch module for this node
      inputs - input tensors for this node
      outputs - output tensors for this node
      out - output dictionary
      is_first_layer - True if this is the first layer in the DNN

    Returns:
      True if the node should be processed, False if it should be ignored
    """
    if is_first_layer:
      # if this is the first operation, the input will be a float and the output will be the actual quantized input
      # that we want to use for the DNN in firmware
      self.__model_inputs.append(outputs[0].debugName())
      if module:
        self.__input_scale = module.scale
        self.__input_zeropoint = module.zero_point
      else:
        # If there isn't a module associated with this layer, then we expect the quantize layer to
        # have input[1] = scale, input[2] = zeropoint, where scale and zeropoint are variables in
        # the model. Get their names, then look up their values by name
        scale_var_name = inputs[1].debugName()
        zero_var_name = inputs[2].debugName()
        self.__input_scale = getattr(self.__model, scale_var_name).item()
        self.__input_zeropoint = getattr(self.__model, zero_var_name).item()
      return False
    else:
      # Get buffer descriptor for input and output
      in_buf, idx = self.__find_buffer_by_id(out["buffers"], inputs[0].debugName())
      out_buf = self.__init_buf_dict_for_io_buffer(outputs[0].type(), outputs[0].debugName(), False)
      if out_buf["data_signed"] == False:
        convert_uint8_to_int8 = True
        out_buf["data_signed"] = True
      else:
        convert_uint8_to_int8 = False
      # Sometimes the scale and zeropoint are stored as a 1-element array, and sometimes they are a scalar value
      if type(module.scale) == torch.Tensor:
        out_quant_params = self.__convert_quant_params(module.scale[0], module.zero_point[0], convert_uint8_to_int8)
      else:
        out_quant_params = self.__convert_quant_params(module.scale, module.zero_point, convert_uint8_to_int8)

      # If the buffer doesn't have quantization parameters yet, add them now
      if in_buf["quant_start_idx"] == None:
        out_buf["quant_start_idx"] = len(out["quant_params"])
        out_buf["parent"] = in_buf["parent"]
        out["quant_params"].append(out_quant_params)
        out["buffers"][idx] = copy.deepcopy(out_buf)
        return False
      else:
        # Check if the quantization parameters have changed. If they have, then we need to include this quantize layer
        # to 'requantize' the tensor. If they haven't, it can be ignored.
        in_quant_params = out["quant_params"][in_buf["quant_start_idx"]]
        if (in_quant_params[0] == out_quant_params[0]) and (in_quant_params[1] == out_quant_params[1]):
          return False
        else:
          return True

  def __process_node_io(self, module, inputs, outputs, op_dict, op_idx, out):
    """
    Process a PyTorch graph node's inputs and outputs and add them to the output dictionary

    Parameters:
      module - the PyTorch module for this layer
      inputs - list of input tensors
      outputs - list of output tensors
      op_dict - the operation dictionary for this layer
      op_idx - the index of this layer in the output DNN
      out - the output dictionary for the DNN

    Returns:
      True is we should include this node, false if it should be ignored

    Raises:
      RuntimeError if invalid inputs/outputs are encountered
    """
    # Check that the number of inputs and outputs are valid
    if len(inputs) > DNN_OPERATION_MAX_INPUTS:
      raise RuntimeError("Error encountered at operation " +
                         "%d (%s): Maximum number of inputs exceeded. Maximum: %d, Actual: %d" % (
                           op_idx, op_dict["op_id"].name, DNN_OPERATION_MAX_INPUTS, len(inputs)))
    if len(outputs) > DNN_OPERATION_MAX_OUTPUTS:
      raise RuntimeError("Error encountered at operation " +
                         "%d (%s): Maximum number of outputs exceeded. Maximum: %d, Actual: %d" % (
                           op_idx, op_dict["op_id"].name, DNN_OPERATION_MAX_OUTPUTS, len(outputs)))

    # Make sure at least one input is the output of another buffer
    include_layer = False
    for i in range(0, len(inputs)) :
      buf_dict, buf_idx = self.__find_buffer_by_id(out["buffers"], inputs[i].debugName())
      if buf_dict:
        include_layer = True
        break
      if inputs[i].debugName() in self.__model_inputs:
        include_layer = True
        break
    if not include_layer:
      return False


    # Add all input connections to the DNN
    for i in range(0, len(inputs)):
      try:
        self.__parse_tensor_connection(inputs[i], module, out, op_dict, True)
      except RuntimeError as err:
        raise RuntimeError("Error encountered at input %d of operation %d (%s): %s" % (
          i, op_idx, op_dict["op_id"].name, err))

    # Add all output connections to the DNN
    for i in range(0, len(outputs)):
      try:
        self.__parse_tensor_connection(outputs[i], module, out, op_dict, False)
      except RuntimeError as err:
        raise RuntimeError("Error encountered at output %d of operation %d (%s): %s" % (
          i, op_idx, op_dict["op_id"].name, err))

    return True

  def __process_weights(self, tensor, name, op_dict, out, swap_batch_and_chan):
    """
    Process a PyTorch tensor for weights by adding an input to the current operation, creating a
    buffer dictionary for the input, and storing the data in the static_data array

    Parameters:
      tensor - the PyTorch tensor for this data
      name - string name for this tensor
      op_dict - the operation dictionary
      out - the output dictionary
      swap_batch_and_chan - if true, swap the batch (dims[0]) and channel (dims[1]) dimensions, which is required for
        a depthwise convolution's weights

    Raises:
      RuntimeError if data was unsupported format
    """
    # Add an input for this tensor to the current operation
    buf_idx = len(out["buffers"])
    op_dict["inputs"].append(buf_idx)
    # Create buffer dictionary for this tensor
    buf_dict = self.__init_buf_dict_for_static_data(tensor, name, out, swap_batch_and_chan)
    self.__add_quant_params_for_static_data(tensor, op_dict, buf_dict, out)
    out["buffers"].append(buf_dict)
    # Get the raw data and append it to the static data array
    out["static_data"].extend(self.__get_weight_data(tensor, swap_batch_and_chan))
    align_byte_array(out["static_data"], 4)

  def __process_biases(self, tensor, name, op_dict, out):
    """
    Process a PyTorch tensor for biases by adding an input to the current operation, creating a
    buffer dictionary for the input, and storing the data in the static_data array

    Parameters:
      tensor - the PyTorch tensor for this data
      name - string name for this tensor
      op_dict - the operation dictionary
      out - the output dictionary

    Raises:
      RuntimeError if data was unsupported format
    """
    # Add an input for this tensor to the current operation
    buf_idx = len(out["buffers"])
    op_dict["inputs"].append(buf_idx)
    # Create buffer dictionary for this tensor
    buf_dict = self.__init_buf_dict_for_static_data(tensor, name, out, False)
    self.__add_quant_params_for_static_data(tensor, op_dict, buf_dict, out)
    out["buffers"].append(buf_dict)
    # Get the raw data, quantize it, and append it to the static data array
    in_buf = out["buffers"][op_dict["inputs"][0]]
    w_buf = out["buffers"][op_dict["inputs"][1]]
    in_quant = out["quant_params"][in_buf["quant_start_idx"]]
    w_scale_start_idx = w_buf["quant_start_idx"]
    if w_buf["quant_type"] == QuantType.PER_TENSOR:
      w_quant = out["quant_params"][w_scale_start_idx:(w_scale_start_idx+1)]
      w_quant_per_element = False
    else:
      w_quant = out["quant_params"][w_scale_start_idx:(w_scale_start_idx+buf_dict["dimensions"][0])]
      w_quant_per_element = True

    out["static_data"].extend(self.__get_bias_data(tensor, in_quant, w_quant, w_quant_per_element))
    align_byte_array(out["static_data"], 4)



  def __parse_tensor_connection(self, tensor, module, out, op_dict, is_input):
    """
    Get information about a tensor connected to an operation in the pytorch
    model and update the output dictionary

    Parameters:
    tensor    - the tensor (a node input or output)
    module    - the PyTorch module for the current layer
    out       - the output dictionary. Its "buffers" and "static_data" fields
                may be updated.
    op_dict   - dictionary for the current operation. This buffer will be added
                to it's inputs/outputs
    is_input  - True if this tensor is an input to the operation, False if it is
                an output

    Raises:
    RuntimeError if invalid data type or quantization scheme is found
    """
    # Either create a new buffer for this tensor or update an existing one
    buf_dict, buf_idx = self.__find_buffer_by_id(out["buffers"], tensor.debugName())
    if not buf_dict:
      buf_dict = self.__init_buf_dict_for_io_buffer(tensor.type(), tensor.debugName(), is_input)
      buf_idx = len(out["buffers"])
      # If the operation is in-place, the buffer descriptor may be different
      # (e.g. different dimensions) but the data we are using should be the same
      # memory. So, create a new buffer dictionary for the output buffer but make
      # make it a child of the input buffer to force the same memory section to be used.
      if not is_input and op_dict["op_id"] in INPLACE_OPERATIONS:
        buf_dict["parent"] = op_dict["inputs"][0]
        parent_buf = out["buffers"][buf_dict["parent"]]
        buf_dict["quant_start_idx"] = parent_buf["quant_start_idx"]
        buf_dict["data_signed"] = parent_buf["data_signed"]
        buf_dict["data_type"] = parent_buf["data_type"]
        if "origin_op_id" in parent_buf:
          buf_dict["origin_op_id"] = parent_buf["origin_op_id"]
        parent = buf_dict["parent"]
        while parent >= 0:
          out["buffers"][parent]["num_connections"] += 1
          parent = out["buffers"][parent]["parent"]
      else:
        self.__add_quant_params_for_io_buffer(tensor, module, buf_dict, op_dict, out)
      out["buffers"].append(buf_dict)
    else:
      parent = buf_dict["parent"]
      while parent >= 0:
        out["buffers"][parent]["num_connections"] += 1
        parent = out["buffers"][parent]["parent"]
      buf_dict["num_connections"] += 1

    if is_input:
      op_dict["inputs"].append(buf_idx)
    else:
      op_dict["outputs"].append(buf_idx)

  def __get_tensor_list(self, generator):
    """
    Get a list of tensors connected to a pytorch node

    Parameters:
    generator - the node's generate function to get inputs or outputs (e.g.
                node.inputs() or node.outputs()
    
    Returns:
    list of TensoryType nodes
    """
    tensor_list = []
    for n in generator:
      if n.type().__class__ == torch._C.TensorType:
        # Only include tensors, not primatives
        # If any remaps exist, do them now
        while n.debugName() in self.__tensor_remap:
          n = self.__tensor_remap[n.debugName()]
        tensor_list.append(n)
      elif n.type().__class__ == torch._C.ListType:
        tensor_list.extend(self.__get_tensor_list(n.node().inputs()))

    return tensor_list


  @staticmethod
  def __find_buffer_by_id(buffers, id):
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

  def __init_buf_dict_for_io_buffer(self, tensor, tensor_id, is_input):
    """
    Create buffer dictionary for a scratch buffer from a PyTorch TensorType

    Parameters:
    tensor - a TensorType input or output from the PyTorch graph
    tensor_id - string to identify this tensor
    is_input  - True if this tensor is an input to the operation, False if it is
                an output
    Return:
    buffer dictionary

    Raises:
    RuntimeError if unsupported data type is encountered 
    """
    buf_dict = dict()
    dtype = tensor.scalarType()
    # Float is treated as QInt8 since it will be quantized later
    if dtype == "QInt8" or dtype == "Float":
      buf_dict["data_signed"] = True
      buf_dict["data_type"] = DataType.CHAR
      buf_dict["bpp_shift"] = 0
    elif dtype == "QUInt8":
      buf_dict["data_signed"] = False
      buf_dict["data_type"] = DataType.CHAR
      buf_dict["bpp_shift"] = 0
    else:
      raise RuntimeError("Unrecognized data type: %s" % dtype)
    # Ensure there are exactly DNN_BUFFER_MAX_DIMENSIONS dimensions
    dims = tensor.sizes()
    dims, dim_reorder = self.__extend_dims(dims, tensor_id, False)

    buf_dict["dimensions"] = dims
    buf_dict["data_start_row"] = 0
    buf_dict["data_start_col"] = 0
    buf_dict["data_num_rows"] = dims[DIMS_NUM_ROWS]
    buf_dict["data_num_cols"] = dims[DIMS_NUM_COLS]
    buf_dict["quant_type"] = QuantType.PER_TENSOR
    buf_dict["buffer_id"] = tensor_id
    buf_dict["num_connections"] = 1
    buf_dict["parent"] = -1
    buf_dict["dim_reorder"] = dim_reorder

    # If a tensor is first encountered as an input, that means it is an input
    # to the entire DNN and must either be the model's main input or a weight or
    # bias (static data). If it is an output, it needs to be stored in scratch
    # RAM
    if tensor_id in self.__model_inputs:
      buf_dict["buffer_type"] = BufferType.MODEL_INPUT
    elif is_input:
      buf_dict["buffer_type"] = BufferType.STATIC_DATA
    else:
      buf_dict["buffer_type"] = BufferType.SCRATCH_RAM

    buf_dict["start_idx"] = self.__next_scratch_ram_start_idx
    self.__next_scratch_ram_start_idx += get_buffer_size(buf_dict)
    return buf_dict


  def __init_buf_dict_for_static_data(self, tensor, tensor_id, out, swap_batch_and_chan):
    """
    Create buffer dictionary for a static data buffer from a PyTorch Tensor

    Parameters:
    tensor - a Tensor object from a PyTorch state_dict
    tensor_id - string to identify this tensor
    op_id - operation ID of the associated layer
    out - output dictionary
    swap_batch_and_chan - if true, swap the batch (dims[0]) and channel (dims[1]) dimensions, which is required for
      a depthwise convolution's weights

    Return:
    buffer dictionary

    Raises:
    RuntimeError if unsupported data type is encountered
    """
    buf_dict = dict()
    dtype = tensor.type()
    if dtype == 'torch.quantized.QInt8Tensor':
      buf_dict["data_signed"] = True
      buf_dict["data_type"] = DataType.CHAR
      buf_dict["bpp_shift"] = 0
    elif dtype == 'torch.FloatTensor':
      buf_dict["data_signed"] = True
      buf_dict["data_type"] = DataType.LONG
      buf_dict["bpp_shift"] = 2
    else:
      raise RuntimeError("Unrecognized data type: %s" % dtype)

    # Ensure there are exactly DNN_BUFFER_MAX_DIMENSIONS dimensions
    dims = tensor.size()
    dims, dim_reorder  = self.__extend_dims(dims, tensor_id, swap_batch_and_chan)

    buf_dict["dimensions"] = dims
    buf_dict["data_start_row"] = 0
    buf_dict["data_start_col"] = 0
    buf_dict["data_num_rows"] = dims[DIMS_NUM_ROWS]
    buf_dict["data_num_cols"] = dims[DIMS_NUM_COLS]
    buf_dict["quant_type"] = QuantType.PER_TENSOR
    buf_dict["buffer_id"] = tensor_id
    buf_dict["num_connections"] = 1
    buf_dict["parent"] = -1
    buf_dict["dim_reorder"] = dim_reorder
    buf_dict["buffer_type"] = BufferType.STATIC_DATA
    buf_dict["start_idx"] = len(out["static_data"])
    return buf_dict

  def __add_quant_params_for_io_buffer(self, tensor, module, buf_dict, op_dict, out):
    """
    Add quantization parameters for scratch buffer to the output dictionary

    Parameters:
    tensor - the PyTorch tensor for this data
    module - this layer's PyTorch module
    buf_dict - the buffer dictionary to add quantization data to
    op_dict - the operation dictionary
    out - the output dictionary

    Return:
    The start index of the first parameters for this buffer
    The total number of sets of quantization parameters for this buffer

    Raises:
    RuntimeError if quantization scheme is not supported
    """
    start_idx = len(out["quant_params"])
    buf_dict["quant_start_idx"] = start_idx

    # If necessary,convert unsigned data to signed data
    if buf_dict["data_signed"] == False:
      convert_uint8_to_int8 = True
      buf_dict["data_signed"] = True
    else:
      convert_uint8_to_int8 = False
  
    if buf_dict["buffer_type"] == BufferType.MODEL_INPUT:
      # The model input uses the input scale and zeropoint that were saved from the QUANTIZE operation at the
      # start of the DNN
      if type(self.__input_scale) == torch.Tensor:
        scale = self.__input_scale[0]
        zero = self.__input_zeropoint[0]
      else:
        scale = self.__input_scale
        zero = self.__input_zeropoint
      if "INPUT_SCALE" in self.__cfg_dict:
        logging.info("Overriding input scale found in model (%.3f) with input scale from configuration file (%.3f)" % (
          scale, self.__cfg_dict["INPUT_SCALE"]))
        scale = self.__cfg_dict["INPUT_SCALE"]
      if "INPUT_ZEROPOINT" in self.__cfg_dict:
        logging.info("Overriding input zeropoint found in model (%d) with input zeropoint from configuration file (%d)" % (
          zero, self.__cfg_dict["INPUT_ZEROPOINT"]))
        zero = self.__cfg_dict["INPUT_ZEROPOINT"]
      out["quant_params"].append(self.__convert_quant_params(scale, zero, convert_uint8_to_int8))
    elif buf_dict["buffer_type"] == BufferType.STATIC_DATA:
      # If this is static data, we need to quantize it ourselves
      self.__quantize_static_input(tensor, buf_dict, out)
    elif hasattr(module, "scale"):
      # For most layer types, the scale and zeropoint are stored in the layer's module
      if type(module.scale) == torch.Tensor:
        out["quant_params"].append(self.__convert_quant_params(module.scale[0], module.zero_point[0], convert_uint8_to_int8))
      else:
        out["quant_params"].append(self.__convert_quant_params(module.scale, module.zero_point, convert_uint8_to_int8))
    elif "quant_scale" in op_dict:
      # When the scale and zeropoint aren't stored in the layer's module, they are determined some other way in
      # DNNOperation and stored in the op_dict.
      out["quant_params"].append(self.__convert_quant_params(op_dict["quant_scale"], op_dict["quant_zero"], convert_uint8_to_int8))
    elif op_dict["op_id"] in OPERATIONS_WITH_QUANTIZATION_PRESERVED:
      # When this operation preserves quantization parameters, just copy the input
      # quantization to the output buffer.
      in_buf_idx = op_dict["inputs"][0]
      in_buf = out["buffers"][in_buf_idx]
      in_quant_params = out["quant_params"][in_buf["quant_start_idx"]]
      out["quant_params"].append(copy.deepcopy(in_quant_params))
    else:
      # Output is not quantized, but the next layer might be a quantize layer that adds quantization parameters. Leave
      # the quantization parameters empty for now and hopefully they will get filled in by the next layer. At the end
      # of processing, there is a check to make sure this happened and throw an error if they are still empty.
      buf_dict["quant_start_idx"] = None
      buf_dict["origin_op_id"] = op_dict["op_id"]

  def __quantize_static_input(self, tensor, buf_dict, out):
    """
    Quantize the data for a static input to a layer, store it, and set the buffer's
    quantization parameters.

    tensor - this buffer's Tensor
    buf_dict - the buffer dictionary to update
    out - the DNN output dictionary to store data in
    """
    buf_dict["start_idx"] = len(out["static_data"])

    dtype = tensor.type().scalarType()
    if dtype == 'Float':
      data = tensor.toIValue().numpy().flatten()
    else:
      raise RuntimeError("Unexpected data type for tensor: %s" % dtype)
    
    # determine scale and zeropoint
    if max(data) == min(data) and max(data) > 0:
        scale = max(data) / 127.0
        zeropoint = 0
    elif max(data) == min(data) and max(data) < 0:
        scale = -max(data) / 127.0
        zeropoint = 0
    elif max(data) == min(data) and max(data) == 0:
        scale = 1.0
        zeropoint = 0
    else:
        maxval = max(abs(min(data)), abs(max(data)))
        scale = maxval / 127.0
        zeropoint = 0

    # quantize and store the data
    bytes = []
    for val in data:
      val = np.round(val / scale + zeropoint).astype(np.int8)
      bytes.append(val) 
    out["static_data"].extend(bytes)
    align_byte_array(out["static_data"], 4)

    # set the zeropoint and scale
    buf_dict["quant_start_idx"] = len(out["quant_params"])
    out["quant_params"].append([np.float32(scale), zeropoint])

  def __add_quant_params_for_static_data(self, tensor, op_dict, buf_dict, out):
    """
    Add quantization parameters for static data buffer to the output dictionary

    Parameters:
    tensor - this buffer's Tensor from state_dict
    op_dict - the operation dictionary
    buf_dict - the buffer dictionary to add quantization data to
    out - the output dictionary

    Return:
    The start index of the first parameters for this buffer
    The total number of sets of quantization parameters for this buffer

    Raises:
    RuntimeError if quantization scheme is not supported
    """
    start_idx = len(out["quant_params"])
    if tensor.type() == "torch.FloatTensor":
      # No quantization
      buf_dict["quant_type"] = QuantType.PER_TENSOR
      scales = [1.0]
      zeropoints = [0]
    else:
      if tensor.qscheme() == torch.per_channel_affine:
        if op_dict["op_id"] == DNNOperationID.CONV_2D:
          # Standard (non-depthwise) convolution has one set of quantization parameters per OUTPUT channel, which is
          # actually per batch of data
          buf_dict["quant_type"] = QuantType.PER_BATCH
        else:
          buf_dict["quant_type"] = QuantType.PER_CHANNEL
        scales = tensor.q_per_channel_scales()
        zeropoints = tensor.q_per_channel_zero_points()
      else:
        buf_dict["quant_type"] = QuantType.PER_TENSOR
        scales = [tensor.q_scale()]
        zeropoints = [tensor.q_zero_point()]
    buf_dict["quant_start_idx"] = start_idx
    for i in range(0, len(scales)):
      out["quant_params"].append(self.__convert_quant_params(scales[i], zeropoints[i], False))

  @staticmethod
  def __convert_quant_params(scale, zeropoint, convert_uint8_to_int8):
    """
    Convert PyTorch quantization parameters to values used by DNN firmware (np.float32 and np.int8 in a list)

    Parameters:
    scale - the raw scale value from PyTorch
    zeropoint - the raw zeropoint from PyTorch
    convert_uint8_to_int8 - if True, zeropoint and data are uint8 and needs to be converted to int8

    Returns:
    list containing adjusted scale and zeropoint
    """
    if convert_uint8_to_int8:
      zeropoint = np.int8(np.int32(zeropoint) - 128)
    else:
      zeropoint = np.int8(zeropoint)
    return [np.float32(scale), zeropoint]

  def __get_weight_data(self, tensor, swap_batch_and_chan):
    """
    Get a tensor's weight data as an array of uint8's

    Parameters:
    tensor - the tensor from the PyTorch state_dict
    swap_batch_and_chan - if true, swap the batch (dims[0]) and channel (dims[1]) dimensions, which is required for
      a depthwise convolution's weights

    Returns:
      list of np.uint8 values representing the data as bytes

    Raises:
      RuntimeError if unexpected data type or size received
    """
    dtype = tensor.type()
    size = tensor.size()
    if dtype == 'torch.quantized.QInt8Tensor':
      data = tensor.int_repr().numpy()
      if len(size) == 4:
        if swap_batch_and_chan:
          if self.__mem_order == MemoryOrder.CHANNEL_FIRST:
            # Convert from Chan x Batch x Row x Col to Batch x Row x Col x Chan
            data = data.transpose([1, 2, 3, 0])
          else:
            # Convert from Chan x Batch x Row x Col to Batch x Chan x Row x Col
            data = data.transpose([1, 0, 2, 3])
        else:
          if self.__mem_order == MemoryOrder.CHANNEL_FIRST:
            # Convert from Batch x Chan x Row x Col to Batch x Row x Col x Chan
            data = data.transpose([0, 2, 3, 1])
          else:
            # No need to transpose, data is already Batch x Chan x Row x Col
            pass
      elif len(size) == 3:
        if self.__mem_order == MemoryOrder.CHANNEL_FIRST:
            # Convert from Chan x Row x Col to Row x Col x Chan
          data = data.transpose([1, 2, 0])
        else:
          # No need to transpose, data is already Chan x Row x Col
          pass
      bytes = np.uint8(data.flatten())
    else:
      raise RuntimeError("Unexpected data type for weights tensor: %s" % dtype)
    return bytes

  def __get_bias_data(self, tensor, in_quant, w_quant, w_quant_per_element):
    """
    Get a tensor's bias data as an array of uint8's. The bias terms are quantized
    using the input and weight quantization scales and converted from a float
    to an int32.

    Parameters:
    tensor - the tensor from the PyTorch state_dict
    in_quant - input quantization parameters
    w_quant - weights quantization parameters
    w_quant_per_element - if true, there is a separate weight for each bias element

    Returns:
      list of np.uint8 values representing the data as bytes

    Raises:
      RuntimeError if unexpected data type or size received
    """
    dtype = tensor.type()
    if dtype == 'torch.FloatTensor':
      data = tensor.detach().numpy().flatten()
      bytes = []
      in_scale = in_quant[0]
      i = 0
      for val in data:
        w_scale = w_quant[i][0]
        val = np.round(val / (in_scale * w_scale)).astype(np.int32)
        bytes.extend(val_to_uint8_array(val, 4, self.__endianness))
        if w_quant_per_element:
          i += 1
    else:
      raise RuntimeError("Unexpected data type for bias tensor: %s" % dtype)
    return bytes

  @staticmethod
  def __extend_dims(dims, tensor_id, swap_batch_and_chan):
    """
    If less than 4 dimensions are listed, imply the remaining dimensions to create a 4D buffer

    Parameters:
      dims - the original dimensions (list of 1-4 numbers)
      tensor_id - the ID of this tensor
      swap_batch_and_chan - if true, number of batches dims[0] and number of channels dims[1] are swapped (e.g. for
         depthwise convolution kernel)

    Returns:
      final dimensions (list of 4 numbers)
      number of dimensions that were prepended
      list for reordering inputs, such that dnn_reorder[pytorch_axis_index] = fw_axis_idx

    Raises:
      RuntimeError if too many dimensions were provided
    """
    dim_reorder = []
    if len(dims) == 0:
      dims = [1, 1, 1, 1]
      dim_reorder = []
    elif len(dims) == 1:
      dims = [dims[0], 1, 1, 1]
      dim_reorder = [0]
    elif len(dims)== 2:
      # [Row, Col] --> [Batch, Row, Col, Chan]
      dims = [1, dims[0], dims[1], 1]
      dim_reorder = [1, 2]
    elif len(dims) == 3:
      # [Chan, Row, Col] --> [Batch, Row, Col, Chan]
      dims = [1, dims[1], dims[2], dims[0]]
      dim_reorder = [3, 1, 2]
    elif len(dims) == 4:
      if swap_batch_and_chan:
        # [Chan, Batch, Row, Col] --> [Batch, Row, Col, Chan]
        dims = [dims[1], dims[2], dims[3], dims[0]]
        dim_reorder = [3, 0, 1, 2]
      else:
        # [Batch, Chan, Row, Col] --> [Batch, Row, Col, Chan]
        dims = [dims[0], dims[2], dims[3], dims[1]]
        dim_reorder = [0, 3, 1, 2]
    else:
      raise RuntimeError("Unexpected number of dimensions listed for tensor " +
        "%s. Max: %dD, Actual: %dD" % (
          tensor_id, DNN_BUFFER_MAX_DIMENSIONS, len(dims)))
    return dims, dim_reorder
