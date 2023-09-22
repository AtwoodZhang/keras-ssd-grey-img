# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
from internal.constants import *
from internal.utils import *
from internal.operations.dnn_operation import DNNOperation

class BufferAllocator:
  """
  Class that handles memory allocation for buffers in the DNN
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self):
    self.__scratch_buffers = []

  def allocate_scratch_buffers(self, dnn, config, max_scratch_ram_size):
    """
    Parses the DNN and allocates space in the scratch RAM for each buffer,
    determining when the buffer can be freed and its space can be reused by
    another buffer. The "start_idx" field on each SCRATCH_RAM buffer will be
    modified to point to its allocated space, and a "scratch_ram_size" field
    will be added to the dnn dictionary indicating the maximum size needed at
    any given time

    Parameters:
    dnn - the dnn dictionary
    config - the configuration dictionary
    max_scratch_ram_size - maximum amount of space available for scratch buffers

    Raises:
    RuntimeError if a buffer allocation error occurs
    """
    dnn["scratch_ram_size"] = 0
    self.__scratch_buffers = []
    op_idx = 0
    for op in dnn["operations"]:
      for in_idx in range(0, len(op["inputs"])):
        buf_idx = op["inputs"][in_idx]
        buf = dnn["buffers"][buf_idx]
        if buf["buffer_type"] in [BufferType.SCRATCH_RAM]:
          # When an intermediate result is encountered as an input, assume that
          # we've already created the buffer (from another layer's output) and
          # try to add a connection
          try:
            while buf["parent"] >= 0:
              buf_idx = buf["parent"]
              buf = dnn["buffers"][buf_idx]
            self.__add_scratch_buffer_connection(buf_idx)
          except RuntimeError as err:
            raise RuntimeError("Allocation error on input " +
              " %d of operation %d (%s): %s" % (
              in_idx, op_idx, op["op_id"].name, err))
        elif buf["buffer_type"] == BufferType.MODEL_INPUT:
          # When the input to the entire DNN is encountered, create a buffer for
          # it
          try:
            ram_size = self.__create_scratch_buffer(dnn, buf_idx)
          except RuntimeError as err:
            raise RuntimeError("Allocation error on input " +
              " %d of operation %d (%s): %s" % (
              in_idx, op_idx, op["op_id"].name, err))
          if ram_size > dnn["scratch_ram_size"]:
            dnn["scratch_ram_size"] = ram_size
          # change buffer type to "SCRATCH_RAM", since the input will always be
          # copied into scratch RAM by the firmware
          buf["buffer_type"] = BufferType.SCRATCH_RAM
          while (buf["parent"] >= 0):
            buf = dnn["buffers"][buf["parent"]]
            buf["buffer_type"] = BufferType.SCRATCH_RAM

      for out_idx in range(0, len(op["outputs"])):
        buf_idx = op["outputs"][out_idx]
        # Each time a new output is encountered, create a scratch buffer for it
        try:
          ram_size = self.__create_scratch_buffer(dnn, buf_idx)
        except RuntimeError as err:
          raise RuntimeError("Allocation error on output " +
            " %d of operation %d (%s): %s" % (
            out_idx, op_idx, op["op_id"].name, err))
        if ram_size > dnn["scratch_ram_size"]:
          dnn["scratch_ram_size"] = ram_size
      op_idx += 1
      # If this buffer needs working memory for temporary calculations, create a
      # create a scratch buffer for it
      ram_available = max_scratch_ram_size - ram_size
      working_mem_size = DNNOperation.get_working_memory_size(dnn, op, config, ram_available)
      op["working_mem_size"] = working_mem_size
      if (working_mem_size > 0):
        op["working_mem_addr"], ram_size = self.__create_working_mem_scratch_buffer(dnn, working_mem_size)
        if ram_size > dnn["scratch_ram_size"]:
          dnn["scratch_ram_size"] = ram_size
      # At the end of each operation, remove any scratch rams that are no longer
      # needed
      self.__cleanup_scratch_buffers()
    if len(self.__scratch_buffers) > 0:
      raise RuntimeError("%d scratch buffers were never freed!" % len(self.__scratch_buffers))

  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __create_scratch_buffer(self, dnn, buffer_idx):
    """
    Allocate space for a buffer in scratch RAM and record how many times it will
    be used before it can be freed.

    Parameters:
    dnn - the dnn dictionary
    buffer_idx - index of the buffer that is being used

    Returns:
    The size of scratch RAM after adding this buffer

    Raises:
    RuntimeError if a buffer is referenced after it is freed
    """
    buffers = dnn["buffers"]
    buf = buffers[buffer_idx]
    buffer_size = get_buffer_size(buf)
    if buf["parent"] >= 0:
      parent_idx = buf["parent"]
      while dnn["buffers"][parent_idx]["parent"] >= 0:
        parent_idx = dnn["buffers"][parent_idx]["parent"]
      # If this buffer is a child of a larger buffer, just use the larger buffer
      # for memory allocation purposes
      self.__create_scratch_buffer(dnn, parent_idx)
      buf["start_idx"] = dnn["buffers"][parent_idx]["start_idx"]
    else:
      # Check if a scratch buffer already exists for this buffer
      for scratch_buf in self.__scratch_buffers:
        if scratch_buf["buffer_idx"] == buffer_idx:
          self.__add_scratch_buffer_connection(buffer_idx)
          return self.__scratch_buffers[-1]["start_idx"] + self.__scratch_buffers[-1]["size"]
      start_idx = self.__get_scratch_buffer_start_idx(buffer_size)
      # Add this scratch buffer to self.__scratch_buffers
      self.__finalize_scratch_buffer(buffer_idx, start_idx, buffer_size, buf["num_connections"])
      buf["start_idx"] = start_idx
    # Ensure that scratch ram size is a multiple of 4 bytes to keep temp buffer
    # aligned
    ram_size = self.__scratch_buffers[-1]["start_idx"] + self.__scratch_buffers[-1]["size"]
    ram_size = align_value(ram_size, 4)
    return ram_size

  def __create_working_mem_scratch_buffer(self, dnn, size):
    """
    Create a buffer in scratch RAM for an operation's working memory

    Parameters:
    dnn - the DNN dictionary
    size - the size, in bytes, of working memory needed

    Return:
    start index of this buffer in scratch memory
    total size of the scratch ram
    """
    start_idx = self.__get_scratch_buffer_start_idx(size)
    self.__finalize_scratch_buffer(-1, start_idx, size, 1)
    total_ram_size = self.__scratch_buffers[-1]["start_idx"] + self.__scratch_buffers[-1]["size"]
    # Ensure that scratch ram size is a multiple of 4 bytes to keep temp buffer
    # aligned
    total_ram_size = align_value(total_ram_size, 4)
    return start_idx, total_ram_size

  def __get_scratch_buffer_start_idx(self, buffer_size):
    """
    Find a location in scratch RAM to store a buffer based on its size

    Parameters:
    buffer_size - the total size, in bytes, of the buffer

    Returns:
    the start index in scratch RAM
    """
    # Check if there is a large enough gap in the scratch RAM to fit this data.
    # Otherwise, place this data at the end
    start_idx = 0
    for scratch_buf in self.__scratch_buffers:
      end_idx = scratch_buf["start_idx"]
      if (end_idx - start_idx) >= buffer_size:
        break
      else:
        start_idx = end_idx + scratch_buf["size"]
        # Ensure the start index is properly aligned
        start_idx = align_value(start_idx, 4)
    return start_idx

  def __finalize_scratch_buffer(self, buffer_idx, start_idx, buffer_size, num_connections):
    """
    Finalize a scratch buffer and add it to memory

    Parameters:
    buffer_idx - a buffer index used to identify this buffer
    start_idx - the start location of the buffer in scratch RAM
    buffer_size - the size, in bytes, of the buffer
    num_connections - number of nodes that this buffer connects to in the DNN
    """
    # Store information about this buffer
    scratch_buf = dict()
    scratch_buf["buffer_idx"] = buffer_idx
    scratch_buf["start_idx"] = start_idx
    # Round size to the nearest multiple of 4 bytes
    scratch_buf["size"] = align_value(buffer_size, 4)
    scratch_buf["connections_remaining"] = num_connections - 1
    self.__scratch_buffers.append(scratch_buf)
    # Sort the scratch buffers by address
    self.__scratch_buffers = sorted(self.__scratch_buffers,
                                    key = lambda b: b["start_idx"])

  def __add_scratch_buffer_connection(self, buffer_idx):
    """
    Add a connection to a scratch buffer in RAM, indicating that the buffer has
    been used by a new operation

    Paramaters:
    buffer_idx - index of the buffer that is being used

    Raises:
    RuntimeError if a scratch RAM buffer is not found
    """
    for scratch_buf in self.__scratch_buffers:
      if scratch_buf["buffer_idx"] == buffer_idx:
        scratch_buf["connections_remaining"] -= 1
        return
    raise RuntimeError("Scratch buffer referenced after it was freed: " +
      str(buffer_idx))

  def __cleanup_scratch_buffers(self):
    """
    Remove any buffers from scratch RAM that are no longer needed (no connections
    remaining)

    Parameters:
    scratch_buffers - list of all scratch buffers currently stored in RAM
    """
    self.__scratch_buffers = [
      x for x in self.__scratch_buffers if x["connections_remaining"] > 0]
