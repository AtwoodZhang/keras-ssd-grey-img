# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import logging

from internal.constants import *
from internal.utils import *
from internal.operations.dnn_operation import DNNOperation

class MemoryManager:
  """
  Class that handles converting a DNN to memory structures
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self):
    """
    Constructor
    """
    self.__mem_structs = [{} for sub in range(len(MemoryStructure))]

    self.__mem_regions = [{} for sub in range(len(MemoryRegion))]

  def get_max_scratch_ram_size(self, dnn, cfg):
    """
    Get the total amount of memory available for scratch RAM based on total
    DNN RAM size and the amount of static data that needs to be stored in it.

    This is an estimate and may be larger than the actual availability.
    The act of finalizing the DNN and creating the memory image may use
    more space than expected.

    Once the DNN is finalized and create_memory_structures() has been
    called, recalculate_max_scratch_ram_size() can be called to get a
    better estimate.

    Parameters:
    dnn - the DNN dictionary
    cfg - the configuration dictionary

    Returns:
    maximum size of scratch RAM, in bytes
    """
    if cfg["OUTPUT_MODE"] == OutputMode.ROM:
      return cfg["DNN_RAM_MAX_SIZE"]
    else:
      static_data_size = align_value(len(dnn["static_data"]), 4)

      dnn_ram_available = cfg["DNN_RAM_MAX_SIZE"]
      return dnn_ram_available - static_data_size


  def recalculate_max_scratch_ram_size(self, cfg) -> int:
    """
    Get the total amount of scratch RAM available from a built memory image.

    This walks the memory image as created by create_memory_structures, 
    to figure out how much DNN_RAM was used by pieces other than the
    SCRATCH_RAM.

    This can be used to make a better estimate of available scratch ram
    size than get_max_scratch_ram_size() but only works after the image
    is created.

    Parameters:
    cfg - the configuration dictionary

    Returns:
    The amount of memory used in DNN_RAM, not including scratch_data
    """
    region = self.__mem_regions[MemoryRegion.DNN_RAM.value]   
    used = sum([ self.__mem_structs[s.value]["size"] for s in region["structs"]
                                                  if s != MemoryStructure.SCRATCH_RAM])
    dnn_ram_available = cfg["DNN_RAM_MAX_SIZE"]

    return dnn_ram_available - used

  def create_memory_structures(self, dnn, cfg):
    """
    Convert a DNN dictionary to byte arrays and check if they fit in memory.

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    
    Returns:
    mem_regions - dictionary describing each region of memory
    mem_structs - dictionary describing each structure stored in memory
    
    Raises:
    ValueError if configuration is invalid
    RuntimeError if there is not enough memory available
    """
    # Make sure configuration parameters are valid
    self.__check_alignment(cfg, "SYSTEM_RAM_START_ADDR")
    self.__check_alignment(cfg, "DNN_ROM_START_ADDR")
    self.__check_alignment(cfg, "DNN_RAM_START_ADDR")

    # Initialize memory regions
    mem_regions = [{} for sub in range(len(MemoryRegion))]
    mem_regions[MemoryRegion.SYS_ROM.value]["desc"] = "System ROM"
    mem_regions[MemoryRegion.SYS_ROM.value]["start_addr"] = 0
    mem_regions[MemoryRegion.SYS_ROM.value]["max_size"] = cfg["SYSTEM_ROM_MAX_SIZE"]
    mem_regions[MemoryRegion.SYS_ROM.value]["structs"] = []
    mem_regions[MemoryRegion.SYS_ROM.value]["next_addr"] = mem_regions[MemoryRegion.SYS_ROM.value]["start_addr"]
    mem_regions[MemoryRegion.SYS_ROM.value]["size"] = 0

    mem_regions[MemoryRegion.SYS_RAM.value]["desc"] = "System RAM"
    mem_regions[MemoryRegion.SYS_RAM.value]["start_addr"] = cfg["SYSTEM_RAM_START_ADDR"]
    mem_regions[MemoryRegion.SYS_RAM.value]["max_size"] = cfg["SYSTEM_RAM_MAX_SIZE"]
    mem_regions[MemoryRegion.SYS_RAM.value]["structs"] = []
    mem_regions[MemoryRegion.SYS_RAM.value]["next_addr"] = mem_regions[MemoryRegion.SYS_RAM.value]["start_addr"]
    mem_regions[MemoryRegion.SYS_RAM.value]["size"] = 0

    mem_regions[MemoryRegion.DNN_ROM.value]["desc"] = "DNN ROM"
    mem_regions[MemoryRegion.DNN_ROM.value]["start_addr"] = cfg["DNN_ROM_START_ADDR"]
    mem_regions[MemoryRegion.DNN_ROM.value]["max_size"] = cfg["DNN_ROM_MAX_SIZE"]
    mem_regions[MemoryRegion.DNN_ROM.value]["structs"] = []
    mem_regions[MemoryRegion.DNN_ROM.value]["next_addr"] = mem_regions[MemoryRegion.DNN_ROM.value]["start_addr"]
    mem_regions[MemoryRegion.DNN_ROM.value]["size"] = 0

    mem_regions[MemoryRegion.DNN_RAM.value]["desc"] = "DNN RAM"
    mem_regions[MemoryRegion.DNN_RAM.value]["start_addr"] = cfg["DNN_RAM_START_ADDR"]
    mem_regions[MemoryRegion.DNN_RAM.value]["max_size"] = cfg["DNN_RAM_MAX_SIZE"]
    mem_regions[MemoryRegion.DNN_RAM.value]["structs"] = []
    mem_regions[MemoryRegion.DNN_RAM.value]["next_addr"] = mem_regions[MemoryRegion.DNN_RAM.value]["start_addr"]
    mem_regions[MemoryRegion.DNN_RAM.value]["size"] = 0
    self.__mem_regions = mem_regions

    # Initialize each memory structure
    self.__init_static_data(dnn, cfg)
    self.__init_optable(dnn, cfg)
    self.__init_quant_params(dnn, cfg)
    self.__init_misc_data(dnn, cfg)
    self.__init_test_ram_usage(dnn, cfg)
    self.__init_scratch_ram(dnn, cfg)

    # Return the memory dictionary
    return self.__mem_regions, self.__mem_structs


  def get_memory_report(self):
    """
    Get a string describing the total sizes of memory blocks

    Parameters:
    mem_regions - memory regions dictionary
    mem_structs - memory structures dictionary

    Returns:
    String describing the memory usage
    List of error strings if the DNN does not fit
    """
    errors = []
    mem_usage_report = "";
    mem_usage_report += "Memory Sizes         Bytes Used     Total Available"
    for region in self.__mem_regions:
      if (region["size"] > 0):
        mem_usage_report += "\n"
        mem_usage_report += "---------------------------------------------------\n"
        mem_usage_report += "%-24s%7d             %7d\n" % (
          region["desc"], region["size"], region["max_size"])
        mem_usage_report += "---------------------------------------------------\n"
        for struct in region["structs"]:
          s = self.__mem_structs[struct.value]
          mem_usage_report += "%-24s%7d\n" % (s["desc"], s["size"])
        if region["size"] > region["max_size"]:
          errors.append("Maximum size of %s exceeded!" % (
            region["desc"]))
    return mem_usage_report, errors


  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __check_alignment(self, cfg, key):
    """
    Make sure a value from the configuration file is 4-byte aligned

    Parameters:
    cfg - configuration dictionary
    key - the parameter name

    Raises:
    ValueError if alignment is wrong
    """
    if (cfg[key] % 4) != 0:
      raise ValueError("%s must be 4-byte aligned! Try increasing from %d to %d" % (
        key, cfg[key], align_val(cfg[key], 4)))

  def __init_static_data(self, dnn, cfg):
    """
    Create the Static Data memory structure and add it to a memory region

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    """
    static_data = dict()
    static_data["desc"] = "Weights & Biases"
    static_data["bytes"] = [align_byte_array(dnn["static_data"], 4)]
    static_data["size"] = len(static_data["bytes"][0])
    self.__mem_structs[MemoryStructure.STATIC_DATA.value] = static_data

    if cfg["OUTPUT_MODE"] == OutputMode.ROM:
      self.__add_to_memory(MemoryStructure.STATIC_DATA, static_data["size"], MemoryRegion.DNN_ROM, False)
    else:
      self.__add_to_memory(MemoryStructure.STATIC_DATA, static_data["size"], MemoryRegion.DNN_RAM, False)

  def __init_scratch_ram(self, dnn, cfg):
    """
    Create the Scratch RAM memory structure and add it to a memory region

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    """
    scratch_ram = dict()
    scratch_ram["desc"] = "Scratch RAM"
    scratch_ram["bytes"] = []
      
    scratch_ram["size"] = dnn["scratch_ram_size"]  
  
    self.__mem_structs[MemoryStructure.SCRATCH_RAM.value] = scratch_ram
    self.__add_to_memory(MemoryStructure.SCRATCH_RAM, scratch_ram["size"], MemoryRegion.DNN_RAM, False)
 

  def __init_test_ram_usage(self, dnn, cfg):
    """
    Create of block of DNN RAM to mimic a mem structure overflowing SYS_RAM
    and being put in DNN_RAM.

    This is for test purposes. It is controlled by both by DNN_COMPILER_INTERNAL_USE
    and the optional config file entry TEST_USE_DNN_RAM. 

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    """
    if DNN_COMPILER_INTERNAL_USE and "TEST_USE_DNN_RAM" in cfg:
      test_ram = dict()
      test_ram["desc"] = "Test RAM"
      test_ram["bytes"] = []
      test_ram["size"] = cfg["TEST_USE_DNN_RAM"] 
      
      self.__mem_structs[MemoryStructure.TEST_RAM.value] = test_ram
      self.__add_to_memory(MemoryStructure.TEST_RAM, test_ram["size"], MemoryRegion.DNN_RAM, False)
   

  def __init_optable(self, dnn, cfg):
    """
    Create the Optable memory structure and add it to a memory region

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    """
    endianness = cfg["OUTPUT_ENDIANNESS"]
    reverse_alloc = cfg["REVERSE_ALLOC_SCRATCH_RAM"]
    optable = dict()
    optable["desc"] = "Operation Table"
    header, header_len = self.__optable_header_to_byte_array(dnn, endianness, reverse_alloc)
    buffers, buffers_len = self.__buffers_to_byte_arrays(dnn, endianness)
    operations, operations_len = self.__operations_to_byte_arrays(dnn, endianness)
    optable["bytes"] = []
    optable["bytes"].append(header)
    optable["bytes"].extend(buffers)
    optable["bytes"].extend(operations)
    optable["size"] = header_len + buffers_len + operations_len
    self.__mem_structs[MemoryStructure.OPTABLE.value] = optable

    if cfg["OUTPUT_MODE"] == OutputMode.ROM:
      self.__add_to_memory(MemoryStructure.OPTABLE, optable["size"], MemoryRegion.SYS_ROM, False)
    else:
      self.__add_to_memory(MemoryStructure.OPTABLE, optable["size"], MemoryRegion.SYS_RAM, False)

  def __init_misc_data(self, dnn, cfg):
    """
    Create the Misc Data memory structure and add it to a memory region

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    """
    misc_data = dict()
    misc_data["desc"] = "Miscellaneous Data"
    misc_data["bytes"] = [align_byte_array(dnn["misc_data"], 4)]
    misc_data["size"] = len(misc_data["bytes"][0])
    self.__mem_structs[MemoryStructure.MISC_DATA.value] = misc_data

    if cfg["OUTPUT_MODE"] == OutputMode.ROM:
      fits_in_sys_mem = self.__add_to_memory(MemoryStructure.MISC_DATA, misc_data["size"], MemoryRegion.SYS_ROM, True)
      # If it doesn't fit in System ROM, put it in DNN ROM instead
      if not fits_in_sys_mem:
        logging.warning("Misc Data array does not fit in System ROM. Storing it in DNN ROM instead.")
        self.__add_to_memory(MemoryStructure.MISC_DATA, misc_data["size"], MemoryRegion.DNN_ROM, False)
    else:
      fits_in_sys_mem = self.__add_to_memory(MemoryStructure.MISC_DATA, misc_data["size"], MemoryRegion.SYS_RAM, True)
      # If it doesn't fit in System RAM, put it in DNN RAM instead
      if not fits_in_sys_mem:
        logging.warning("Misc Data array does not fit in System RAM. Storing it in DNN RAM instead.")
        self.__add_to_memory(MemoryStructure.MISC_DATA, misc_data["size"], MemoryRegion.DNN_RAM, False)


  def __init_quant_params(self, dnn, cfg):
    """
    Create the Quant Params memory structure and add it to a memory region

    Parameters:
    dnn - DNN dictionary
    cfg - configuration dictionary
    """
    endianness = cfg["OUTPUT_ENDIANNESS"]
    quant_params = dict()
    quant_params["desc"] = "Quantization Parameters"
    header, header_len = self.__quant_header_to_byte_arrays(dnn, endianness)
    scales, zeros, data_len = self.__quant_params_to_byte_array(dnn, endianness)
    quant_params["bytes"] = [header]
    quant_params["bytes"].extend(scales)
    quant_params["bytes"].extend(zeros)
    quant_params["size"] = header_len + data_len
    self.__mem_structs[MemoryStructure.QUANT_PARAMS.value] = quant_params

    if cfg["OUTPUT_MODE"] == OutputMode.ROM:
      fits_in_sys_mem = self.__add_to_memory(MemoryStructure.QUANT_PARAMS, quant_params["size"], MemoryRegion.SYS_ROM, True)
      # If it doesn't fit in System ROM, put it in DNN ROM instead
      if not fits_in_sys_mem:
        logging.warning("Quantization Parameters do not fit in System ROM. Storing them in DNN ROM instead.")
        self.__add_to_memory(MemoryStructure.QUANT_PARAMS, quant_params["size"], MemoryRegion.DNN_ROM, False)
    else:
      fits_in_sys_mem = self.__add_to_memory(MemoryStructure.QUANT_PARAMS, quant_params["size"], MemoryRegion.SYS_RAM, True)
      # If it doesn't fit in System RAM, put it in DNN RAM instead
      if not fits_in_sys_mem:
        logging.warning("Quantization Parameters do not fit in System RAM. Storing them in DNN RAM instead.")
        self.__add_to_memory(MemoryStructure.QUANT_PARAMS, quant_params["size"], MemoryRegion.DNN_RAM, False)


  def __add_to_memory(self, struct, size, region, check_size):
    """
    Add a given memory structure to a given region of memory

    Parameters:
    struct - the memory structure that is being added (MemoryStructure enum)
    size - the size, in bytes, of the structure
    region - the region of memory to add to (MemoryRegion enum)
    check_size - if True, only add it if there is room for it. If False, try
                 adding it regardless.

    Returns:
    True if the memory was added, False if it was not
    """
    if check_size:
      if (self.__mem_regions[region.value]["size"] + size) > self.__mem_regions[region.value]["max_size"]:
        return False
    self.__mem_structs[struct.value]["start_addr"] = self.__mem_regions[region.value]["next_addr"]
    self.__mem_regions[region.value]["size"] += size
    self.__mem_regions[region.value]["next_addr"] += size
    self.__mem_regions[region.value]["structs"].append(struct)
    return True

  def __quant_header_to_byte_arrays(self, dnn, endianness):
    """
    Create byte array containing the header for the quantization parameters
    memory structure

    Parameters:
    dnn - dnn dictionary
    endianness - Endianness enum value

    Returns:
    header_bytes - byte array representing the header
    total_length - the total number of bytes in header
    """
    header_bytes = val_to_uint8_array(len(dnn["quant_params"]), 4, endianness)
    return header_bytes, 4

  def __quant_params_to_byte_array(self, dnn, endianness):
    """
    Convert array of quantization parameters from compiler output to raw byte
    arrays for each set of parameters

    Parameters:
    dnn - dnn dictionary
    endianness - Endianness enum value

    Returns:
    two list of byte arrays, where each byte array represents a single parameter.
    first is scale values, second is zeropoint values
    total_length - the total number of bytes in both byte arrays combined
    """
    s_byte_arrays = []
    z_byte_arrays = []
    total_length = 0
    for q in dnn["quant_params"]:
      s_bytes = []
      z_bytes = []
      scale_fp = float_to_fp_uint32(q[0], DNN_QUANT_SCALE_Q_FORMAT)
      s_bytes.extend(val_to_uint8_array(scale_fp, 4, endianness))
      z_bytes.append(np.uint8(q[1]))
      total_length += (len(s_bytes) + len(z_bytes))
      s_byte_arrays.append(s_bytes)
      z_byte_arrays.append(z_bytes)
    # Add padding to ensure that the number of quantization parameter sets is a
    # multiple of 4 to align data to 32-bit boundaries.
    while (len(z_byte_arrays) % 4) != 0:
      z_bytes = [np.uint8(0)]
      total_length += len(z_bytes)
      z_byte_arrays.append(z_bytes)
    return s_byte_arrays, z_byte_arrays, total_length

  def __optable_header_to_byte_array(self, dnn, endianness, reverse_alloc):
    """
    Create byte array containing the header for the optable
    memory structure

    Parameters:
    dnn - dnn dictionary
    endianness - Endianness enum value
    reverse_alloc - True if scratch memory is allocated in reverse

    Returns:
    header_bytes - byte array representing the header
    total_length - the total number of bytes in header
    """
    static_data_size = align_value(len(dnn["static_data"]), 4)
    num_buffers = len(dnn["buffers"])
    # If scratch RAM is allocated in reverse, all addresses are in reference to
    # the start of DNN RAM. However, if it is allocated forward, all addresses
    # are in reference to the beginning of the Scratch RAM region.
    if reverse_alloc:
      scratch_addr = self.__mem_regions[MemoryRegion.DNN_RAM.value]["start_addr"]
    else:
      scratch_addr = self.__mem_structs[MemoryStructure.SCRATCH_RAM.value]["start_addr"]

    header_bytes = val_to_uint8_array(static_data_size, 4, endianness)
    header_bytes.extend(val_to_uint8_array(num_buffers, 4, endianness))
    header_bytes.extend(val_to_uint8_array(scratch_addr, 4, endianness))
    return header_bytes, len(header_bytes)

  def __buffers_to_byte_arrays(self, dnn, endianness):
    """
    Convert array of buffer dictionaries from compiler output to raw byte
    arrays for each buffer

    Parameters:
    dnn - dnn dictionary
    endianness - Endianness enum value

    Returns:
    list of byte arrays, where each byte array represents a single buffer
    total_length - total number of bytes for all operations
    """
    buffer_byte_arrays = []
    total_length = 0
    for buf in dnn["buffers"]:
      bytes = []
      bytes.extend(val_to_uint8_array(buf["data_start_offset"], 4, endianness))
      bytes.extend(val_to_uint8_array(buf["batch_size"], 4, endianness))
      bytes.extend(val_to_uint8_array(buf["quant_start_idx"], 2, endianness))
      bytes.extend(val_to_uint8_array(buf["dimensions"][DIMS_NUM_BATCHES], 2, endianness))
      bytes.extend(val_to_uint8_array(buf["data_num_rows"], 2, endianness))
      bytes.extend(val_to_uint8_array(buf["data_num_cols"], 2, endianness))
      bytes.extend(val_to_uint8_array(buf["dimensions"][DIMS_NUM_CHANNELS], 2, endianness))
      bytes.extend(val_to_uint8_array(buf["row_size"], 2, endianness))
      bytes.extend(val_to_uint8_array(buf["buffer_type"].value, 1, endianness))
      bytes.extend(val_to_uint8_array(buf["quant_type"].value, 1, endianness))
      bytes.extend(val_to_uint8_array(buf["bpp_shift"], 1, endianness))
      bytes.extend(val_to_uint8_array(0, 1, endianness))
      total_length += len(bytes)
      buffer_byte_arrays.append(bytes)
    return buffer_byte_arrays, total_length

  def __operations_to_byte_arrays(self, dnn, endianness):
    """
    Convert array of operation dictionaries from compiler output to raw byte
    arrays for each operation

    Parameters:
    dnn - dnn dictionary
    endianness - Endianness enum value

    Returns:
    list of byte arrays, where each byte array represents a single operation
    total_length - total number of bytes for all operations
    """
    op_byte_arrays = []
    total_length = 0
    for op in dnn["operations"]:
      bytes = []
      bytes.append(np.uint8(op["op_id"].value))
      bytes.append(np.uint8(len(op["inputs"])))
      for i in range(0, DNN_OPERATION_MAX_INPUTS):
        if (i < len(op["inputs"])):
          bytes.extend(val_to_uint8_array(op["inputs"][i], 2, endianness))
        else:
          bytes.extend(val_to_uint8_array(0, 2, endianness))
      if len(op["outputs"]) > 0:
        bytes.extend(val_to_uint8_array(op["outputs"][0], 2, endianness))
      else:
        bytes.extend(val_to_uint8_array(0, 2, endianness))
      bytes.extend(val_to_uint8_array(op["working_mem_addr"], 4, endianness))
      param = DNNOperation.op_params_to_byte_array(op, dnn, endianness)
      op["param"] = param
      bytes.extend(val_to_uint8_array(param, 4, endianness))
      total_length += len(bytes) 
      op_byte_arrays.append(bytes)
    # Add an operation with "INVALID" op id at end to signal the end of the
    # array. Ensure that we remain 4-byte aligned
    op_byte_arrays.append([np.uint8(DNNOperationID.INVALID.value) for i in range(0, 4)])
    total_length += 4
    return op_byte_arrays, total_length
