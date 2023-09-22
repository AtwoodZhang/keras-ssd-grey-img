# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
from datetime import datetime
from enum import Enum
import numpy as np
import os
import sys
import logging

from internal.constants import *
from internal.utils import *
from internal.operations.dnn_operation import DNNOperation
from internal.dnn_processing.memory_manager import MemoryManager

# Template file used to generate output .h & .c files
C_DNN_AUTO_TEMPLATE_FILENAME = "auto_template.c"
H_DNN_COMMON_AUTO_TEMPLATE_FILENAME = "dnn_common_auto_template.h"

# Number of bytes of data to print per line when generating C files
C_FILE_BYTES_PER_LINE = 16

# Number of spaces for each indentation level
SPACES_PER_INDENT = 4

class OutputWriter:
  """
  This class handles generating .c source or other output files from the output
  of the TFLite file reader.
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self, cfg_dict):
    """
    Constructor

    Parameters:
    cfg_dict - dictionary containing configuration parameters
    """
    self.__in_dir = cfg_dict["TEMPLATE_DIRECTORY"]
    self.__out_dir = cfg_dict["OUTPUT_DIRECTORY"]
    self.__out_mode = cfg_dict["OUTPUT_MODE"]
    self.__endianness = cfg_dict["OUTPUT_ENDIANNESS"]
    self.__num_postproc_compare_cols = cfg_dict["POSTPROC_COMPARE_VALS_NUM_COLS"]
    self.__cfg = cfg_dict
    self.__run_time = datetime.now().astimezone()


  def write(self, dnn_name, dnn, mem_regions, mem_structs, mem_report, inc_instructions):
    """
    Writes dnn data out to files in out_dir based on the output mode.

    Parameters:
    dnn_name - name of this DNN, used as a prefix for files and variables
    dnn - DNN dictionary
    mem_regions - list of dicts describing all memory regions
    mem_structs - list of dicts describing all structures stored in memory regions
    mem_report - string describing the memory usage of the DNN
    inc_instructions - if True, include information about all VPU instruction 
                       counts in txt file output

    Raises:
    RuntimeError if invalid mode provided or file I/O error occurred
    ValueError if invalid value found in configuration file
    """
    # if the output directory doesn't exist, create it now
    if not os.path.exists(self.__out_dir):
      try:
        os.makedirs(self.__out_dir)
      except OSError as e:
        raise RuntimeError("Could not create output directory: %s. %s" % (
          self.__out_dir, e))

    if DNN_COMPILER_INTERNAL_USE and self.__out_mode == OutputMode.ROM:
      self.__write_rom_files(dnn_name, mem_regions, mem_structs)
    elif self.__out_mode == OutputMode.I2C:
      self.__write_i2c_files(dnn_name, mem_regions, mem_structs)
      self.__write_sim_files(dnn_name, mem_regions, mem_structs)
      self.__write_registers_for_sim(dnn_name, dnn, mem_regions, mem_structs)

    else:
      raise RuntimeError("Unsupported value of OUTPUT_MODE in configuration: %s" % self.__out_mode.name)

    self.__write_txt_file(dnn_name, dnn, mem_regions, mem_structs, mem_report, inc_instructions)

  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __write_rom_files(self, dnn_name, mem_regions, mem_structs):
    """
    Write .c files that can be included in ROM firmware build

    Parameters:
    dnn_name - DNN name (used for naming files and data structures)
    mem_regions - memory region dictionary
    mem_structs - memory structures dictionary
  
    Raises:
    RuntimeError if file cannot be read or written
    """
    # Check if memory structures are stored in system memory or DNN memory
    if (
      MemoryStructure.QUANT_PARAMS in mem_regions[MemoryRegion.DNN_RAM.value]["structs"]) or (
      MemoryStructure.QUANT_PARAMS in mem_regions[MemoryRegion.DNN_ROM.value]["structs"]):
      quant_in_dnn_mem = 1
    else:
      quant_in_dnn_mem = 0
    if (
      MemoryStructure.MISC_DATA in mem_regions[MemoryRegion.DNN_RAM.value]["structs"]) or (
      MemoryStructure.MISC_DATA in mem_regions[MemoryRegion.DNN_ROM.value]["structs"]):
      misc_in_dnn_mem = 1
    else:
      misc_in_dnn_mem = 0

    try:
      in_filename = os.path.join(self.__in_dir, H_DNN_COMMON_AUTO_TEMPLATE_FILENAME)
      out_filename = os.path.join(self.__out_dir, "dnn_common_auto.h")
      with open(in_filename, "r") as in_file, open(out_filename, "w") as out_file:
        self.__write_from_template(
          in_file, out_file, dnn_name, mem_regions, mem_structs, quant_in_dnn_mem, misc_in_dnn_mem)

      in_filename = os.path.join(self.__in_dir, C_DNN_AUTO_TEMPLATE_FILENAME)
      out_filename = os.path.join(self.__out_dir, "dnn_op_func_table_auto.c")
      with open(in_filename, "r") as in_file, open(out_filename, "w") as out_file:
        self.__write_from_template(
          in_file, out_file, dnn_name, mem_regions, mem_structs, quant_in_dnn_mem, misc_in_dnn_mem)
        self.__write_op_func_table(out_file)

      in_filename = os.path.join(self.__in_dir, C_DNN_AUTO_TEMPLATE_FILENAME)
      out_filename = os.path.join(self.__out_dir, dnn_name + "_optable_auto.c")
      with open(in_filename, "r") as in_file, open(out_filename, "w") as out_file:
        self.__write_from_template(
          in_file, out_file, dnn_name, mem_regions, mem_structs, quant_in_dnn_mem, misc_in_dnn_mem)
        self.__write_optable(out_file, dnn_name, mem_structs[MemoryStructure.OPTABLE.value])

      if not quant_in_dnn_mem:
        in_filename = os.path.join(self.__in_dir, C_DNN_AUTO_TEMPLATE_FILENAME)
        out_filename = os.path.join(self.__out_dir, dnn_name + "_quant_params_auto.c")
        with open(in_filename, "r") as in_file, open(out_filename, "w") as out_file:
          self.__write_from_template(
            in_file, out_file, dnn_name, mem_regions, mem_structs, quant_in_dnn_mem, misc_in_dnn_mem)
          self.__write_quant_params(out_file, dnn_name, mem_structs[MemoryStructure.QUANT_PARAMS.value])

      if mem_structs[MemoryStructure.MISC_DATA.value]["size"] > 0:
        if not misc_in_dnn_mem:
          in_filename = os.path.join(self.__in_dir, C_DNN_AUTO_TEMPLATE_FILENAME)
          out_filename = os.path.join(self.__out_dir, dnn_name + "_misc_data_auto.c")
          with open(in_filename, "r") as in_file, open(out_filename, "w") as out_file:
            self.__write_from_template(
              in_file, out_file, dnn_name, mem_regions, mem_structs, quant_in_dnn_mem, misc_in_dnn_mem)
            self.__write_misc_data(out_file, dnn_name, mem_structs[MemoryStructure.MISC_DATA.value])
    except OSError as e:
      raise RuntimeError("Failed to create %s from template %s: %s" % (
        in_filename, out_filename, e))

    # Write out the memory image for the DNN memory
    try:
      dnn_filename = os.path.join(self.__out_dir, dnn_name + "_dnn_memory.bin")
      with open(dnn_filename, "wb") as out_file:
        self.__write_memory_structure(out_file, mem_regions[MemoryRegion.DNN_ROM.value], mem_structs)
    except OSError as e:
      raise RuntimeError("Failed to write DNN memory image to %s: %s" % (
        dnn_filename, e))

  def __write_sim_files(self, dnn_name, mem_regions, mem_structs):
    """
    Write memory images to .bin files for simulation

    Parameters:
    dnn_name - DNN name (used for naming files and data structures)
    mem_regions - memory region dictionary
    mem_structs - memory structures dictionary

    Raises:
    RuntimeError if file cannot be written
    """
    sys_filename = os.path.join(self.__out_dir, dnn_name + "_system_memory.bin")
    dnn_filename = os.path.join(self.__out_dir, dnn_name + "_dnn_memory.bin")
    sys_ram = mem_regions[MemoryRegion.SYS_RAM.value]
    dnn_ram = mem_regions[MemoryRegion.DNN_RAM.value]

    try:
      with open(sys_filename, "wb") as out_file:
        self.__write_memory_structure(out_file, sys_ram, mem_structs)
    except OSError as e:
      raise RuntimeError("Failed to write system memory image to %s: %s" % (
        sys_filename, e))
    
    try:
      with open(dnn_filename, "wb") as out_file:
        self.__write_memory_structure(out_file, dnn_ram, mem_structs)
    except OSError as e:
      raise RuntimeError("Failed to write DNN memory image to %s: %s" % (
        dnn_filename, e))
  

  def __write_i2c_files(self, dnn_name, mem_regions, mem_structs):
    """
    Write I2C sequence to load this DNN to .bin file

    Parameters:
    dnn_name - DNN name (used for naming files and data structures)
    mem_regions - memory region dictionary
    mem_structs - memory structures dictionary

    Raises:
    RuntimeError if file cannot be written
    ValueError if invalid value found in configuration file
    """
    filename = os.path.join(self.__out_dir, dnn_name + "_load_sequence_i2c.bin")
    
    try:
      with open(filename, "wb") as out_file:
        self.__write_i2c_init_seq(out_file)
        self.__write_reg_setting_seq(out_file, mem_structs)
        self.__write_mem_region_seq(out_file, mem_regions[MemoryRegion.DNN_RAM.value], mem_structs)
        self.__write_mem_region_seq(out_file, mem_regions[MemoryRegion.SYS_RAM.value], mem_structs)
    except OSError as e:
      raise RuntimeError("Failed to write I2C load sequence to %s: %s" % (
        filename, e))


  def __write_registers_for_sim(self, dnn_name, dnn, mem_regions, mem_structs):
    """
    Write the registers file for the simulator

    Parameters:
    dnn_name - DNN name (used for naming files and data structures)
    dnn - DNN dictionary
    mem_regions - memory region dictionary
    mem_structs - memory structure dictionary

    Raises:
    RuntimeError if file cannot be written
    """
    # Registers file
    registers_filename = os.path.join(self.__out_dir, dnn_name + "_registers.txt")
    registers_lines = []

    registers_lines.append("ENDIANNESS              = %s" % (
      self.__endianness.name))

    registers_lines.append("")

    registers_lines.append("SYS_ADDR                = 0x%08x" % (
      mem_regions[MemoryRegion.SYS_RAM.value]["start_addr"]))
    registers_lines.append("DNN_ADDR                = 0x%08x" % (
      mem_regions[MemoryRegion.DNN_RAM.value]["start_addr"]))

    registers_lines.append("")

    registers_lines.append("DNN_WEIGHT_ADDR         = 0x%08x" % (
      mem_structs[MemoryStructure.STATIC_DATA.value]["start_addr"]))
    registers_lines.append("DNN_OPTABLE_ADDR        = 0x%08x" % (
      mem_structs[MemoryStructure.OPTABLE.value]["start_addr"]))

    registers_lines.append("DNN_QUANT_PARAM_ADDR    = 0x%08x" % (
      mem_structs[MemoryStructure.QUANT_PARAMS.value]["start_addr"]))
    registers_lines.append("DNN_MISC_DATA_ADDR      = 0x%08x" % (
      mem_structs[MemoryStructure.MISC_DATA.value]["start_addr"]))

    if "spu" in dnn:
      registers_lines.append("DNN_PARAM_TABLES_ADDR   = 0x%08x" % (
        mem_structs[MemoryStructure.PARAM_TABLES.value]["start_addr"]))
      registers_lines.append("DNN_SPU_CODE_ADDR       = 0x%08x" % (
        mem_structs[MemoryStructure.SPU_CODE.value]["start_addr"]))

    for i in range(0, len(self.__cfg["POSTPROC_THRESHOLD_VALS"])):
      registers_lines.append("DNN_POSTPROC_THRESHOLD%d = 0x%02x" % (
        i, self.__cfg["POSTPROC_THRESHOLD_VALS"][i]))

    registers_lines.append("")

    try:
      with open(registers_filename, "w") as registers_file:
        registers_file.write("\n".join(registers_lines))
    except OSError as e:
      raise RuntimeError("Failed to create registers file %s: %s" % (
        registers_filename, e)) 


  def __write_txt_file(self, dnn_name, dnn, mem_regions, mem_structs, mem_report, inc_instructions):
    """
    Write information about this DNN to a human readable text file

    Parameters:
    dnn_name - DNN name (used for naming files and data structures)
    dnn - DNN dictionary
    mem_regions - memory region dictionary
    mem_structs - memory structure dictionary
    mem_report - String containing information about memory usage
    inc_instructions - If true, include VPU instruction usage information

    Raises:
    RuntimeError if file cannot be written
    """
    out_filename = os.path.join(self.__out_dir, dnn_name + "_summary.txt")
    if DNN_COMPILER_INTERNAL_USE:
      inputs_filename = os.path.join(self.__out_dir, dnn_name + "_inputs.txt")
      inputs_lines = []
      outputs_filename = os.path.join(self.__out_dir, dnn_name + "_outputs.txt")
      outputs_lines = []

    processing_time = dict()
    instr_cnts = dict()
    instr_cycles = dict()

    # Print size information to file
    lines = []
    lines.append("-------------------------------------------------")
    lines.append("MEMORY USAGE")
    lines.append("-------------------------------------------------")
    lines.append(mem_report)

    lines.append("-------------------------------------------------")
    lines.append("REGISTER SETTINGS")
    lines.append("-------------------------------------------------")
    # if this section changes update the function __write_registers_for_sim
    # above. These two pieces must match.
    lines.append("DNN_WEIGHT_ADDR         = 0x%08x" % (
      mem_structs[MemoryStructure.STATIC_DATA.value]["start_addr"]))
    lines.append("DNN_OPTABLE_ADDR        = 0x%08x" % (
      mem_structs[MemoryStructure.OPTABLE.value]["start_addr"]))
    lines.append("DNN_QUANT_PARAM_ADDR    = 0x%08x" % (
      mem_structs[MemoryStructure.QUANT_PARAMS.value]["start_addr"]))
    lines.append("DNN_MISC_DATA_ADDR      = 0x%08x" % (
      mem_structs[MemoryStructure.MISC_DATA.value]["start_addr"]))
    for i in range(0, len(self.__cfg["POSTPROC_THRESHOLD_VALS"])):
      lines.append("DNN_POSTPROC_THRESHOLD%d = 0x%02x" % (
        i, self.__cfg["POSTPROC_THRESHOLD_VALS"][i]))
    lines.append("")
    lines.append("-------------------------------------------------")
    lines.append("STRUCTURE LOAD ADDRESSES")
    lines.append("-------------------------------------------------")
    if self.__out_mode != OutputMode.ROM:
      lines.append("%s_system_memory.bin = 0x%08x" % (dnn_name,
        mem_structs[MemoryStructure.OPTABLE.value]["start_addr"]))
    lines.append("%s_dnn_memory.bin    = 0x%08x" % (dnn_name,
      mem_structs[MemoryStructure.STATIC_DATA.value]["start_addr"]))

    lines.append("")
    lines.append("-------------------------------------------------")
    lines.append("DNN OPERATIONS SUMMARY")
    lines.append("-------------------------------------------------")
    # Print operations to file
    op_num = 0
    if self.__cfg["REVERSE_ALLOC_SCRATCH_RAM"]:
      start_addr = mem_regions[MemoryRegion.DNN_RAM.value]["start_addr"]
    else:
      start_addr = mem_structs[MemoryStructure.SCRATCH_RAM.value]["start_addr"]
    for op in dnn["operations"]:
      # Convert op id to op name
      op_name = op["op_id"].name
      param_val = op["param"] 

      lines.append("OPERATION %d: %s (Working Mem: 0x%08x, Params: 0x%08x)" % (
        op_num, op_name,
        start_addr + op["working_mem_addr"],
        param_val
      ))
      param_str = DNNOperation.op_params_to_string(dnn, op)
      if param_str != "":
        lines.append("  Parameters:")
        lines.append("    %s" % (param_str))
      lines.append("  Inputs:")
      lines.append(self.__get_txt_buffer_header())
      for inp in op["inputs"]:
        buf = dnn["buffers"][inp]
        lines.append(self.__buffer_to_string(buf, inp, dnn, mem_regions, mem_structs))
        if DNN_COMPILER_INTERNAL_USE and (buf["buffer_type"] == BufferType.SCRATCH_RAM):
          inputs_lines.append(self.__buffer_to_io_string(op_num, buf))
      if len(op["outputs"]) > 0:
        lines.append("  Outputs:")
        lines.append(self.__get_txt_buffer_header())
        for out in op["outputs"]:
          buf = dnn["buffers"][out]
          lines.append(self.__buffer_to_string(buf, out, dnn, mem_regions, mem_structs))
          if DNN_COMPILER_INTERNAL_USE:
            outputs_lines.append(self.__buffer_to_io_string(op_num, buf))
      proc_time = DNNOperation.get_processing_time(dnn, op, instr_cnts, instr_cycles)
      lines.append("  Est. Processing Time: %.3f ms" % proc_time)
      if op_name in processing_time:
        processing_time[op_name] += proc_time
      else:
        processing_time[op_name] = proc_time
      lines.append("");
      op_num += 1
    # Add performance summary
    lines.append("-------------------------------------------------")
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-------------------------------------------------")
    lines.append("%-25s%s" % ("Operation Type", "Total Time (ms)"))
    # Record VSUB_C that is used to copy the image out of SIF RAM and convert
    # it from unsigned to signed
    cycles = record_instruction("VSUB_C (Scalar 1)", 1, DNN_INPUT_COLS*DNN_INPUT_ROWS, instr_cnts, instr_cycles)
    total_processing_time = cycles_to_ms(cycles)
    for op_name in processing_time:
      lines.append("%-25s%.3f" % (op_name, processing_time[op_name]))
      total_processing_time += processing_time[op_name]
    lines.append("----------")
    lines.append("%-25s%.3f" % ("TOTAL", total_processing_time))
    if inc_instructions:
      lines.append("")
      lines.append("-------------------------------------------------")
      lines.append("VPU INSTRUCTIONS")
      lines.append("-------------------------------------------------")
      lines.append("%-25s%-10s%s" % ("Instruction", "Count", "Cycles"))
      for instr_name in sorted(instr_cnts.keys()):
        lines.append("%-25s%-10d%d" % (instr_name, instr_cnts[instr_name], instr_cycles[instr_name]))

    # Write out the file
    try:
      with open(out_filename, "w") as out_file:
        out_file.write("\n".join(lines))
    except OSError as e:
      raise RuntimeError("Failed to create summary file %s: %s" % (
        out_filename, e))

    if DNN_COMPILER_INTERNAL_USE:
      try:
        with open(inputs_filename, "w") as inputs_file:
          inputs_file.write("\n".join(inputs_lines))
      except OSError as e:
        raise RuntimeError("Failed to create inputs file %s: %s" % (
          inputs_filename, e))
      try:
        with open(outputs_filename, "w") as outputs_file:
          outputs_file.write("\n".join(outputs_lines))
      except OSError as e:
        raise RuntimeError("Failed to create outputs file %s: %s" % (
          outputs_filename, e)) 

  def __write_from_template(self, in_file, out_file, dnn_name, mem_regions, mem_structs, quant_in_dnn_mem, misc_in_dnn_mem):
    """
    Write out C file header by replacing strings in template file

    Parameters:
    in_file - template file handle, opened for reading
    out_file - output file handle, opened for writing
    dnn_name - the name of this DNN instance
    mem_structs - memory structure dictionary
    mem_regions - memory regions dictionary
    quant_in_dnn_mem - true if quantization parameters go in DNN memory
    misc_in_dnn_mem - true if misc data goes in DNN memory
    """
    for line in in_file:
      if "%ENUMS%" in line:
        self.__write_enums(out_file)
      elif "%DNN_OP_FUNCS%" in line:
        self.__write_op_funcs(out_file, line)
      else:
        # Get sizes of structures without headers
        quant_params_size = mem_structs[MemoryStructure.QUANT_PARAMS.value]["size"]
        quant_params_size -= len(mem_structs[MemoryStructure.QUANT_PARAMS.value]["bytes"][0])
        optable_size = mem_structs[MemoryStructure.OPTABLE.value]["size"]
        optable_size -= len(mem_structs[MemoryStructure.OPTABLE.value]["bytes"][0])

        line = line.replace("%YEAR%", self.__run_time.strftime("%Y"))
        line = line.replace("%TIME%", self.__run_time.strftime("%Y-%m-%d %H:%M:%S %Z (%z)"))
        line = line.replace("%AUTO_GEN%", "This is an auto-generated file. DO NOT HAND EDIT.")
        line = line.replace("%CMD%", " ".join(sys.argv[:]))
        line = line.replace("%NAME%", dnn_name)
        line = line.replace("%SCALE_Q_FORMAT%", str(DNN_QUANT_SCALE_Q_FORMAT))
        line = line.replace("%OFFSET_Q_FORMAT%", str(DNN_QUANT_OFFSET_Q_FORMAT))
        line = line.replace("%RELU_NONE%", hex(RELU_NONE))
        line = line.replace("%RELU_NO_MAX_CLIP%", hex(RELU_NO_MAX_CLIP))
        line = line.replace("%GENERATE_PROPOSALS_END%", hex(GENERATE_PROPOSALS_END))
        line = line.replace("%MISC_DATA_SIZE%", str(mem_structs[MemoryStructure.MISC_DATA.value]["size"]))
        line = line.replace("%QUANT_IN_DNN_ROM%", str(quant_in_dnn_mem))
        line = line.replace("%COMPARE_COLS%", str(self.__num_postproc_compare_cols))
        line = line.replace("%ANCHOR_IN_Q_FORMAT%", str(DNN_ANCHOR_IN_Q_FORMAT))
        line = line.replace("%ANCHOR_DECODE_Q_FORMAT%", str(DNN_ANCHOR_DECODE_Q_FORMAT))
        line = line.replace("%ROI_SCALE_Q_FORMAT%", str(DNN_ROI_SCALE_Q_FORMAT))
        line = line.replace("%POSTPROC_THRESHOLD_Q_FORMAT%", str(DNN_POSTPROC_THRESHOLD_Q_FORMAT))
        line = line.replace("%MISC_IN_DNN_ROM%", str(misc_in_dnn_mem))
        line = line.replace("%MAX_INPUTS%", str(DNN_OPERATION_MAX_INPUTS))
        line = line.replace("%NUM_OP_IDS%", str(len(DNNOperationID)-1))
        line = line.replace("%QUANT_PARAMS_SIZE%", str(quant_params_size))
        # Get the size of the optable without the 12 byte header
        line = line.replace("%OPTABLE_SIZE%", str(optable_size))
        out_file.write(line)

  def __write_enums(self, file):
    """
    Convert python enums to C enums and write them out to file

    Inputs:
    file - handle of file to write to
    """
    self.__write_enum("DNN_BUFFER_TYPE", BufferType, file)
    self.__write_enum("DNN_QUANT_TYPE", QuantType, file)
    self.__write_enum("DNN_BUFFER_AXIS", BufferAxis, file)
    self.__write_enum("DNN_OP_ID", DNNOperationID, file)
    self.__write_enum("DNN_CONV_MODE", ConvMode, file)
    self.__write_enum("DNN_ARITHMETIC_MODE", ArithmeticMode, file)
    self.__write_enum("DNN_ROI_POOL_MODE", ROIPoolMode, file)
    self.__write_enum("DNN_ANCHOR_BOX_INPUT", AnchorBoxInput, file)
    self.__write_enum("DNN_ANCHOR_BOX_OUTPUT", AnchorBoxOutput, file)
    self.__write_enum("DNN_THRESHOLD_COMPARE_MODE", ThresholdCompareMode, file)
    self.__write_enum("DNN_THRESHOLD_REPORT_MODE", ThresholdReportMode, file)
    self.__write_enum("DNN_THRESHOLD_REPORT_FIELD", ThresholdReportField, file)

  def __write_enum(self, name, enum_class, file):
    """
    Write an enum outto the file

    Parameters:
    name - typename and prefix to use on values
    enum_class - the class to write out
    file - handle of file to write to
    """
    line = "\ntypedef enum TAGE_%s {\n" % name
    file.write(line)
    for member in list(enum_class):
      item_name = "%s_%s" % (name, member.name)
      line = "%s%-40s = 0x%02x,\n" % (self.__indent(1), item_name, member.value)
      file.write(line)
    line = "} E_%s;\n" % name
    file.write(line)

  def __write_op_func_table(self, file):
    """
    Write the operation function table to a file

    Parameters:
    file - handle of file to write to
    """
    line = "const DNN_OP_FUNC_T dnn_op_func_table[DNN_NUM_OPERATION_IDS] = {\n"
    file.write(line)
    for member in list(DNNOperationID)[:-1]:
      func_name = "dnn_op_%s," % (member.name.lower())
      line = "%s%-40s /* DNN_OP_ID_%s */\n" % (self.__indent(1), func_name, member.name)
      file.write(line)
    file.write("};\n\n")

  def __write_op_funcs(self, file, line):
    """
    Replace a line with all operation function declarations

    Parameters:
    file - handle of file to write to
    line - the original line, with %DNN_OP_FUNCS% tag in it
    """
    for member in list(DNNOperationID)[:-1]:
      func_name = "dnn_op_%s" % (member.name.lower())
      new_line = line.replace("%DNN_OP_FUNCS%", func_name)
      file.write(new_line)
    file.write("\n")

  def __write_quant_params(self, out_file, dnn_name, quant_params):
    """
    Write out quantization parameters structure to C file

    Parameters:
    out_file - output file handle, opened for writing
    dnn_name - the name of this DNN instance
    quant_params - the memory structure dictionary for the quant_params
    """
    bytes = quant_params["bytes"]
    num_quant_params = uint8_array_to_val(bytes[0], self.__endianness)
    s_bytes = bytes[1:(num_quant_params+1)]
    z_bytes = bytes[num_quant_params+1:]
    out_file.write("const S_DNN_QUANT_PARAMS %s = {\n" % self.__get_quant_params_name(dnn_name))
    out_file.write("%s%s, // Num Parameters\n" % (self.__indent(1), str(num_quant_params)))
    out_file.write("%s{\n" % self.__indent(1))
    out_file.write("%s// Scale values\n%s" % (self.__indent(2), self.__indent(2)))
    for i in range(0, len(s_bytes)):
      bytes = s_bytes[i]
      out_file.write(self.__uint8_array_to_string(bytes))
      if (i+1) % (C_FILE_BYTES_PER_LINE/4) == 0:
        out_file.write(",\n%s" % self.__indent(2))
      else:
        out_file.write(", ")
    out_file.write("\n%s// Zeropoint values\n%s" % (self.__indent(2), self.__indent(2)))
    for i in range(0, len(z_bytes)):
      bytes = z_bytes[i]
      out_file.write(self.__uint8_array_to_string(z_bytes[i]))
      if i % C_FILE_BYTES_PER_LINE == C_FILE_BYTES_PER_LINE-1:
        out_file.write(",\n%s" % self.__indent(2))
      else:
        out_file.write(", ")
    out_file.write("\n%s}\n" % self.__indent(1))
    out_file.write("};\n")

  def __write_misc_data(self, out_file, dnn_name, misc_data):
    """
    Write out misc data array to C file

    Parameters:
    out_file - output file handle, opened for writing
    dnn_name - the name of this DNN instance
    misc_data - the memory structure dictionary for the misc_data

    """
    # Convert numpy.uint8 array to array of strings with hex formatting (0xXX)
    hex_strings = []
    for bytes in misc_data["bytes"]:
      hex_strings = ["{0:#0{1}x}".format(x,4) for x in bytes]

    # Convert array of values to an array of lines to write to the file, where
    # each line contains C_FILE_BYTES_PER_LINE values
    lines = []
    for i in range(0, len(hex_strings), C_FILE_BYTES_PER_LINE):
      lines.append(", ".join(hex_strings[i:i+C_FILE_BYTES_PER_LINE]))

    # Write a combined string that includes all lines of data
    out_file.write("const uint8_t %s[] = {\n" % self.__get_misc_data_name(dnn_name));
    out_file.write(self.__indent(1))
    out_file.write((",\n"+self.__indent(1)).join(lines))
    out_file.write("\n};\n")

  def __write_optable(self, out_file, dnn_name, optable):
    """
    Write out operation data structure to a C file

    Parameters:
    out_file - output file handle, opened for writing
    dnn_name - the name of this DNN instance
    optable - the memory structure dictionary for the optable
    """
    bytes = optable["bytes"]
    # Read field from header
    static_data_size = uint8_array_to_val(bytes[0][0:4], self.__endianness)
    num_buffers = uint8_array_to_val(bytes[0][4:8], self.__endianness)
    scratch_addr = uint8_array_to_val(bytes[0][8:12], self.__endianness)
    # Break up bytes list into buffers and operations
    buffer_bytes = bytes[1:(num_buffers+1)]
    op_bytes = bytes[num_buffers+1:]

    out_file.write("const S_DNN_OPTABLE %s = {\n" % self.__get_optable_name(dnn_name));
    out_file.write("%s%-11s // Static Data Size\n" % (self.__indent(1), str(static_data_size)+","))
    out_file.write("%s%-11s // Number of Buffers\n" % (self.__indent(1), str(num_buffers)+","))
    out_file.write("%s0x%08x, // Scratch RAM Start Address\n" % (self.__indent(1), scratch_addr))
    out_file.write("%s{\n" % self.__indent(1))
    # Write out buffer descriptors
    out_file.write("%s// Buffer Descriptors\n" % self.__indent(2))
    out_file.write("%s// %-20s| %-22s| %-10s| %-10s| %-10s| %-10s| %-10s| %-10s|%-5s|%-5s|%-5s|%-5s\n" % (
      self.__indent(2), "Start Offset", "Batch Stride", "Quant Idx", "# Bat", "# Rows", "# Cols", "# Chan",
      "Row Stride", "BType", "QType", "BPP", "Res"))
    # Write out one buffer per line
    for buf in buffer_bytes:
      out_file.write("%s%s,\n" % (self.__indent(2), self.__uint8_array_to_string(buf)))
    # Write out header for operations structure
    out_file.write("%s// Operations\n" % self.__indent(2))
    out_file.write("%s//%-3s| %-4s| %-10s| %-10s| %-10s| %-10s| %-10s| %-10s| %-10s| %-22s| %-22s|\n" % (
      self.__indent(2), "Op", "# In", "Input[0]", "Input[1]", "Input[2]", "Input[3]", "Input[4]", "Input[5]", "Output",
      "Working Mem Addr", "Params"))
    # Write out one operation per line
    for op in op_bytes:
      out_file.write("%s%s,\n" % (self.__indent(2), self.__uint8_array_to_string(op)))
    out_file.write("%s}\n" % self.__indent(1))
    out_file.write("};\n")

  def __write_i2c_init_seq(self, out_file):
    """
    Write the i2c initialization sequence provided in the configuration file to
    an i2c load sequence file

    Parameters:
    out_file - the file to write to

    Raises:
    ValueError if an integer in the i2c init seq is out of range (0 <= x <= 255)
    """
    try:
      i2c_init_seq = bytearray(self.__cfg["I2C_INIT_SEQ"])
      out_file.write(i2c_init_seq)
    except ValueError as e:
      raise ValueError("Invalid value found in I2C_INIT_SEQ from config file: [%s]. %s" % (
        ", ".join(str(b) for b in self.__cfg["I2C_INIT_SEQ"]), e))

  def __write_reg_setting_seq(self, out_file, mem_structs):
    """
    Add DNN Register setting writes to the I2C initialization sequence file

    Parameters:
    out_file - the file to write to
    mem_structs - dictionary describing memory structures in the DNN
    """
    bytes = []
    try:
      name = "REG_DNN_WEIGHT_ADDR"
      addr = self.__cfg[name]
      val = mem_structs[MemoryStructure.STATIC_DATA.value]["start_addr"]
      num_bytes = 4
      bytes.extend(self.__get_write_sequence(addr, val, num_bytes))

      name = "REG_DNN_OPTABLE_ADDR"
      addr = self.__cfg[name]
      val = mem_structs[MemoryStructure.OPTABLE.value]["start_addr"]
      num_bytes = 4
      bytes.extend(self.__get_write_sequence(addr, val, num_bytes))

      name = "REG_DNN_QUANT_PARAM_ADDR"
      addr = self.__cfg[name]
      val = mem_structs[MemoryStructure.QUANT_PARAMS.value]["start_addr"]
      num_bytes = 4
      bytes.extend(self.__get_write_sequence(addr, val, num_bytes))

      name = "REG_DNN_MISC_DATA_ADDR"
      addr = self.__cfg[name]
      val = mem_structs[MemoryStructure.MISC_DATA.value]["start_addr"]
      num_bytes = 4
      bytes.extend(self.__get_write_sequence(addr, val, num_bytes))

      name = "REG_DNN_POSTPROC_THRESHOLD0"
      addr = self.__cfg[name]
      num_bytes = 1
      for i in range(0, len(self.__cfg["POSTPROC_THRESHOLD_VALS"])):
        val = self.__cfg["POSTPROC_THRESHOLD_VALS"] [i]
        bytes.extend(self.__get_write_sequence(addr, val, num_bytes))
        addr += 1
    except ValueError as e:
      err_msg = "Unable to write 0x%08x (%d) to %d byte register %s (0x%04x): %s" % (
        val, val, num_bytes, name, addr, e)
      raise ValueError(err_msg)

    out_file.write(bytearray(bytes))

  def __write_mem_region_seq(self, out_file, region, mem_structs):
    """
    Write out a memory region to an i2c sequence file.

    Parameters:
    out_file - the file to write to
    region - the memory region to write
    mem_structs - dictionary containing information about each memory structure
    """
    bytes = []
    base_addr = int(self.__cfg["GROUP_REMAP_ID"], 16) << 12
    next_addr = region["start_addr"]
    size = region["size"]
    # Get data array associated with this region
    data = []
    for struct in region["structs"]:
      for b in mem_structs[struct.value]["bytes"]:
        data.extend(b)
    # Write out data, updating the group remap base address as needed
    for i in range(0, len(data)):
      # Update GROUP_REMAP base address 
      if (i == 0) or ((next_addr & 0xFFF) == 0):
        reg_val = (next_addr & 0xFFFFF000) >> 8
        try:
          bytes.extend(self.__get_write_sequence(self.__cfg["REG_GROUP_REMAP"], reg_val, 3))
        except ValueError as e:
          err_msg = "Unable to write 0x%08x (%d) to GROUP_REMAP register (0x%04x): %s" % (
              reg_val, reg_val, self.__cfg["REG_GROUP_REMAP"], e)
          raise ValueError(err_msg)
      try:
        bytes.extend(self.__get_write_sequence(base_addr + (next_addr & 0xFFF), data[i], 1))
      except ValueError as e:
        err_msg = "Unable to write 0x%08x (%d) to address 0x%08x: %s" % (
          val, val, next_addr, e)
        raise ValueError(err_msg)      
      next_addr += 1
    out_file.write(bytearray(bytes))
    
  def __get_write_sequence(self, addr, val, num_bytes):
    """
    Convert a register write to an i2c sequence.

    Parameters:
    addr - the 16-bit address of the register
    val - the value of the register
    num_bytes - number of bytes in the register

    Returns:
    list of bytes in the i2c sequence
    
    Raises:
    ValueError if address or value are out of bounds
    """
    bytes = []
    if addr < 0 or addr >= 2**16:
      raise ValueError("Address out of range!")
    if val < 0 or val >= 2**(8*num_bytes):
      raise ValueError("Value out of range!")
    curr_addr = addr
    val_bytes = val_to_uint8_array(val, num_bytes, self.__endianness)
    for i in range(0, num_bytes):
      addr_bytes = val_to_uint8_array(curr_addr, 2, self.__endianness)
      bytes.extend([addr_bytes[0], addr_bytes[1], val_bytes[i]])
      curr_addr += 1
    return bytes


  def __uint8_array_to_string(self, bytes):
    """
    Convert byte array to string of comma separated hex values

    Parameters:
    bytes - uint8 array

    Returns:
    string representation
    """
    hex_strings = ["{0:#0{1}x}".format(x,4) for x in bytes]
    return ", ".join(hex_strings)

  def __indent(self, level):
    """
    Return string containing the proper number of spaces to indent a certain
    number of levels

    Paramters:
    level - the indentation level

    Returns:
    indentation string
    """
    return " " * level * SPACES_PER_INDENT

  def __get_quant_params_name(self, dnn_name):
    """
    Get name of the quantization parameters structure

    Parameters:
    dnn_name - name of this DNN instance

    Returns:
    name of quantization parameters structure
    """
    return dnn_name + "_quant_params"

  def __get_static_data_name(self, dnn_name):
    """
    Get name of the static data structure

    Parameters:
    dnn_name - name of this DNN instance

    Returns:
    name of static data structure
    """
    return dnn_name + "_static_data"

  def __get_misc_data_name(self, dnn_name):
    """
    Get name of the misc data structure

    Parameters:
    dnn_name - name of this DNN instance

    Returns:
    name of misc data structure
    """
    return dnn_name + "_misc_data"

  def __get_optable_name(self, dnn_name):
    """
    Get name of the dnn_optable structure

    Parameters:
    dnn_name - name of this DNN instance

    Returns:
    name of dnn_optable structure
    """
    return dnn_name + "_optable"

  def __write_memory_structure(self, out_file, region, mem_structs):
    """
    Write the binary contents of a memory structure out to a file

    Parameters:
    out_file - the file to write
    region - the memory region to write
    mem_structs - list of memory structures in the DNN
    
    Raises:
    OSError if file cannot be created
    """
    bytes = []
    for struct in region["structs"]:
      data = mem_structs[struct.value]["bytes"]
      for d in data:
        bytes.extend(d)
    out_file.write(bytearray(bytes))

  def __get_txt_buffer_header(self):
    """
    Get header to print to an output .txt. file that describes fields that are
    printed for each input/output buffer
    """
    return "    %-8s%-22s%-8s%-14s%-10s%-12s%-12s%-12s%-12s%-8s%-8s%-14s%-8s%-8s" % (
      "ID", "Dimensions", "Type", "Location", "Address", "Data Start", "Data Size", "Row Stride", "Bat Stride", "Parent", "# Con", "Quant Type", "QS[0]", "QZ[0]")

  def __buffer_to_string(self, buf, idx, dnn, mem_regions, mem_structs):
    """
    Convert a buffer dictionary from the reader's output to a string that
    can be written to a .txt file

    Parameters:
    buf - the buffer dictionary
    idx - the buffer's index'
    dnn - DNN dictionary
    mem_structs - memory structure dictionary
    mem_regions - memory regions dictionary
    """
    q_scale = dnn["quant_params"][buf["quant_start_idx"]][0]
    q_zero = dnn["quant_params"][buf["quant_start_idx"]][1]
    if (buf["buffer_type"] == BufferType.STATIC_DATA):
      addr = mem_structs[MemoryStructure.STATIC_DATA.value]["start_addr"]
    elif self.__cfg["REVERSE_ALLOC_SCRATCH_RAM"]:
      addr = mem_regions[MemoryRegion.DNN_RAM.value]["start_addr"]
    else:
      addr = mem_structs[MemoryStructure.SCRATCH_RAM.value]["start_addr"]

    addr += buf["data_start_offset"]
    if buf["data_signed"]:
      data_type = "S " + buf["data_type"].name
    else:
      data_type = "U " + buf["data_type"].name

    return "    %-8s%-22s%-8s%-14s%-10s%-12s%-12s%-12s%-12s%-8s%-8s%-14s%.5f %-8s" % (
          idx,
          " x ".join(str(x) for x in buf["dimensions"]),
          data_type,
          buf["buffer_type"].name,
          "{0:#0{1}x}".format(addr, 6),
          "(" + str(buf["data_start_row"]) + ", " + str(buf["data_start_col"]) + ")",
          str(buf["data_num_rows"]) + " x " + str(buf["data_num_cols"]),
          str(buf["row_size"]), str(buf["batch_size"]),
          buf["parent"],
          buf["num_connections"],
          buf["quant_type"].name,
          q_scale,
          q_zero)


  def __buffer_to_io_string(self, op_idx, buf):
    """
    Convert a buffer dictionary from the reader's output to a string that
    can be written to a inputs.txt or outputs.txt file

    Parameters:
    op_idx - this operation's index in the DNN
    buf - the buffer dictionary
    """
    vals = []
    vals.append(str(op_idx))
    vals.append("0x%05x" % (buf["data_start_offset"]))
    vals.append(str(buf["dimensions"][DIMS_NUM_BATCHES]))
    vals.append(str(buf["data_num_rows"]))
    vals.append(str(buf["data_num_cols"]))
    vals.append(str(buf["dimensions"][DIMS_NUM_CHANNELS]))
    vals.append(str(buf["row_size"]))
    vals.append(str(buf["batch_size"]))
    return " ".join(vals)
