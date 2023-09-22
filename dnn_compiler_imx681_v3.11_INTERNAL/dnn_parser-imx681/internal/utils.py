# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import numpy as np
import logging

from internal.constants import *

def val_to_uint8_array(val, num_bytes, endianness):
  """
  Convert a value to an array of uint8's 

  Parameters:
  val - the value
  num_bytes - number of bytes to use to represent the value (e.g. 4 for a uint32)
  endianness - "big" for MSB first, "little" for LSB first

  Returns
  array of <num_bytes> uint8 values
  """
  bytes = []
  shift = (num_bytes-1)*8
  while shift >= 0:
    if endianness == Endianness.BIG:
      bytes.append(np.uint8(np.uint32(val) >> shift)) # MSB first
    else:
      bytes.insert(0, np.uint8(np.uint32(val) >> shift)) # LSB first
    shift -= 8
  return bytes

def uint8_array_to_val(bytes, endianness):
  """
  Convert an array of uint8's to bytes

  Parameters:
  bytes - byte_array
  endianness - "big" for MSB first, "little" for LSB first

  Returns
  value
  """
  val = 0
  for idx in range(0, len(bytes)):
    if endianness == Endianness.BIG:
      val |= (bytes[idx] << (8*(len(bytes)-idx-1)))
    else:
      val |= (bytes[idx] << (8*idx))
  return val

def float_to_fp_uint32(x, q_format):
  """
  Converts a float value to a uint32 in fixed point format

  Parameters:
  x - the input float value
  q_format - number of fractional bits in fixed point result

  Returns:
  fixed point result
  """

  # Right shift the float value by <q_format> bits
  if (x == float("inf")):
    raise RuntimeError("Weight with value of infinity found in model!")
  else:
    temp = round(x * 2**q_format)
    # max_val = np.iinfo(np.int32).max
    # min_val = np.iinfo(np.int32).min
    # if temp > max_val:
    #   logging.warning("Clipping fixed point value from %d to %d" % (temp, max_val))
    #   temp = max_val
    # if temp < min_val:
    #   logging.warning("Clipping fixed point value from %d to %d" % (temp, min_val))
    #   temp = min_val
    return np.uint32(temp)


def align_value(val, alignment, round_up=True):
  """
  Align a integer to be an even multiple of another integer

  Parameters:
  val - the value to align
  alignment - the multiple to align to
  round_up - if True, output will be >= val.
             if False, output will be <= val.
  
  Returns:
  aligned value
  """
  if round_up:
    return int(alignment*np.ceil(float(val)/alignment))
  else:
    return int(alignment*np.floor(float(val)/alignment))

def align_byte_array(data, alignment):
  """
  Align the length of a list of byte values to be an even multiple of another 
  integer by adding padding

  Parameters:
  data - the data array
  alignment - the multiple to align to
  
  Returns:
  the aligned data
  """
  while (len(data) % alignment) != 0:
    data.append(0);
  return data

def get_buffer_size(buf):
  """
  Get the total buffer size, in bytes, based on its dimensions and type

  Parameter:
  buf - the buffer dictionary

  Returns:
  total size, in bytes
  """
  return np.product(buf["dimensions"]) << buf["bpp_shift"]

def buffer_has_stride(buf):
  """
  Return true if this buffer has has stride between rows or columns (e.g.
  buffer dims != data size).

  This test will also detect padding as buffer dims includes the padding.
  
  Parameters:
  buf - the buffer dictionary

  Returns:
  True if there is stride
  """
  return (buf["dimensions"][DIMS_NUM_ROWS] != buf["data_num_rows"]) or (
          buf["dimensions"][DIMS_NUM_COLS] != buf["data_num_cols"])

def buffer_remove_stride(buf):
  """
  Remove any stride between rows and/or columns from a buffer's dictionary.

  This also removes padding.

  Parameters:
  buf - the buffer dictionary (modified by this function)
  """
  buf["data_start_row"] = 0
  buf["data_start_col"] = 0
  buf["dimensions"][DIMS_NUM_ROWS] = buf["data_num_rows"]
  buf["dimensions"][DIMS_NUM_COLS] = buf["data_num_cols"]

def add_to_misc_data(misc_data, bytes):
  """
  Add data to the misc data array, or get a pointer to existing data in
  the array if identical bytes are already in it

  Parameters:
  misc_data - the misc data array
  bytes - the bytes to add

  Returns:
  index of the bytes, either where they were added or where the existing data
  was found
  """
  # Search misc data for an existing string of bytes that matches the input
  length = len(bytes)
  for i in range(0, len(misc_data) - length + 1):
    if misc_data[i:i+length] == bytes:
      return i
  # If it wasn't found add it now
  idx = len(misc_data)
  misc_data.extend(bytes)
  # Align the end of misc data to a 4 byte boundary
  while len(misc_data) % 4 != 0:
    misc_data.append(np.uint8(0))
  return idx

def record_instruction(instr, cnt, cycles, cnt_dict, cycles_dict):
  """
  Record a VPU instruction call in instruction count/cycles dictionary and
  determine the total number of cycles

  Parameters:
  instr - the instruction name
  cnt - the number of times this instruction is called
  cycles - the number of cycles per call
  cnt_dict - the instruction count dictionary to add to
  cycle_dict - the instruction cycles dictionary to add to

  Return:
  total cycles consumed by calling this instruction <cnt> times
  """
  total_cycles = cnt*cycles

  if instr in cnt_dict:
    cnt_dict[instr] += cnt
  else:
    cnt_dict[instr] = cnt

  if instr in cycles_dict:
    cycles_dict[instr] += total_cycles
  else:
    cycles_dict[instr] = total_cycles

  return total_cycles

def cycles_to_ms(cycles):
  """
  Convert a number of VPU cycles to an amount of milliseconds

  Parameters:
  cycles - number of cycles

  Returns:
  number of milliseconds (as a float)
  """
  return 1000*cycles*VPU_CLOCK_PERIOD_S
  
def flatbuf_enum_to_str(val, class_type):
  """
  Helper function that converts a value to a string for an enum from flatbuffers

  Parameters:
  val - the value (integer)
  class_type - the enum class type

  Returns:
  string representation
  """
  for name, value in class_type.__dict__.items():
    if value == val:
       return name
  return "Unknown"

def check_param_range(name, val, min_val, max_val):
  """
  Check if a parameter value is in range. If it is not, raise a RuntimeError

  Parameters:
  name - the name of the parameter (for error message)
  val - the value of the parameter
  min_val - the minimum supported value
  max_val - the maximum supported value

  Raises:
  RuntimeError if value is out of range
  """
  if (val > max_val) or (val < min_val):
    raise RuntimeError("%s parameter out of range: %d. Supported range is [%d, %d]" % (
      name, val, min_val, max_val))

def check_param_valid(name, val, valid_list):
  """
  Check if a parameter value is valid from a list of valid values. 
  If it is not, raise a RuntimeError.

  Parameters:
  name - the name of the parameter (for error message)
  val - the value of the parameter
  valid_list - list of valid values

  Raises:
  RuntimeError if value is out of range
  """
  if not (val in valid_list):
    raise RuntimeError("%s parameter is not valid: %d. Supported values are [%s]" % (
      name, val, ",".join(str(valid_list))))
