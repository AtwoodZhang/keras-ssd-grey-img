# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import os
import logging

from internal.constants import *
from internal.utils import *

# Mapping of each required configuration file parameter to a type
#  If parameter type is a list, a element type is given
#  If parameter type is a string, a bool is given to indicate whether or not this
#    is a relative path
REQUIRED_PARAMETERS = {
  "DATA_DIRECTORY"                  : [str, True],
  "SENSOR_VERSION"                  : [SensorVersion],
  "DNN_MODEL"                       : [str, True],
  "DNN_NAME"                        : [str, False],
  "DNN_RAM_MAX_SIZE"                : [int],
  "DNN_RAM_START_ADDR"              : [int],
  "DNN_ROM_MAX_SIZE"                : [int],
  "DNN_ROM_START_ADDR"              : [int],
  "GROUP_REMAP_ID"                  : [str, False],
  "I2C_INIT_SEQ"                    : [list, int],
  "MEMORY_ORDER"                    : [MemoryOrder],
  "ML_CONV_MAX_OUT_SIZE"            : [int],
  "ML_CONV_NUM_PARTITIONS"          : [int],
  "NUM_DNN_POSTPROC_THRESHOLD_REGS" : [int],
  "OUTPUT_DIRECTORY"                : [str, True],
  "OUTPUT_ENDIANNESS"               : [Endianness],
  "OUTPUT_MODE"                     : [OutputMode],
  "POSTPROCESSING"                  : [str, False],
  "POSTPROC_DNN_DATA_NUM_COLS"      : [int],
  "POSTPROC_DNN_DATA_NUM_ROWS"      : [int],
  "POSTPROC_DNN_DATA_OUT_IDX"       : [int],
  "POSTPROC_DNN_DATA_START_COL"     : [int],
  "POSTPROC_DNN_DATA_START_ROW"     : [int],
  "POSTPROC_COL_TO_OBJECT_TYPE"     : [list, int],
  "POSTPROC_COMPARE_VALS_NUM_COLS"  : [int],
  "POSTPROC_COMPARE_VALS_NUM_ROWS"  : [int],
  "POSTPROC_COMPARE_VALS_OUT_IDX"   : [int],
  "POSTPROC_COMPARE_VALS_START_COL" : [int],
  "POSTPROC_COMPARE_VALS_START_ROW" : [int],
  "POSTPROC_THRESHOLD_VALS"         : [list, int],
  "REG_DNN_WEIGHT_ADDR"             : [int],
  "REG_DNN_OPTABLE_ADDR"            : [int],
  "REG_DNN_QUANT_PARAM_ADDR"        : [int],
  "REG_DNN_MISC_DATA_ADDR"          : [int],
  "REG_DNN_POSTPROC_THRESHOLD0"     : [int],
  "REG_GROUP_REMAP"                 : [int],
  "REVERSE_ALLOC_SCRATCH_RAM"       : [bool],
  "ROI_POOL_MODE"                   : [ROIPoolMode],
  "SYSTEM_RAM_MAX_SIZE"             : [int],
  "SYSTEM_RAM_START_ADDR"           : [int],
  "SYSTEM_ROM_MAX_SIZE"             : [int],
  "TEMPLATE_DIRECTORY"              : [str, True],
}

# Mapping of each optional configuration file parameter to a type
#  See REQUIRED_PARAMETERS above for more details on format
OPTIONAL_PARAMETERS = {
  "INPUT_SCALE"                     : [float],
  "INPUT_ZEROPOINT"                 : [int],
  "FORCE_PATCHES"                   : [list, str],
  "OPTIMIZATION_1X1_BEST_EFFORT"    : [bool],
  "TEST_USE_DNN_RAM"                : [int],
}

# All valid configuration parameters
# This syntax creates a single dictionary that includes all parameters from
# REQUIRED_PARAMETERS and all parameters from OPTIONAL_PARAMETERS
SUPPORTED_PARAMETERS = {**REQUIRED_PARAMETERS, **OPTIONAL_PARAMETERS}

class ConfigReader():
  """
  This class handles reading a configuration file, checking that its format is
  valid, and creating a dictionary containing all parameters and values.
  """

  # ----------------------------------------------------------------------------
  # Public functions
  # ----------------------------------------------------------------------------
  def __init__(self, in_file):
    """
    Constructor

    Parameters:
    in_file - .cfg file path
    """
    self.__in_file = in_file
    self.__in_dir = os.path.split(in_file)[0]
    self.__in_filename = os.path.split(in_file)[1]
    self.__cfg_dict = dict()

  def load_dict(self):
    """
    Read the contents of the configuration file and loads it into a python dictionary

    Raises:
    IOError if file cannot be opened
    ValueError if a line in the file is invalid
    """
    lines = self.__read(self.__in_file)
    self.__build_cfg_dict(lines)
    self.__cfg_dict["INPUT_DIRECTORY"] = self.__in_dir

  def load_overrides(self, overrides):
    """
    Parse command-line overrides for configuration file and update configuration
    dictionary

    Parameters:
    overrides - list of strings in format key=val to override configuration file

    Raises:
    ValueError if an override is invalid
    """
    for override in overrides:
      keyval = override.split("=")
      if len(keyval) != 2:
        raise ValueError(
          "Invalid configuration file override: %s. Must be formatted key=val" % (
          override))
      try:
        self.__add_to_dict(keyval[0], keyval[1])
      except ValueError as err:
        raise ValueError(
          "Invalid configuration file override: %s. %s" % (override, err))

  def check_for_completeness(self):
    """
    Check that this configuration has all required parameters in it

    Raises:
    ValueError if a required parameter is missing
    """
    for key in REQUIRED_PARAMETERS:
      if not key in self.__cfg_dict:
        raise ValueError("Configuration file is missing a required parameter: %s" % key)

  def get_cfg_dict(self):
    """
    Get the configuration dictionary
    """
    return self.__cfg_dict


  # ----------------------------------------------------------------------------
  # Private functions
  # ----------------------------------------------------------------------------
  def __read(self, in_file):
    """
    Read the contents of the configuration file into a list of lines

    Parameters:
    in_file - the input file path

    Returns:
    lines - list of raw lines in the file

    Raises:
    IOError if file cannot be opened
    """
    path = in_file
    with open(path, "r") as f:
      lines = f.readlines()
      return lines

  def __build_cfg_dict(self, lines):
    """
    Parse the lines from a configuration file and build dictionary of all key/
    value pairs

    Parameters:
    lines - list of lines in the file

    Raises:
    ValueError if a line is invalid
    """
    line_num = 0
    for line in lines:
      line_num += 1
      # Remove new line character from end of line
      line = line.rstrip('\n')
      # Ignore comments (starting with #) and blank lines
      if (line != '') and (not line.startswith("#")):
        # Extract a key and value pair
        tokens = line.split()
        if len(tokens) < 2:
          raise ValueError(
            "Error parsing line %d of configuration file: Line must have format \"KEY VAL\": %s" % (
            line_num, line))
        key = tokens[0]
        val = " ".join(tokens[1:])
        if key == "INC_CONFIG":
          # Parse the included file and add all of its values to the dictionary
          inc_file = os.path.normpath(os.path.join(self.__in_dir, val))
          logging.info("Including base configuration: %s" % inc_file)
          inc_config = ConfigReader(inc_file)
          inc_config.load_dict()
          self.__cfg_dict.update(inc_config.get_cfg_dict())
        else:
          try:
            self.__add_to_dict(key, val)
          except ValueError as err:
            raise ValueError(
              "Error parsing line %d of %s: %s" % (line_num, self.__in_filename, err))

  def __add_to_dict(self, key, val):
    """
    Attempt to add a key and value pair to the configuration dictionary, checking
    for invalid format

    Raises:
    ValueError if a key or value is invalid
    """
    # Check that the key is valid
    if key not in SUPPORTED_PARAMETERS:
      raise ValueError("Unrecognized parameter (%s)" % key)

    # Check that the value is the correct format and cast to the correct type
    param_data = SUPPORTED_PARAMETERS[key]
    param_type = param_data[0]
    try:
      if param_type == str:
        if (param_data[1] == True) and (("/" in val) or ("\\" in val)):
          val = os.path.normpath(os.path.join(self.__in_dir, val))
      elif param_type == int:
        val = self.__val_to_int(val)
      elif param_type == bool:
        val = self.__val_to_bool(val)
      elif param_type == float:
        val = self.__val_to_float(val)
      elif param_type == list:
        val = self.__val_to_list(val, param_data[1])
      else:
        val = self.__val_to_enum(val, param_type)
    except ValueError as err:
      raise ValueError("Invalid value for %s: %s. %s" % (
        key, val, err))

    # Add to the dictionary, overriding any previous values
    if key in self.__cfg_dict:
      logging.info("Overriding configuration parameter %s: %s -> %s" % (
        key, str(self.__cfg_dict[key]), str(val)))
    self.__cfg_dict[key] = val

  def __val_to_list(self, val, elem_type):
    """
    Convert a value from a string to a list of elem_type

    Parameters:
    val - the value
    elem_type - the type of each value in the list

    Returns:
    val as a list

    Raises:
    ValueError if the string is not a valid list, or a value in it is not elem_type.
    """
    if (not val.startswith("[")) or (not val.endswith("]")):
      raise ValueError("Expecting list of values with format [a, b, c]")
    # Convert string of format "[a, b, c]" to a list of strings
    val = val.replace("[","").replace("]","").replace(" ","").split(",")
    # Remove empty strings
    while "" in val:
      val.remove("")
    # Convert to int if required
    if elem_type == int:
      for i in range(0, len(val)):
        try:
          val[i] = int(val[i], 0)
        except ValueError:
          raise ValueError("Invalid integer value found in list: %s" % (val[i]))
    return val

  def __val_to_int(self, val):
    """
    Convert a value from a string (hex or dec) to an integer

    Parameters:
    val - the value

    Returns:
    val as a int

    Raises:
    ValueError if the string is not a valid integer
    """
    try:
      val = int(val, 0)
    except ValueError:
      raise ValueError("Value must be an integer")
    return val

  def __val_to_bool(self, val):
    """
    Convert a value from a string to a boolean

    Parameters:
    val - the value

    Returns:
    val as a bool

    Raises:
    ValueError if the string is not a valid boolean (True/False or 1/0)
    """
    if val.lower() in ["true", "1"]:
      val = True
    elif val.lower() in ["false", "0"]:
      val = False
    else:
      raise ValueError("Value must be a boolean string [True, False, 1, 0]")
    return val


  def __val_to_float(self, val):
    """
    Convert a value from a string (dec) to a float

    Parameters:
    val - the value

    Returns:
    val as a float

    Raises:
    ValueError if the string is not a valid float
    """
    try:
      val = float(val)
    except ValueError:
      raise ValueError("Value must be an floating point number")
    return val


  def __val_to_enum(self, val, enum_type):
    """
    Convert a value from a string to enum_type

    Parameters:
    val - the value
    enum_type - the enum type to convert to

    Returns:
    val as enum

    Raises:
    ValueError if the string is not a valid value for the specified enum type
    """
    try:
      val = enum_type[val.upper()]
    except KeyError:
      valid_vals = [v.name.lower() for v in enum_type]
      raise ValueError("Valid values are [%s]" % ", ".join(valid_vals))
    return val
