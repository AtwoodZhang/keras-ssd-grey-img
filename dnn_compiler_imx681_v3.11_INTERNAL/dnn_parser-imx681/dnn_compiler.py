# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import argparse
import logging
import os
import sys
import copy
from enum import Enum

from internal.utils import *
from internal.config_reader import ConfigReader
from internal.dnn_reader.tflite_reader import TFLiteReader
from internal.dnn_reader.pytorch_reader import PyTorchReader
from internal.dnn_processing.dnn_finalizer import DNNFinalizer
from internal.dnn_processing.conv_optimizer import ConvOptimizer
from internal.dnn_processing.buffer_allocator import BufferAllocator
from internal.dnn_processing.memory_manager import MemoryManager
from internal.output_writer.output_writer import OutputWriter
from internal.operations.dnn_operation import DNNOperation

from internal.operations.dnn_operation import DNNOperationConv2D


from internal.constants import DNN_COMPILER_VERSION, DNN_COMPILER_TARGET_PLATFORM, DNN_COMPILER_INTERNAL_USE


# ------------------------------------------------------------------------------
# Module Globals
# ------------------------------------------------------------------------------

# Estimate of DNN_RAM space available for scratch buffers.
# The amount of DNN_RAM that is available for scratch buffers is not
# fully known until after generation and may change as a result of
# generation.
dnn_ram_available_for_scratch = 0

# Number of 1x1 optimizations to skip per processing loop.
# Conv2D can use an `OPTIMIZED_1X1` to trade RAM for speed. Unfortunately 
# when this decision is made it doesn't have full info. When the system 
# has full info it's too late to update the decision. 
# When the DNN doesn't fit we can use this to force some optimizations to
# be skipped. We skip the early potential optimizations because the 1X1
# optimization save more time on smaller buffers and buffer size generally
# decreases with later operations.
dnn_skip_1x1_optimization_count = 0

# Maximum number of 1x1 optimizations that could be made.  
# This is used to search for a result faster than disabling 1 at a time.
dnn_skip_1x1_optimization_max = 0

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

# List of valid values of log level
VALID_LOG_LEVELS = ["debug", "info", "warning", "error", "critical"]

# String listing all valid log levels
VALID_LOG_LEVELS_STRING = "[%s]" % ", ".join(VALID_LOG_LEVELS)


class ProcessModelState(Enum):
  """
  State machine states for looping calls to __process_dnn_model.

  Because of the way that the data structures are intertwined certain kinds
  of failures cause us to restart processing for scratch. There is a specific
  problem where available scratch ram in DNN_RAM is calculated before we
  know for sure that nothing has overflowed from SYS_RAM into DNN_RAM.  If
  something has overflowed the available scratch RAM was calculated wrong.
  This can cause the scratch RAM buffer buffer generation to make descisions
  
  In this case the only option is revert and re-process the DNN. 
  __process_dnn_model uses ProcessModelState across multiple calls to build
  increasingly accurate estimate of usage and therefore available space.

  The generation can succeed if the model fits even when teh estimate is
  incorrect.
  """
  # Initial state, this is the first call to __process_dnn_model. 
  # __process_dnn_model will assume nothing has overflowed into DNN ram.
  # This will transition to:
  #     SUCCESS - if the model fits
  #     FAILURE - if the model doesn't fit and there are no smaller 
  #               options to try.
  #     SECOND  - if the model doesn't fit and there are memory savings
  #               that can be tried.
  INITIAL = 0

  # Second pass, try with no optimizations to see if there is any
  # combination of optimizations so the model fits in RAM
  # This will transition to:
  #     SUCCESS - if the model fits and the configuration is set to accept
  #               success at this stage.
  #     FAILURE - if the model doesn't fit
  #     RETRY   - if the model does fit and configuration allows a search
  #               for a more optimal solution.
  SECOND  = 1

  # Buffer allocation failed but the estimate can be improved, reset the
  # dnn object and call __process_dnn_model again with this state.
  # This will transition to:
  #     SUCCESS - if the model fits
  #     FAILURE - if the model doesn't fit and the estimate is correct
  #     RETRY   - if the model doesn't fit and the estimate is smaller
  #               than actual use.
  RETRY   = 2

  # Buffer allocation has succeeded
  SUCCESS = 3

  # Buffer allocation has failed and estimate can not revised further. This
  # failure is permanent and can't be fixed in this processes. 
  FAILURE = 4

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
def run(config_filename, model=None, log_level="info", config_overrides=[]):
  """
  Compile a DNN model from Tensorflow or PyTorch and create output structures that
  are compatible with Sony image sensors (IMX681 or similar)

  Parameters:
  config_filename - relative path and filename of a DNN compiler configuration file
  model (optional) - For PyTorch models only: the quantized, trained model class
  log_level (optional) - Minimum logging level to print to console (see VALID_LOG_LEVELS)
  config_overrides (optional) - list of "KEY=VAL" strings to override configuration file settings

  Raises:
  RuntimeError on failure
  """

  __init_logging(log_level)

  cfg_dict = __load_configuration(config_filename, config_overrides)

  print()
  print("====================================================")
  print("Sony %s %s DNN Compiler version %s" % (DNN_COMPILER_TARGET_PLATFORM, cfg_dict["SENSOR_VERSION"].value, DNN_COMPILER_VERSION))
  print("====================================================")
  print()

  print("--- Reading DNN model...")
  dnn = __read_dnn_model(cfg_dict, model)
  print()

  print ("--- Processing DNN model...")
  # mem_regions, mem_structs, mem_report = __process_dnn_model(dnn, cfg_dict)
  process_state = ProcessModelState.INITIAL
  dnn_orig = dnn

  processing_passes = 0
  # condition test and exit is at end of loop.
  while True:
    processing_passes += 1
    if processing_passes > 1:
      print("  Pass %d:" % processing_passes)

    dnn = copy.deepcopy(dnn_orig)
    mem_regions, mem_structs, mem_report, process_state = __process_dnn_model(dnn, cfg_dict, process_state)

    if process_state not in [ProcessModelState.SECOND, ProcessModelState.RETRY]:
      break
  
  if process_state != ProcessModelState.SUCCESS:
      raise RuntimeError("DNN is too large to fit in image sensor memory!")

  print()

  print ("--- Writing output files...")
  writer = OutputWriter(cfg_dict)
  if DNN_COMPILER_INTERNAL_USE:
    inc_instructions = True
  else:
    inc_instructions = False
  try:
    writer.write(cfg_dict["DNN_NAME"], dnn, mem_regions, mem_structs, mem_report, inc_instructions)
  except RuntimeError as e:
    logging.error(e)
    raise RuntimeError("Failed to write output file!")
  except ValueError as e:
    logging.error(e)
    raise RuntimeError("Invalid configuration parameters for writing output file!")
  print("Done. Output files written to .\\%s" % cfg_dict["OUTPUT_DIRECTORY"])

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
  """
  Main function that runs the DNN compiler using command line arguments (only
  supports Tensorflow models, not PyTorch)
  """

  # Parser for command line arguments
  parser = argparse.ArgumentParser(
    description='Convert a saved Tensorflow DNN model (.tflite) to formats that ' +
    'are compatible with Sony image sensors (IMX681 or similar).',
   formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--log', default="info",
                      help='minimum log level to display ' + VALID_LOG_LEVELS_STRING)
  parser.add_argument("config_file",
    help="Path to configuration file (.cfg)")
  parser.add_argument('config_overrides', nargs='*',
                      help='overrides for configuration file parameters (KEY=VAL)')

  args = parser.parse_args()

  try:
    run(args.config_file, log_level=args.log, config_overrides=args.config_overrides)
  except RuntimeError as e:
    logging.critical(e)
    return -1

  return 0
# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def __init_logging(level):
  """
  Initialize logging at the specified level

  Parameters:
  level - string indicating log level (one of VALID_LOG_LEVELS)
  """
  msg = ""

  if not level.lower() in VALID_LOG_LEVELS:
    # Create a warning message, but don't print it until after logging is
    # initialized
    msg = "Invalid log level: %s (defaulting to info). Use --help for usage information." % level
    level = "info"

  numeric_level = getattr(logging, level.upper(), None)
  logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

  if msg:
    logging.warning(msg)

def __load_configuration(config_file, config_overrides):
  """
  Load configuration dictionary from file and command line overrides

  Parameters:
  config_file - path to configuration file
  config_overrides - list of command line overrides
  """
  logging.info("Reading configuration file: %s" % config_file)
  cfg = ConfigReader(config_file)
  try:
    cfg.load_dict()
  except IOError as e:
    logging.error(e)
    raise RuntimeError("Failed to read configuration file!")
  except ValueError as e:
    logging.error(e)
    raise RuntimeError("Failed to parse configuration file!")

  try:
    cfg.load_overrides(config_overrides)
  except ValueError as e:
    logging.error(e)
    raise RuntimeError("Invalid command line arguments. Use --help for usage information.")

  try:
    cfg.check_for_completeness()
  except ValueError as e:
    logging.error(e)
    raise RuntimeError("Configuration file is incomplete!")

  return cfg.get_cfg_dict()

def __read_dnn_model(cfg_dict, model):
  """
  Read in a DNN from a .tflite or pytorch file and create a DNN dictionary

  Parameters:
  cfg_dict - the configuration dictionary

  Returns:
  dnn_dict - dictionary describing the DNN as it was read from the file
  """
  if cfg_dict["DNN_MODEL"] == "pytorch":
    if not model:
      msg = "DNN_MODEL is set to \"pytorch\" in the configuration file, but no PyTorch model was provided!"
      msg += " To compile PyTorch model, call dnn_compiler() function and pass in the PyTorch model."
      raise RuntimeError(msg)
    logging.info("Loading data from PyTorch model")
    reader = PyTorchReader(model, cfg_dict)
  else:
    file = cfg_dict["DNN_MODEL"]
    if model:
      logging.warning("Ignoring DNN model that was passed into dnn_compiler(). In Tensorflow mode, DNN model " + 
        "is read from file instead.")
    logging.info("Loading data from saved Tensorflow model: %s" % file)
    reader = TFLiteReader(file, cfg_dict)

  try:
    dnn_dict = reader.load_dict()
    dnn_dict["cfg"] = cfg_dict    # record cfg_dict for use in building SPU Param Tables
  except IOError as e:
    logging.error(e)
    raise RuntimeError("Failed to read DNN model file!")
  except RuntimeError as e:
    logging.error(e)
    raise RuntimeError("Failed to load DNN model!")

  return dnn_dict


def __process_dnn_model(dnn, cfg_dict, process_state):
  """
  Process a DNN and modify it to be compatible with IMX firmware

  Parameters:
  dnn - the dnn dictionary to modify
  cfg_dict - the configuration dictionary
  cfg_dict - the configuration dictionary
  process_state - State of processing for op

  Returns:
  mem_regions - assembled memory regions
  mem_structs - memory structs that have been built.
  mem_report  - text report of memory usage
  process_state - results of pass and potential indication to call again 
                  for better optimization results.
  """
  global dnn_skip_1x1_optimization_count
  global dnn_skip_1x1_optimization_max
  global dnn_ram_available_for_scratch

  finalizer = DNNFinalizer()
  allocator = BufferAllocator()
  optimizer = ConvOptimizer()
  mem_manager = MemoryManager()

  best_effort = ( "OPTIMIZATION_1X1_BEST_EFFORT" in cfg_dict
                  and cfg_dict["OPTIMIZATION_1X1_BEST_EFFORT"] )

  # On the second pass we try with no optimizations to see if it's
  # possible to make this model fit at all
  optimizations_to_skip = dnn_skip_1x1_optimization_count
  if process_state == ProcessModelState.SECOND:
    optimizations_to_skip = dnn_skip_1x1_optimization_max

  enabled_1x1_optimizations = DNNOperationConv2D.pre_check_opt1x1_criteria(dnn, optimizations_to_skip)
  if process_state == ProcessModelState.INITIAL:
    dnn_skip_1x1_optimization_max = enabled_1x1_optimizations

  logging.info("Adding post-processing layer to DNN")
  try:
    finalizer.add_postprocessing(dnn, cfg_dict)
  except ValueError as e:
    logging.error(e)
    raise RuntimeError("Failed to add post-processing layer to DNN!")

  logging.info("Finalizing parameters for %d layers" % len(dnn["operations"]))
  finalizer.pad_buffers(dnn)
  try:
    finalizer.finalize_op_params(dnn, cfg_dict["OUTPUT_ENDIANNESS"], cfg_dict["MEMORY_ORDER"])
  except RuntimeError as e:
    logging.error(e)
    raise RuntimeError("Unsupported layer parameters found!")
  finalizer.split_operations(dnn, cfg_dict)

  if (cfg_dict["ML_CONV_MAX_OUT_SIZE"] > 0):
    logging.info("Attempting to create multilayer convolutions with %d partitions" % (
      cfg_dict["ML_CONV_NUM_PARTITIONS"]))
    try:
      optimizer.split_multilayer_convolutions(dnn, cfg_dict)
    except ValueError as e:
      logging.error(e)
      raise RuntimeError("Failed to create multi-layer convolutions!")

  max_scratch_ram_size = mem_manager.get_max_scratch_ram_size(dnn, cfg_dict)
  if process_state != ProcessModelState.INITIAL:
    max_scratch_ram_size = dnn_ram_available_for_scratch

  logging.info("Allocating memory for scratch buffers")
  try:
    allocator.allocate_scratch_buffers(dnn, cfg_dict, max_scratch_ram_size)
  except RuntimeError as e:
    logging.error(e)
    raise RuntimeError("Failed to allocate buffers!")

  finalizer.finalize_buffer_descriptors(dnn, cfg_dict)

  # Get list of firmware patches needed to run this DNN
  if "FORCE_PATCHES" in cfg_dict:
    logging.info("Forcing compatibility with patches: %s" % cfg_dict["FORCE_PATCHES"])
    patches = cfg_dict["FORCE_PATCHES"]
  else:
    patches = []

  for i in range(0, len(dnn["operations"])):
    op = dnn["operations"][i]
    patch_list = DNNOperation.check_for_patches(dnn, op)
    if patch_list:
      logging.warning("Operation %d (%s) requires firmware patch %s to be loaded." % (
        i, op["op_id"].name, " and ".join(patch_list)))
      patches.extend(patch_list)
  # Remove duplicates
  patches = set(patches)
  # If any patches are needed, adjust operations for the patch
  if patches:
    for op in dnn["operations"]:
      DNNOperation.adjust_for_patches(dnn, op, patches)

  logging.info("Creating memory structures")
  try:
    mem_regions, mem_structs = mem_manager.create_memory_structures(dnn, cfg_dict)
  except ValueError as e:
    logging.error(e)
    raise RuntimeError("Invalid configuration for memory structures!")
  except RuntimeError as e:
    logging.error(e)
    raise RuntimeError("Memory size limits exceeded!")

  mem_report, errors = mem_manager.get_memory_report()

  if process_state == ProcessModelState.SECOND:
    if errors:
      process_state = ProcessModelState.FAILURE
    elif best_effort:
      # Success on the second pass shows there is a solution. With best_effort
      # enabled we return RETRY to try to find an optimized solution.
      process_state = ProcessModelState.RETRY
    else:
      process_state = ProcessModelState.SUCCESS
  elif not errors:
    process_state = ProcessModelState.SUCCESS
  else:
    dnn_ram_usage = mem_manager.recalculate_max_scratch_ram_size(cfg_dict)
    if ( process_state == ProcessModelState.INITIAL
         and ( dnn_ram_available_for_scratch < dnn_ram_usage 
           or enabled_1x1_optimizations > 0 ) ):
      process_state = ProcessModelState.SECOND
    elif dnn_ram_available_for_scratch < dnn_ram_usage:
      dnn_ram_available_for_scratch = dnn_ram_usage
      process_state = ProcessModelState.RETRY
    elif enabled_1x1_optimizations and best_effort:
      dnn_skip_1x1_optimization_count += 1
      process_state = ProcessModelState.RETRY
    else:
      print(mem_report)
      for e in errors:
        logging.error(e)
      process_state = ProcessModelState.FAILURE

  if process_state == ProcessModelState.FAILURE:
      print(mem_report)
      for e in errors:
        logging.error(e)
  return mem_regions, mem_structs, mem_report, process_state

if __name__ == "__main__":
    main()
