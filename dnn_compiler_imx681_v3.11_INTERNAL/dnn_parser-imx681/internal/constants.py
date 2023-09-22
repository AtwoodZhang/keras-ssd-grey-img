# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------

from enum import Enum

# Version number
DNN_COMPILER_VERSION = 3.11

# Target hardware for this version of the compiler
DNN_COMPILER_TARGET_PLATFORM = "IMX681"

# If True, build for internal use. This includes information about VPU instructions
# in the output and reduces restrictions on parameter ranges. If False, build
# for external customer use
DNN_COMPILER_INTERNAL_USE = True

class BufferType(Enum):
  """
  Values of "buffer_type" field in compiler output dictionary
  """
  MODEL_INPUT  = 0    # Buffer is an input to the entire DNN
  STATIC_DATA  = 1    # Buffer is static data (e.g. weight or bias)
  SCRATCH_RAM  = 2    # Buffer is a temporary result to be stored in scratch RAM

class DataType(Enum):
  """
  Values of "data_type" field in compiler output dictionary. Must match
  the VPU's type definitions (see vpu.h)
  """
  CHAR   = 0
  SHORT  = 1
  LONG   = 2

class BufferAxis(Enum):
  """
  Indexes of each axis in a buffer
  """
  ROW     = 1
  COLUMN  = 2
  CHANNEL = 3

class QuantType(Enum):
  """
  Types of quantization that can be applied in a model
  """
  PER_TENSOR  = 0
  PER_CHANNEL = 1
  PER_BATCH   = 2

class PostProcTransposeType(Enum):
  """
  Type of transposing to do to DNN outputs before passing them into post processing
  """
  NONE              = 0
  DATA_ONLY         = 1
  COMP_ONLY         = 2
  BOTH_SAME_BUFFER  = 3
  BOTH_DIFF_BUFFERS = 4

class ConvMode(Enum):
  """
  Mode of a convolution that selects how it is implemented in firmware. Different
  implementations are used for different sized buffers and filters.
  """
  DEFAULT              = 0
  OPTIMIZED_1X1_FILTER = 1
  OPTIMIZED_3X3_FILTER = 2

class ArithmeticMode(Enum):
  """
  Mode of an arithmetic operation (ADDSUB, MULTIPLY, etc)
  """
  SCALAR             = 0
  SCALAR_FLATTENED   = 1
  MATRIX             = 2
  MATRIX_FLATTENED   = 3

class ROIPoolMode(Enum):
  """
  Mode of the ROI Pool operation
  """
  FW_ONLY            = 0  # Firmware only (low speed, high accuracy)
  HW_ASSIST          = 1  # HW-assisted (high speed, low accuracy, experimental)

class AnchorBoxInput(Enum):
  """
  The index of each value in the Anchor Box post processing raw box inputs
  """
  Y          = 0
  X          = 1
  H          = 2
  W          = 3
  SIZE       = 4

class AnchorBoxOutput(Enum):
  """
  The index of each value in the Anchor Box post processing output
  """
  CLASS     = 0
  X         = 1
  Y         = 2
  W         = 3
  H         = 4
  CONF      = 5
  SIZE      = 6

class ThresholdDataMode(Enum):
  """
  Data modes supported by threshold postprocessing. These determine what
  data is compared to the threshold
  """
  RAW            = 0  # Raw quantized input (-128 to 127)
  DEQUANT        = 1  # Dequantized and scaled input (0 to 100)


class ThresholdCompareMode(Enum):
  """
  Compare modes supported by threshold postprocessing. These determine what
  criteria a value must meet to pass the threshold check.
  """
  GT        = 0  # Value must be greater than the threshold
  LT        = 1  # Value must be less than the threshold

class ThresholdReportMode(Enum):
  """
  Report modes supported by threshold postprocessing. These determine what data
  is written to the output when the threshold check passes.
  """
  ALL         = 0  # All rows of data are sent whenever any row passes threshold check
  FILT_BY_CNT = 1  # Data is filtered and only rows that pass threshold check are sent.
                   # If there isn't space for all rows in output, ones with higher count
                   # are sent.
  FILT_BY_VAL = 2  # Data is filtered and only rows that pass threshold check are sent.
                   # If there isn't space for all rows in output, ones with higher best val
                   # are sent.

class ThresholdReportField(Enum):
  """
  Report fields supported by threshold postprocessing. These determine what data
  is reported to the host when the threshold check passes. Note that these use
  one-hot encoding so that multiple can be enabled.
  """
  ROW      = 0x01  # Row index
  BEST_COL = 0x02  # Column index of "best" value (min or max)
  BEST_VAL = 0x04  # "Best" value (min or max)
  CNT      = 0x08  # Number of values that passed threshold check
  DATA     = 0x10  # Full row from DNN_DATA input matrix
  COMP     = 0x20  # Full row from COMP_VALS input matrix

class DNNOperationID(Enum):
  """
  Values of "op_id" field in compiler output dictionary
  """
  CONV_2D                 = 0x00
  DEPTHWISE_CONV_2D       = 0x01
  FULLY_CONNECTED         = 0x02
  RELU                    = 0x03
  ADDSUB                  = 0x04
  MULTIPLY                = 0x05
  CONCATENATE             = 0x06
  INTERPOLATE             = 0x07
  SIGMOID                 = 0x08
  SOFTMAX                 = 0x09
  MAX_POOL                = 0x0A
  RESHAPE                 = 0x0B
  POSTPROC_ANCHOR_BOXES   = 0x0C
  POSTPROC_THRESHOLD      = 0x0D
  TRANSPOSE               = 0x0E
  ROI_POOL                = 0x0F
  GENERATE_PROPOSALS      = 0x10
  # Placeholders for future operations that can be implemented via FW patches
  CUSTOM0                 = 0x11
  CUSTOM1                 = 0x12
  CUSTOM2                 = 0x13
  CUSTOM3                 = 0x14
  CUSTOM4                 = 0x15
  CUSTOM5                 = 0x16
  CUSTOM6                 = 0x17
  CUSTOM7                 = 0x18
  INVALID                 = 0xFF

class Endianness(Enum):
  """
  Values of "endianness" in configuration file
  """
  BIG = 0
  LITTLE = 1

class SensorVersion(Enum):
  """
  Values of "SensorVersion" in configuration file.
  """
  ES1 = "ES1"
  ES2 = "ES2"

class MemoryOrder(Enum):
  """
  Values of "memory-order" in configuration file.

  Determines if buffer data is laid out channel-first or channel-last in memory
  """
  CHANNEL_FIRST = 0
  CHANNEL_LAST  = 1

class OutputMode(Enum):
  """
  Values of "output_mode" in configuration file
  """
  ROM = 0
  I2C = 1

class MemoryRegion(Enum):
  """
  Memory regions that structures can be stored in
  """
  SYS_ROM = 0
  SYS_RAM = 1
  DNN_ROM = 2
  DNN_RAM = 3

class MemoryStructure(Enum):
  """
  Memory structures that are stored for a DNN
  """
  STATIC_DATA  = 0
  SCRATCH_RAM  = 1
  OPTABLE      = 2
  MISC_DATA    = 3
  QUANT_PARAMS = 4
  TEST_RAM     = 5


# List of operations that can be done in place, and do not need separate buffers
# for input and output
INPLACE_OPERATIONS = [
  DNNOperationID.RESHAPE,
]

# Maximum number of dimensions a buffer can have
DNN_BUFFER_MAX_DIMENSIONS = 4

# Maximum number of input buffers an operation can have
DNN_OPERATION_MAX_INPUTS = 6

# Maximum number of output buffers an operation can have
DNN_OPERATION_MAX_OUTPUTS = 1

# Number of output object types supported
DNN_NUM_OBJECT_TYPES = 5

# Number of fractional bits in the fixed point reprentation of quantization
# scale factors
DNN_QUANT_SCALE_Q_FORMAT = 24

# Number of fractional bits in the fixed point reprentation of pre-computed
# scale and offset factors
DNN_QUANT_OFFSET_Q_FORMAT = 22

# Minimum/Maximum supported value in DNN_QUANT_SCALE_Q_FORMAT format
DNN_QUANT_SCALE_MIN = -1*(2.0**31.0) / (2.0**DNN_QUANT_SCALE_Q_FORMAT)
DNN_QUANT_SCALE_MAX = (2.0**31.0 - 1.0) / (2.0**DNN_QUANT_SCALE_Q_FORMAT)

# Constant used to indicate that no relu operation is performed
RELU_NONE = 0x80

# Constant used to indicate that relu doesn't have a max clip (only a min clip)
RELU_NO_MAX_CLIP = 0x7F

# Magic number marking the end of GENERATE_PROPOSALS output
GENERATE_PROPOSALS_END = 0xFF

# Dimension indexes for each axis
DIMS_NUM_BATCHES = 0
DIMS_NUM_ROWS = 1
DIMS_NUM_COLS = 2
DIMS_NUM_CHANNELS = 3

# Convolution input buffer indices
CONV_INPUT_IMAGE = 0
CONV_INPUT_WEIGHTS = 1
CONV_INPUT_BIASES = 2

# Size of the header that is added to the output of the DNN
DNN_OUTPUT_HEADER_SIZE = 4

# Number of fractional bits in the anchor box data
DNN_ANCHOR_IN_Q_FORMAT = 7

# Number of fractional bits to use for decoded anchor boxes
DNN_ANCHOR_DECODE_Q_FORMAT = 15

# Number of fractional bits in the ROI_POOL's scale factor
DNN_ROI_SCALE_Q_FORMAT = 15

# Number of fractional bits in POSTPROC_THRESHOLD's scale/offset
DNN_POSTPROC_THRESHOLD_Q_FORMAT = 15

# VPU runs at 5.556 ns period, but there is some VPU overhead. Based on actual
# simulations, a period of 5.965 ns per element seems to give a more accurate
# estimation
VPU_CLOCK_PERIOD_S = 5.965E-9

# Size of the DNN's input buffer
DNN_INPUT_COLS = 160
DNN_INPUT_ROWS = 120

VALID_ML_CONV_NUM_PARTITIONS = [4, 9, 16]
