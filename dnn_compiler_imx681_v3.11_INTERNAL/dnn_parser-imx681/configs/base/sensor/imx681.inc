# ==============================================================================
# Common configuration parameters for the IMX681 image sensor.
#
# This configuration should be included any time a DNN is targeted to
# be run on IMX681.
# ==============================================================================

# ------------------------
# Memory addresses & sizes
#-------------------------

# DNN RAM: 
#   0x00280000 - 0x002BB4FF: DNN RAM (242,944 bytes)
#   0x002BB500 - 0x002BFFFF: Image RAM (19,200 bytes)
DNN_RAM_START_ADDR            0x00280000
DNN_RAM_MAX_SIZE              242944

# DNN ROM:
#   0x002C0000 - 0x002D4000: DNN ROM (80 KB)
DNN_ROM_START_ADDR            0x002C0000
DNN_ROM_MAX_SIZE              81920

# System RAM:
#   0x00180000 - 0x0018200F: Reserved for ROM firmware
#   0x00182010 - 0x00185BFF: Reserved for firmware patches
#   0x00185C00 - 0x0018BBFF: Reserved for DNN structures (24 KB)
#   0x0018BC00 - 0x0018BFFF: Reserved for stack
SYSTEM_RAM_START_ADDR         0x00185C00
SYSTEM_RAM_MAX_SIZE           24576

# System ROM:
#   Approximately 44 KB available for DNN structures in ROM
SYSTEM_ROM_MAX_SIZE           45056

# ------------------------
# Register addresses & i2c sequences
# ------------------------

# Addresses of customer registers for DNN
REG_DNN_WEIGHT_ADDR             0x0500
REG_DNN_OPTABLE_ADDR            0x0504
REG_DNN_QUANT_PARAM_ADDR        0x0508
REG_DNN_MISC_DATA_ADDR          0x050c
REG_DNN_POSTPROC_THRESHOLD0     0x0510
NUM_DNN_POSTPROC_THRESHOLD_REGS      9

# Group used for remapped register writes (Must be a hex digit 0 - F)
GROUP_REMAP_ID                  A

# Address of group remap register to use for loading DNN
# In this case, it is set to Group A
REG_GROUP_REMAP                0x3E28

# I2C initialization sequence to perform before register writes
#   0x30EB = 0x05
#   0x30EB = 0x0C
#   0x300A = 0xFFFF
#   0x30EB = 0x05
#   0x30EB = 0x09
#   0x3532 = 0xFFFF
I2C_INIT_SEQ  [0x30, 0xeb, 0x05, 0x30, 0xeb, 0x0c, 0x30, 0x0a, 0xff, 0x30, 0x0b, 0xff, 0x30, 0xeb, 0x05, 0x30, 0xeb, 0x09, 0x35, 0x32, 0xff, 0x35, 0x33, 0xff]

# ------------------------
# Other configuration parameters
#-------------------------

# If 1, allocate scratch RAM backwards from the end of memory (prefer high addresses)
# If 0, allocate scratch RAM forwards from the start of memory (prefer low addresses)
# 
# On IMX681, higher addresses are preferred because accesses consume less power
REVERSE_ALLOC_SCRATCH_RAM 1

# Endianness of data in generated output files ("big" or "little")
OUTPUT_ENDIANNESS big

# Memory order for storing 3D buffer data.
#  channel_first: channels stored first, then columns, then rows
#  channel_last: columns stored first, then rows, then channels
MEMORY_ORDER channel_first

# Input directory containing .c and .h file templates
TEMPLATE_DIRECTORY ..\..\..\templates

# Input directory containing data files (e.g. anchor boxes)
DATA_DIRECTORY ..\..\..\data

# Sensor version either es1 or es2
SENSOR_VERSION es1

# During normal operation, the DNN compiler will determine which patches are needed
# and adjust the output to be compatible with those patches. If FORCE_PATCHES is
# defined, it will adjust the output to be compatible with all listed patches even
# if it does not detect that they are required.
#
# Options:
#   ES1:
#      170940_177827 - patch for multibatch, constant overflow, and pytorch reshape issues in DNN
#   ES2:
#      189270 - Multiply Scalar bug
# Example:
#   ES1:
#      FORCE_PATCHES ["170940_177827"]
#   ES2:
#      FORCE_PATCHES ["189270"]

