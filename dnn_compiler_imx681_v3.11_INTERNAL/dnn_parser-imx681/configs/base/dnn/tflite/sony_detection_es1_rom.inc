# ==============================================================================
# Common configuration parameters for Sony's face\hand\person detection algorithm
#
# This is the default algorithm that is stored in IMX681's ROM.
# ==============================================================================

# Name used as prefix on all generated file\variable names
DNN_NAME dnn_fd

# For Tensorflow: relative path and filename of the input .tflite file
# For PyTorch: "pytorch", which indicates that the model is read directly from
#  a pre-loaded data structure
DNN_MODEL ..\..\..\..\models\sony_detection_es1_rom.tflite

# ---------------------------
# Post-Processing
# ---------------------------

# Post-Processing layer type
# Opertions are:
#   ANCHOR_BOXES <data filename>
#   THRESHOLD    <data_mode> <compare_mode> <report mode> <report fields...>
#      data_mode: RAW, DEQUANT
#      compare_mode: GT, LT
#      report_mode: ALL, FILT_BY_CNT, FILT_BY_VAL
#      report_fields: ROW, BEST_COL, BEST_VAL, CNT, DATA, COMP
POSTPROCESSING ANCHOR_BOXES sony_detection_anchor_boxes.txt

# Mapping of DNN outputs to post-processing inputs
#  INPUT 0: anchor boxes
#    All of output buffer 0 (1242x4)
POSTPROC_DNN_DATA_OUT_IDX 0
POSTPROC_DNN_DATA_START_ROW 0
POSTPROC_DNN_DATA_START_COL 0
POSTPROC_DNN_DATA_NUM_ROWS 1242
POSTPROC_DNN_DATA_NUM_COLS 4

#  INPUT 1: confidences 
#    Cropped region of output buffer 1 (1242x3)
#      First column is cropped out since it represents the background class
POSTPROC_COMPARE_VALS_OUT_IDX 1
POSTPROC_COMPARE_VALS_START_ROW 0
POSTPROC_COMPARE_VALS_START_COL 1
POSTPROC_COMPARE_VALS_NUM_ROWS 1242
POSTPROC_COMPARE_VALS_NUM_COLS 3

# Mapping of columns in the COMPARE_VALS input to object types
#   Each column represents a different object type in this case
POSTPROC_COL_TO_OBJECT_TYPE [0, 1, 2]

# Postprocessing threshold values that need to be written to registers.
# For anchor boxes: [IOU_THRESH, HEIGHT_THRESH, WIDTH_THRESH, 
#                    CONF_THRESH0, CONF_THRESH1, CONF_THRESH2, 
#                    CONF_THRESH3, CONF_THRESH4, CONF_THRESH5]
# For threshold: [COMPARE_THRESH, CNT_THRESH]
POSTPROC_THRESHOLD_VALS [50, 15, 15, 45, 45, 45, 45, 45, 45]

# ---------------------------
# Multi-Layer Convolution Optimizations
# ---------------------------

# Maximum size, in bytes, of a convolution's output buffer before the compiler
# attempts to convert it to a multi-layer convolution (0 = Disabled)
ML_CONV_MAX_OUT_SIZE 4096

# Number of partitions an image is split into when a multi-layer convolution is
# performed. This should be a square number (e.g. 4, 9, 16) (0 = Disabled)
ML_CONV_NUM_PARTITIONS 4

# ------------------------
# Other configuration parameters
#-------------------------

# Selects implementation of ROI_POOL layer to use:
#  FW_ONLY - firmware-only implementation (low speed, high accuracy)
#  HW_ASSIST - hardware-assisted implementation (high speed, low accuracy, experimental)
ROI_POOL_MODE FW_ONLY
