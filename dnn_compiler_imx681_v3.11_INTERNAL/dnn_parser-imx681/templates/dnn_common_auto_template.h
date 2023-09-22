/*##############################################################################
 * Copyright %YEAR% Sony Semiconductor Solutions Corporation.
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of
 * Sony Semiconductor Solutions Corporation.
 * No part of this file may be copied, modified, sold, and distributed in any
 * form or by any means without prior explicit permission in writing of
 * Sony Semiconductor Solutions Corporation.
 */
/*############################################################################*/
/*
 * %AUTO_GEN%
 *
 * This file was generated at %TIME%
 * using the following command:
 *  python %CMD%
 */
#ifndef DNN_COMMON_AUTO_H_
#define DNN_COMMON_AUTO_H_

#include "imx_project.h"

/*******************************************************************************
 * Constant definitions
 ******************************************************************************/

/**
 * Maximum number of inputs an operation can have
 */
#define DNN_OPERATION_MAX_INPUTS %MAX_INPUTS%

/**
 * Number of different operations that are supported and stored in the operation
 * table.
 */
#define DNN_NUM_OPERATION_IDS %NUM_OP_IDS%

/**
 * Size of the misc_data memory block, containing pre-computed constants
 */
#define DNN_MISC_DATA_SIZE %MISC_DATA_SIZE%

/**
 * If 1, default quantization parameters are stored in VPU ROM. If 0,
 * they are stored in system ROM
 */
#define DNN_QUANT_PARAMS_IN_VPU_ROM %QUANT_IN_DNN_ROM%

/**
 * If 1, default misc data array is stored in VPU ROM. If 0,
 * it is stored in system ROM
 */
#define DNN_MISC_DATA_IN_VPU_ROM %MISC_IN_DNN_ROM%

/**
 * Number of fractional bits in quantization scale factors
 */
#define DNN_QUANT_SCALE_Q_FORMAT %SCALE_Q_FORMAT%

/**
 * Number of fractional bits in pre-computed offset factors
 */
#define DNN_QUANT_OFFSET_Q_FORMAT %OFFSET_Q_FORMAT%

/**
 * Magic number for clip_max parameter to a convolution, indicating that there
 * is no RELU activation attached to an operation
 */
#define DNN_RELU_NONE %RELU_NONE%

/**
 * Magic number for clip_max parameter to a convolution or relu indicating that
 * max clipping is not used (only min clipping)
 */
#define DNN_RELU_NO_MAX_CLIP %RELU_NO_MAX_CLIP%

/**
 * Magic number marking the end of the GENERATE_PROPOSALS output
 */
#define DNN_GENERATE_PROPOSALS_END %GENERATE_PROPOSALS_END%

/**
 * Number of columns in post processing compare values input
 */
#define DNN_POSTPROC_COMPARE_COLS %COMPARE_COLS%

/**
 * Number of fractional bits in anchor box data
 */
#define DNN_ANCHOR_IN_Q_FORMAT %ANCHOR_IN_Q_FORMAT%

/**
 * Number of fractional bits to use for decoded anchor boxes
 */
#define DNN_ANCHOR_DECODE_Q_FORMAT %ANCHOR_DECODE_Q_FORMAT%

/**
 * Number of fractional bits in an ROI's scale factor (for ROI_POOL layer)
 */
#define DNN_ROI_SCALE_Q_FORMAT %ROI_SCALE_Q_FORMAT%

/**
 * Number of fractional bits in POSTPROC_THRESHOLD's scale/offset
 */
#define DNN_POSTPROC_THRESHOLD_Q_FORMAT %POSTPROC_THRESHOLD_Q_FORMAT%

/*******************************************************************************
 * Enum definitions
 ******************************************************************************/
%ENUMS%

/*******************************************************************************
 * Structure definitions
 ******************************************************************************/

/**
 * Quanitization parameters for a single input buffer channel
 *
 * Values are converted from a uint8 value, q, to a floating point value, r,
 * using the following equation:
 *
 * r =  S (q - Z)
 */
typedef struct TAGS_DNN_QUANT_PARAMS {
  uint32_t num_params;    /*!< Total number of sets of quantization parameters */
  /**
   * All quantization values
   * First <num_params> * 4 bytes are uint32 scale values (S) in Q%Q_FORMAT%
   * fixed point format. Next <num_params> bytes are uint8 zeropoint values (z)
   */
  uint8_t values[%QUANT_PARAMS_SIZE%];

} S_DNN_QUANT_PARAMS;

/**
 * Buffer descriptor for a single input/output buffer in the DNN
 *
 * NOTE: This structure must be aligned such that uint32's start on 4 byte
 * boundaries and uint16's start on 2 byte boundaries
 */
typedef struct TAGS_DNN_BUFFER_DESCRIPTOR {
  uint32_t data_start_offset;  /*!< start index of this buffer's data in its
                                *!  memory region, in bytes */
  uint32_t batch_stride;       /*!< stride, in bytes, from the start of one batch to the
                                *!  start of the next batch */
  uint16_t quant_start_idx;    /*!< start index of this buffer's quantization
                                *!  parameters in S_DNN_QUANT_PARAMS data array */
  uint16_t num_batches;        /*!< number of batches in buffer (dims[0]) */
  uint16_t num_rows;           /*!< number of rows in buffer (dims[1]) */
  uint16_t num_cols;           /*!< number of columns in buffer (dims[2]) */
  uint16_t num_channels;       /*!< number of channels in buffer (dims[3]) */
  uint16_t row_stride;         /*!< stride, in bytes, from the start of one row to the
                                *!  start of the next row */
  uint8_t buffer_type;         /*!< buffer type, indicating where data is
                                *!  stored (see E_DNN_BUFFER_TYPE for values)*/
  uint8_t quant_type;          /*!< type of quantization, see E_DNN_QUANT_TYPE */
  uint8_t bpp_shift;           /*!< each element is 2^bpp_shift bytes in size (e.g.
                                *!  int32 has (1 << 2) = 4 bytes per pixel, so 
                                *!  bpp_shift = 2 */
  uint8_t reserved;            /*!< Reserved to make total size of struct aligned
                                *!  to 4 bytes */
} S_DNN_BUFFER_DESCRIPTOR;


/**
 * Union for a S_DNN_OPERATION's parameter field.  For some operations, the value
 * of param is interpretted as various bitfields. For operations that need more
 * than just 32-bits for their parameters, the value is interpretted as an index
 * into the misc data array where the operation's data is structure is stored.
 */
typedef union TAGU_DNN_OPERATION_PARAM {
  // Generic operation
  uint32_t misc_data_idx;         /*!< Index into misc data array where this
                                   *!  operation's data is stored */
  // RELU operation
  struct {
    uint32_t clip_max;            /*!< The maximum value to pass through, or:
                                   *!  DNN_RELU_NO_MAX = RELU with no max clipping */

  } relu;
  // INTERPOLATE operation
  struct {
    uint8_t  h_scale;             /*!< Scale factor for height */
    uint8_t  w_scale;             /*!< Scale factor for width */
    uint8_t  num_cols_dropped;    /*!< Number of columns to reduce the output size by:
                                   *!  out_width = w_scale * in_width - num_cols_dropped */
    uint8_t  num_rows_dropped;    /*!< Number of rows to reduce the output size by:
                                   *!  out_height = h_scale * in_height - num_rows_dropped */
  } interp;
  // MAX_POOL operation
  struct {
    uint32_t pad_left    : 4;  /*!< Number of columns of padding on left */
    uint32_t pad_right   : 4;  /*!< Number of columns of padding on right */
    uint32_t pad_top     : 4;  /*!< Number of rows of padding on top */
    uint32_t pad_bottom  : 4;  /*!< Number of rows of padding on bottom */
    uint32_t stride_w    : 4;  /*!< Window stride, in columns */
    uint32_t stride_h    : 4;  /*!< Window stride, in rows */
    uint32_t filter_w    : 4;  /*!< Window size, in columns */
    uint32_t filter_h    : 4;  /*!< Window size, in rows */
  } max_pool;
  // SOFTMAX operation
  struct {
    uint32_t axis; /*!< The axis to perform a softmax operation on */
  } softmax;
  // RESHAPE operation
  struct {
    uint16_t transpose; /*!< 1 if this reshape requires data to be converted between 
                         *   channel first and channel last memory order */
    uint16_t unpadded;  /*!< 1 if the output buffer does not have any padding,
                         *   and therefore the operation can be done in place */
  } reshape;
} U_DNN_OPERATION_PARAM;


/**
 * Parameter data for the ADDSUB operation (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_ADDSUB_DATA {
  uint32_t mode;           /*!< Add/Subtract mode (see E_DNN_ARITHMETIC_MODE) */
  int32_t scale_ratio_a;   /*!< Precalculated scale: sa/sc */
  int32_t scale_ratio_b;   /*!< Precalculated scale: sb/sc */
  int32_t offset;          /*!< Precalculated offset: zc - za*sa/sc - zb*sb/sc */
} S_DNN_ADDSUB_DATA;

/**
 * Parameter data for the MULTIPLY operation (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_MULTIPLY_DATA {
  uint32_t mode;           /*!< Add/Subtract mode (see E_DNN_ARITHMETIC_MODE) */
  int32_t scale_ratio;     /*!< Precalculated scale: sa*sb/sc */
} S_DNN_MULTIPLY_DATA;

/**
 * Parameter data for the CONCATENATE operation (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_CONCATENATE_DATA {
  uint32_t  axis;               /*!< Axis to use for concatenation (see
                                 *!  E_DNN_BUFFER_AXIS for values) */
   /* This is followed by:
    *  an int32_t scale_ratio for each input (Sa/Sb)
    *  an int32_t offset for each input (zb - Sa/Sb*za)
    */
} S_DNN_CONCATENATE_DATA;

/**
 * Parameter data for the CONV2D and DEPTHWISE_CONV2D operations (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_CONV2D_DATA {
  uint32_t  pad_left           : 4;  /*!< Number of columns of padding on left */
  uint32_t  pad_right          : 4;  /*!< Number of columns of padding on right */
  uint32_t  pad_top            : 4;  /*!< Number of rows of padding on top */
  uint32_t  pad_bottom         : 4;  /*!< Number of rows of padding on bottom */
  uint32_t  stride_w           : 2;  /*!< Convolution's stride on the X axis */
  uint32_t  stride_h           : 2;  /*!< Convolution's stride on the Y axis */
  uint32_t  mode               : 4;  /*!< Convolution mode (see E_DNN_CONVOLUTION_MODE) */
  int32_t  relu_clip_max       : 8;  /*!< If there is a fused activation function,
                                      *!  the max clip value */
   /* This is followed by:
    *  if this is a depthwise conv2d, a uint32 "channel_multiple"
    *  if this is a 1x1 optimized conv2d, a uint16 "dummy_cols_left" and uint16 "dummy_cols_right"
    *  an array of int32 scale_ratio values (Sa*Sb/Sc)
    *     (one for per-tensor quantization, one per output channel for per-channel
    *      quantization)
    */
} S_DNN_CONV2D_DATA;

/**
 * Parameter data for the FULLY_CONNECTED operation (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_FULLY_CONNECTED_DATA {
  int32_t  relu_clip_max;      /*!< If there is a fused activation function,
                                *!  the max clip value */
   /* This is followed by:
    *  an array of int32 scale_ratio values (Sa*Sb/Sc)
    *     (one for per-tensor quantization, one per output channel for per-channel
    *      quantization)
    */
} S_DNN_FULLY_CONNECTED_DATA;

/**
 * Parameter data for the GENERATE_PROPOSALS operation (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_GENERATE_PROPOSALS_DATA {
  int32_t scale[DNN_ANCHOR_BOX_INPUT_SIZE+1];  /*!< Pre-computed scale factors
                                                *!  for each box coordinate and
                                                *!  confidences in ANCHOR_SCALE_Q_FORMAT */
  int32_t offset[DNN_ANCHOR_BOX_INPUT_SIZE+1]; /*!< Pre-computed offset factors
                                                *!  for each box coordinate and
                                                *!  confidences in ANCHOR_SCALE_Q_FORMAT */
  int32_t out_scale;                           /*!< Output scale factor, which is 1/quant_scale,
                                                *!  represented in DNN_ANCHOR_DECODE_Q_FORMAT */
  int32_t out_offset;                          /*!< Output offset, which is quant_zero */
  uint32_t anchor_box_addr;                    /*!< Address in static VPU memory of
                                                *!  the anchor box data */
  uint8_t width_thresh;                        /*!< Minimum width of an output box */
  uint8_t height_thresh;                       /*!< Minimum height of an output box */
  uint8_t iou_thresh;                          /*!< Minimum IOU between two boxes to include both */
  uint8_t conf_thresh;                         /*!< Minimum confidence of an output box */
} S_DNN_GENERATE_PROPOSALS_DATA;

/**
 * Parameter data for the POSTPROC_ANCHOR_BOXES operations (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_ANCHOR_BOX_DATA {
  int32_t scale[DNN_ANCHOR_BOX_INPUT_SIZE+1];  /*!< Pre-computed scale factors
                                                *!  for each box coordinate and
                                                *!  confidences in ANCHOR_SCALE_Q_FORMAT */
  int32_t offset[DNN_ANCHOR_BOX_INPUT_SIZE+1]; /*!< Pre-computed offset factors
                                                *!  for each box coordinate and
                                                *!  confidences in ANCHOR_SCALE_Q_FORMAT */
  uint32_t anchor_box_addr;                    /*!< Address in static VPU memory of
                                                *!  the anchor box data */
  uint8_t col_to_object_type[DNN_POSTPROC_COMPARE_COLS];  /*!< mapping of columns to object types:
                                                           *!  col_to_object_type[col] = object_type;
                                                           */

} S_DNN_ANCHOR_BOX_DATA;

/**
 * Parameter data for the POSTPROC_THRESHOLD operations (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_THRESHOLD_DATA {
  uint8_t   compare_mode;    /*!< Comparison mode (see E_DNN_THRESHOLD_COMPARE_MODE) */
  uint8_t   report_mode;     /*!< Reporting mode (see E_DNN_THRESHOLD_REPORT_MODE) */
  uint16_t  report_fields;   /*!< Fields to report (bitwise OR of fields from 
                              *!  E_DNN_THRESHOLD_REPORT_FIELDS) */
  int32_t   compare_scale;   /*!< Scale to apply to compare values to dequantize and
                              *!  convert to score of 0 - 100 (in DNN_POSTPROC_THRESHOLD_Q_FORMAT)*/
  int32_t   compare_offset;  /*!< Offset to apply to compare values to dequantize and
                              *!  convert to score of 0 - 100 (in DNN_POSTPROC_THRESHOLD_Q_FORMAT)*/
  uint8_t col_to_object_type[DNN_POSTPROC_COMPARE_COLS];  /*!< mapping of columns to object types:
                                                           *!  col_to_object_type[col] = object_type;
                                                           */

} S_DNN_THRESHOLD_DATA;

/**
 * Parameter data for the ROI_POOL operations (as it is stored
 * in the misc data array)
 */
typedef struct TAGS_DNN_ROI_POOL_DATA {
  uint32_t mode;       /*!< Implementation mode (see E_DNN_ROI_POOL_MODE) */
  int32_t scale_a;     /*!< Pre-calculated scale factor for input a in 
                        *!  DNN_QUANT_SCALE_Q_FORMAT: sa/sc */
  int32_t scale_b;     /*!< Pre-calculated scale factor for input b (roi) in 
                        *!  DNN_ROI_SCALE_Q_FORMAT: sa*spatial_scale) */
  int32_t offset_a;    /*!< Pre-calculated offset for input a in
                        *!  DNN_QUANT_SCALE_Q_FORMAT: zb - sa/sc*za */
  int32_t offset_b;    /*!< Pre-calculated offset for input b in
                        *!  DNN_ROI_SCALE_Q_FORMAT: scale_b * -zb */
} S_DNN_ROI_POOL_DATA;

/**
 * Description of a single operation in the DNN
 *
 * NOTE: This structure must be aligned such that uint32's start on 4 byte
 * boundaries and uint16's start on 2 byte boundaries
 */
typedef struct TAGS_DNN_OPERATION {
  uint8_t op_id;                    /*!< ID of this operation, which defines which
                                     *!  function is used to implement it
                                     *!  (see E_DNN_OP_ID for values) */
  uint8_t num_inputs;               /*!< Number of valid inputs */
  uint16_t input_buffer_idx[DNN_OPERATION_MAX_INPUTS];  /*!< Buffer indexes of each
                                                         *!  input to the operation */
  uint16_t output_buffer_idx;       /*!< Buffer index of the output of this operation */
  uint32_t working_mem_addr;        /*!< Start address of working memory that has
                                     *!  been allocated for this operation's intermediate
                                     *!  calculations. */
  U_DNN_OPERATION_PARAM param;      /*!< Generic parameter value for this operation */
} S_DNN_OPERATION;

/**
 * Top-level structure that describes a DNN's structure. The data field consists of the
 * following information concatenated together:
 *   1. <num_buffers> S_DNN_BUFFER_DESCRIPTOR structures
 *   2. array of S_DNN_OPERATION structures, where the last one has op_id = INVALID.
 */
typedef struct TAGS_DNN_OPTABLE {
  uint32_t static_data_size;        /*!< total bytes of static data in DNN */
  uint32_t num_buffers;             /*!< total number of buffer descriptors */
  uint32_t scratch_ram_start_addr;  /*!< start address of scratch RAM region of memory */
  uint8_t data[%OPTABLE_SIZE%];               /*!< concatenation of buffers and operations */
} S_DNN_OPTABLE;

/**
 * Function pointer type that is used to implement each operation
 *
 * @param[in]  op          pointer to the operation's data structure
 * @return     true if a VPU instruction was started and we need to wait for
 *             it to complete.
 */
typedef void (*DNN_OP_FUNC_T)(S_DNN_OPERATION *op);

/************************************************************************/
/* External function prototype declarations                             */
/************************************************************************/
extern void %DNN_OP_FUNCS%(S_DNN_OPERATION *params);

/************************************************************************/
/* Function table for mapping operation IDs to functions                */
/************************************************************************/
extern const DNN_OP_FUNC_T dnn_op_func_table[DNN_NUM_OPERATION_IDS];

#endif /* DNN_COMMON_AUTO_H_ */


