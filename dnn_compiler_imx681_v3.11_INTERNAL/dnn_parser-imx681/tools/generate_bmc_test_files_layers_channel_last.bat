@setlocal

@Rem Delete existing output directory, then create a new one
@if exist ".\output" @RD /S /Q ".\output"
@mkdir ".\output"

@mkdir ".\output\dnn_operation_add"
@echo DNN_OPERATION_ADD
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_2_project.txt ^
  output\dnn_operation_add\PRELOAD0.txt 1 30 40 8
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_1_project.txt ^
  output\dnn_operation_add\PRELOAD1.txt 1 30 40 8
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_2_add.txt ^
  output\dnn_operation_add\EXPECTED_RESULT.txt 1 30 40 8

@mkdir ".\output\dnn_operation_concatenate"
@echo DNN_OPERATION_CONCATENATE
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_0_conv.txt ^
  output\dnn_operation_concatenate\PRELOAD0.txt 1 600 1 4
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_1_conv.txt ^
  output\dnn_operation_concatenate\PRELOAD1.txt 1 480 1 4
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_2_conv.txt ^
  output\dnn_operation_concatenate\PRELOAD2.txt 1 120 1 4
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_3_conv.txt ^
  output\dnn_operation_concatenate\PRELOAD3.txt 1 36 1 4
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_4_conv.txt ^
  output\dnn_operation_concatenate\PRELOAD4.txt 1 6 1 4
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_concat.txt ^
  output\dnn_operation_concatenate\EXPECTED_RESULT.txt 1 1242 1 4

@mkdir ".\output\dnn_operation_conv2d"
@echo DNN_OPERATION_CONV2D
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_depthwise.txt ^
  output\dnn_operation_conv2d\PRELOAD0.txt 1 30 40 8 ^
  --offset 19520 --rowsize 320 --rowstride 640 --nrows 30
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_project.txt ^
  output\dnn_operation_conv2d\EXPECTED_RESULT.txt 1 30 40 8 ^
  --offset 19520 --rowsize 320 --rowstride 640 --nrows 30

@mkdir ".\output\dnn_operation_conv2d_opt1x1"
@echo DNN_OPERATION_CONV2D_OPT1X1
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_20_depthwise.txt ^
  output\dnn_operation_conv2d_opt1x1\PRELOAD0.txt 1 1 1 32
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_20_project.txt ^
  output\dnn_operation_conv2d_opt1x1\EXPECTED_RESULT.txt 1 1 1 16

@mkdir ".\output\dnn_operation_depthwise_conv2d"
@echo DNN_OPERATION_DEPTHWISE_CONV2D
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_project.txt ^
  output\dnn_operation_depthwise_conv2d\PRELOAD0.txt 1 30 40 8 ^
  --offset 19520 --rowsize 320 --rowstride 640 --nrows 30
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\FeatureExtractor_expanded_conv_1_depthwise.txt ^
  output\dnn_operation_depthwise_conv2d\EXPECTED_RESULT.txt 1 15 20 16 ^
  --offset 9920 --rowsize 320 --rowstride 640 --nrows 15

@mkdir ".\output\dnn_operation_postproc_anchor_boxes"
@echo DNN_OPERATION_POSTPROC_ANCHOR_BOXES
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\BoxPredictor_concat.txt ^
  output\dnn_operation_postproc_anchor_boxes\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\ClassPredictor_sigmoid.txt ^
  output\dnn_operation_postproc_anchor_boxes\PRELOAD1.txt
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\detection_results.txt ^
  output\dnn_operation_postproc_anchor_boxes\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_read_image"
@echo DNN_OPERATION_READ_IMAGE
@python c_model_to_bmc_format.py ^
  output_expected\main_detection\input_image_tensor.txt ^
  output\dnn_operation_read_image\PRELOAD0.txt --convert_unsigned
@python c_model_to_bmc_format.py ^
  output_expected\main_detection\input_image_tensor.txt ^
  output\dnn_operation_read_image\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_sigmoid"
@echo DNN_OPERATION_SIGMOID
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\ClassPredictor_concat.txt ^
  output\dnn_operation_sigmoid\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\ClassPredictor_sigmoid.txt ^
  output\dnn_operation_sigmoid\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_reshape_transpose"
@echo DNN_OPERATION_RESHAPE_TRANSPOSE
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_0_conv.txt ^
  output\dnn_operation_reshape_transpose\PRELOAD0.txt 1 15 20 8
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_detection\BoxPredictor_0_conv.txt ^
  output\dnn_operation_reshape_transpose\EXPECTED_RESULT.txt 1 600 1 4

@mkdir ".\output\dnn_operation_conv2d_3x3"
@echo DNN_OPERATION_CONV2D_3X3
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_conv2d_3x3\PRELOAD0.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_classification\04_conv.txt ^
  output\dnn_operation_conv2d_3x3\PRELOAD1.txt 1 30 40 8
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_conv2d_3x3\SYS_PRELOAD.txt 
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_classification\05_conv.txt ^
  output\dnn_operation_conv2d_3x3\EXPECTED_RESULT.txt 1 30 40 8


@mkdir ".\output\dnn_operation_fully_connected"
@echo DNN_OPERATION_FULLY_CONNECTED
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_fully_connected\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\24_multiply.txt ^
  output\dnn_operation_fully_connected\PRELOAD1.txt
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_fully_connected\SYS_PRELOAD.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\25_dense_4.txt ^
  output\dnn_operation_fully_connected\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_max_pool"
@echo DNN_OPERATION_MAX_POOL
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_max_pool\PRELOAD0.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_classification\08_conv.txt ^
  output\dnn_operation_max_pool\PRELOAD1.txt 1 15 20 40
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_max_pool\SYS_PRELOAD.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\main_classification\09_maxpool.txt ^
  output\dnn_operation_max_pool\EXPECTED_RESULT.txt 1 8 10 40

@mkdir ".\output\dnn_operation_multiply"
@echo DNN_OPERATION_MULTIPLY
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_multiply\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\22_subtract.txt ^
  output\dnn_operation_multiply\PRELOAD1.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\23_dense.txt ^
  output\dnn_operation_multiply\PRELOAD2.txt
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_multiply\SYS_PRELOAD.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\24_multiply.txt ^
  output\dnn_operation_multiply\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_postproc_threshold"
@echo DNN_OPERATION_POSTPROC_THRESHOLD
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_postproc_threshold\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\28_softmax.txt ^
  output\dnn_operation_postproc_threshold\PRELOAD1.txt
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_postproc_threshold\SYS_PRELOAD.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\threshold_results.txt ^
  output\dnn_operation_postproc_threshold\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_softmax"
@echo DNN_OPERATION_SOFTMAX
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_softmax\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\27_dense.txt ^
  output\dnn_operation_softmax\PRELOAD1.txt
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_softmax\SYS_PRELOAD.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\28_softmax.txt ^
  output\dnn_operation_softmax\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_subtract"
@echo DNN_OPERATION_SUBTRACT
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_vpu_memory.bin ^
  output\dnn_operation_subtract\PRELOAD0.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\20_dense.txt ^
  output\dnn_operation_subtract\PRELOAD1.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\21_dense.txt ^
  output\dnn_operation_subtract\PRELOAD2.txt
@python bin_to_bmc_format.py ^
  ..\output\dnn_classification_system_memory.bin ^
  output\dnn_operation_subtract\SYS_PRELOAD.txt
@python c_model_to_bmc_format.py ^
  output_expected\main_classification\22_subtract.txt ^
  output\dnn_operation_subtract\EXPECTED_RESULT.txt

@mkdir ".\output\dnn_operation_interpolate"
@echo DNN_OPERATION_INTERPOLATE
@python bin_to_bmc_format.py ^
  ..\output\test_net_1_vpu_memory.bin ^
  output\dnn_operation_interpolate\PRELOAD0.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\layer_parameter_test_1\layer_05.txt ^
  output\dnn_operation_interpolate\PRELOAD1.txt 1 8 10 16
@python bin_to_bmc_format.py ^
  ..\output\test_net_1_system_memory.bin ^
  output\dnn_operation_interpolate\SYS_PRELOAD.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\layer_parameter_test_1\layer_06.txt ^
  output\dnn_operation_interpolate\EXPECTED_RESULT.txt 1 15 20 16

@mkdir ".\output\dnn_operation_reshape"
@echo DNN_OPERATION_RESHAPE
@python bin_to_bmc_format.py ^
  ..\output\test_net_1_vpu_memory.bin ^
  output\dnn_operation_reshape\PRELOAD0.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\layer_parameter_test_1\layer_00.txt ^
  output\dnn_operation_reshape\PRELOAD1.txt 1 60 80 4
@python bin_to_bmc_format.py ^
  ..\output\test_net_1_system_memory.bin ^
  output\dnn_operation_reshape\SYS_PRELOAD.txt
@python c_model_to_bmc_format_channel_last.py ^
  output_expected\layer_parameter_test_1\layer_00.txt ^
  output\dnn_operation_reshape\EXPECTED_RESULT.txt 1 30 40 16

@endlocal
@exit /b
