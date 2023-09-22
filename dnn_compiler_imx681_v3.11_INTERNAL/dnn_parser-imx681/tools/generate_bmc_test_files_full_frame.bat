@setlocal

@Rem Total number of images to generate files for
@set num_images=64

@Rem Delete existing output directory, then create a new one
@if exist ".\output" @RD /S /Q ".\output"
@mkdir ".\output"

@Rem Generate files for original test, image 0
@mkdir ".\output\dnn_full_frame_processing"
@echo IMAGE0
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\input_image_tensor.txt ^
  output\dnn_full_frame_processing\PRELOAD0.txt --convert_unsigned
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\detection_results.txt ^
  output\dnn_full_frame_processing\EXPECTED_RESULT.txt
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\BoxPredictor_concat.txt ^
  output\dnn_full_frame_processing\EXPECTED_POSTPROC_INPUT0.txt
@python c_model_to_bmc_format.py ^
  face_hand_person_19.2.1_test_examples\test_0\ClassPredictor_sigmoid.txt ^
  output\dnn_full_frame_processing\EXPECTED_POSTPROC_INPUT1.txt

@Rem Generate files for all remaining images
@set /A last_image = %num_images% - 1
@for /l %%x in (1, 1, %last_image%) do @(
  @mkdir ".\output\dnn_full_frame_processing_image%%x"
  @echo IMAGE%%x
  @python c_model_to_bmc_format.py ^
    face_hand_person_19.2.1_test_examples\test_%%x\input_image_tensor.txt ^
    output\dnn_full_frame_processing_image%%x\PRELOAD0.txt --convert_unsigned
  @python c_model_to_bmc_format.py ^
    face_hand_person_19.2.1_test_examples\test_%%x\detection_results.txt ^
    output\dnn_full_frame_processing_image%%x\EXPECTED_RESULT.txt
  @python c_model_to_bmc_format.py ^
    face_hand_person_19.2.1_test_examples\test_%%x\BoxPredictor_concat.txt ^
    output\dnn_full_frame_processing_image%%x\EXPECTED_POSTPROC_INPUT0.txt
  @python c_model_to_bmc_format.py ^
    face_hand_person_19.2.1_test_examples\test_%%x\ClassPredictor_sigmoid.txt ^
    output\dnn_full_frame_processing_image%%x\EXPECTED_POSTPROC_INPUT1.txt
)
@endlocal
@exit /b
