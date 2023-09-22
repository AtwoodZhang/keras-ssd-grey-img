@setlocal

@Rem Total number of test nets to generate files for
@set num_images=11

@Rem Delete existing output directory, then create a new one
@if exist ".\output" @RD /S /Q ".\output"
@mkdir ".\output"

@Rem Generate files for all remaining images
@set /A last_image = %num_images% - 1
@for /l %%x in (0, 1, %last_image%) do @(
  @mkdir ".\output\test_net_%%x"
  @echo TEST_NET_%%x
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_net_%%x_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
  @cd tools
  @copy ..\output\test_net_%%x_* .\output\test_net_%%x
  @python bin_to_bmc_format.py ^
    output\test_net_%%x\test_net_%%x_system_memory.bin ^
    output\test_net_%%x\SYS_PRELOAD.txt
  @python bin_to_bmc_format.py ^
    output\test_net_%%x\test_net_%%x_dnn_memory.bin ^
    output\test_net_%%x\BMC_PRELOAD.txt
)

@mkdir ".\output\test_postproc"
@echo TEST_POSTPROC
@cd ..
@python dnn_compiler.py configs\test\imx681_test_postproc_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_postproc* .\output\test_postproc
@python bin_to_bmc_format.py ^
  output\test_postproc\test_postproc_system_memory.bin ^
  output\test_postproc\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_postproc\test_postproc_dnn_memory.bin ^
  output\test_postproc\BMC_PRELOAD.txt


@mkdir ".\output\dnn_fd"
@echo DETECTION
@cd ..
@python dnn_compiler.py configs\imx681_tflite_detection_es1_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\dnn_fd* .\output\dnn_fd
@python bin_to_bmc_format.py ^
  output\dnn_fd\dnn_fd_system_memory.bin ^
  output\dnn_fd\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\dnn_fd\dnn_fd_dnn_memory.bin ^
  output\dnn_fd\BMC_PRELOAD.txt

@mkdir ".\output\dnn_classification"
@echo CLASSIFICATION
@cd ..
@python dnn_compiler.py configs\imx681_tflite_classification_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\sony_classification_summary.txt .\output\dnn_classification\dnn_classification_summary.txt
@copy ..\output\sony_classification_system_memory.bin .\output\dnn_classification\dnn_classification_system_memory.bin
@copy ..\output\sony_classification_dnn_memory.bin .\output\dnn_classification\dnn_classification_dnn_memory.bin
@python bin_to_bmc_format.py ^
  output\dnn_classification\dnn_classification_system_memory.bin ^
  output\dnn_classification\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\dnn_classification\dnn_classification_dnn_memory.bin ^
  output\dnn_classification\BMC_PRELOAD.txt


@mkdir ".\output\test_roi_pool"
@echo ROI_POOL
@cd ..
@python dnn_compiler.py configs\test\imx681_test_roi_pool_1_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_roi_pool* .\output\test_roi_pool
@python bin_to_bmc_format.py ^
  output\test_roi_pool\test_roi_pool_system_memory.bin ^
  output\test_roi_pool\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_roi_pool\test_roi_pool_dnn_memory.bin ^
  output\test_roi_pool\BMC_PRELOAD.txt

@mkdir ".\output\test_roi_pool_spatial_scale_0.5"
@echo ROI_POOL_SS_05
@cd ..
@python dnn_compiler.py configs\test\imx681_test_roi_pool_2_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_roi_pool_spatial_scale_0.5* .\output\test_roi_pool_spatial_scale_0.5
@python bin_to_bmc_format.py ^
  output\test_roi_pool_spatial_scale_0.5\test_roi_pool_spatial_scale_0.5_system_memory.bin ^
  output\test_roi_pool_spatial_scale_0.5\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_roi_pool_spatial_scale_0.5\test_roi_pool_spatial_scale_0.5_dnn_memory.bin ^
  output\test_roi_pool_spatial_scale_0.5\BMC_PRELOAD.txt

@mkdir ".\output\test_roi_pool_spatial_scale_2"
@echo ROI_POOL_SS_2
@cd ..
@python dnn_compiler.py configs\test\imx681_test_roi_pool_3_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_roi_pool_spatial_scale_2* .\output\test_roi_pool_spatial_scale_2
@python bin_to_bmc_format.py ^
  output\test_roi_pool_spatial_scale_2\test_roi_pool_spatial_scale_2_system_memory.bin ^
  output\test_roi_pool_spatial_scale_2\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_roi_pool_spatial_scale_2\test_roi_pool_spatial_scale_2_dnn_memory.bin ^
  output\test_roi_pool_spatial_scale_2\BMC_PRELOAD.txt


@mkdir ".\output\test_generate_proposals_1"
@echo GENERATE_PROPOSALS_1
@cd ..
@python dnn_compiler.py configs\test\imx681_test_generate_proposals_1_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_generate_proposals_1* .\output\test_generate_proposals_1
@python bin_to_bmc_format.py ^
  output\test_generate_proposals_1\test_generate_proposals_1_system_memory.bin ^
  output\test_generate_proposals_1\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_generate_proposals_1\test_generate_proposals_1_dnn_memory.bin ^
  output\test_generate_proposals_1\BMC_PRELOAD.txt

@mkdir ".\output\test_generate_proposals_2"
@echo GENERATE_PROPOSALS_2
@cd ..
@python dnn_compiler.py configs\test\imx681_test_generate_proposals_2_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_generate_proposals_2* .\output\test_generate_proposals_2
@python bin_to_bmc_format.py ^
  output\test_generate_proposals_2\test_generate_proposals_2_system_memory.bin ^
  output\test_generate_proposals_2\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_generate_proposals_2\test_generate_proposals_2_dnn_memory.bin ^
  output\test_generate_proposals_2\BMC_PRELOAD.txt

@mkdir ".\output\test_generate_proposals_3"
@echo GENERATE_PROPOSALS_3
@cd ..
@python dnn_compiler.py configs\test\imx681_test_generate_proposals_3_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
@cd tools
@copy ..\output\test_generate_proposals_3* .\output\test_generate_proposals_3
@python bin_to_bmc_format.py ^
  output\test_generate_proposals_3\test_generate_proposals_3_system_memory.bin ^
  output\test_generate_proposals_3\SYS_PRELOAD.txt
@python bin_to_bmc_format.py ^
  output\test_generate_proposals_3\test_generate_proposals_3_dnn_memory.bin ^
  output\test_generate_proposals_3\BMC_PRELOAD.txt

@Rem Generate files for all remaining images
@for /l %%x in (0, 1, 18) do @(
  if exist ..\configs\test\imx681_test_postproc_threshold_%%x_sim.cfg (
    @mkdir ".\output\test_postproc_threshold_%%x"
    @echo TEST_POSTPROC_THRESHOLD_%%x
    @cd ..
    @python dnn_compiler.py configs\test\imx681_test_postproc_threshold_%%x_sim.cfg FORCE_PATCHES=["170940_177827"] SENSOR_VERSION=es1
    @cd tools
    @copy ..\output\test_postproc_threshold_%%x_* .\output\test_postproc_threshold_%%x
    @python bin_to_bmc_format.py ^
      output\test_postproc_threshold_%%x\test_postproc_threshold_%%x_system_memory.bin ^
      output\test_postproc_threshold_%%x\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      output\test_postproc_threshold_%%x\test_postproc_threshold_%%x_dnn_memory.bin ^
      output\test_postproc_threshold_%%x\BMC_PRELOAD.txt
  )
)

@endlocal
@exit /b
