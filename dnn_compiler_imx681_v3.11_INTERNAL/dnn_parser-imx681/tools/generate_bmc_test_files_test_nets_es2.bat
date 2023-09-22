@setlocal enabledelayedexpansion

@Rem Total number of test nets to generate files for
@set num_images=11

@Rem Delete existing output directory, then create a new one
@if exist ".\output" @RD /S /Q ".\output"
@mkdir ".\output"

@Rem generate both big endiand and little endian versions of each
@for %%e in (BE LE) do @(
  set "dest_dir=.\output\%%e"
  @mkdir !dest_dir!

  if %%e == BE (
      set "endianness=OUTPUT_ENDIANNESS=big"
  ) else (
      set "endianness=OUTPUT_ENDIANNESS=little"
  )

  @Rem Generate files for all remaining images
  @set /A last_image = %num_images% - 1
  @for /l %%x in (0, 1, !last_image!) do @(
    @mkdir "!dest_dir!\test_net_%%x"
    @echo TEST_NET_%%x %%e
    @cd ..
    @python dnn_compiler.py configs\test\imx681_test_net_%%x_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
    @cd tools
    @copy ..\output\test_net_%%x_* !dest_dir!\test_net_%%x

    @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
    if %%e NEQ LE (
      @python bin_to_bmc_format.py ^
        !dest_dir!\test_net_%%x\test_net_%%x_system_memory.bin ^
        !dest_dir!\test_net_%%x\SYS_PRELOAD.txt
      @python bin_to_bmc_format.py ^
        !dest_dir!\test_net_%%x\test_net_%%x_dnn_memory.bin ^
        !dest_dir!\test_net_%%x\BMC_PRELOAD.txt
    )
  )

  @echo TEST_POSTPROC %%e
  @Rem test_postproc names overlap, it can copy data from earlier passes
  @Rem We remove them from the compiler output directoy to avoid that.
  @DEL /Q "..\output\test_postproc_threshold*"

  @mkdir "!dest_dir!\test_postproc"
    
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_postproc_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_postproc_* .\!dest_dir!\test_postproc

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_postproc\test_postproc_system_memory.bin ^
      !dest_dir!\test_postproc\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_postproc\test_postproc_dnn_memory.bin ^
      !dest_dir!\test_postproc\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\dnn_fd"
  @echo DETECTION %%e
  @cd ..
  @python dnn_compiler.py configs\imx681_tflite_detection_es1_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\dnn_fd* .\!dest_dir!\dnn_fd

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\dnn_fd\dnn_fd_system_memory.bin ^
      !dest_dir!\dnn_fd\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\dnn_fd\dnn_fd_dnn_memory.bin ^
      !dest_dir!\dnn_fd\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\dnn_hd2"
  @echo HUMAN DETECT %%e
  @cd ..
  @python dnn_compiler.py configs\imx681_tflite_detection_es2_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\dnn_hd2* .\!dest_dir!\dnn_hd2

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\dnn_hd2\dnn_hd2_system_memory.bin ^
      !dest_dir!\dnn_hd2\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\dnn_hd2\dnn_hd2_dnn_memory.bin ^
      !dest_dir!\dnn_hd2\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\dnn_classification"
  @echo CLASSIFICATION %%e
  @cd ..
  @python dnn_compiler.py configs\imx681_tflite_classification_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\sony_classification_summary.txt .\!dest_dir!\dnn_classification\dnn_classification_summary.txt
  @copy ..\output\sony_classification_system_memory.bin .\!dest_dir!\dnn_classification\dnn_classification_system_memory.bin
  @copy ..\output\sony_classification_dnn_memory.bin .\!dest_dir!\dnn_classification\dnn_classification_dnn_memory.bin

  @copy ..\output\sony_classification_inputs.txt .\!dest_dir!\dnn_classification\dnn_classification_inputs.txt
  @copy ..\output\sony_classification_outputs.txt .\!dest_dir!\dnn_classification\dnn_classification_outputs.txt
  @copy ..\output\sony_classification_registers.txt .\!dest_dir!\dnn_classification\dnn_classification_registers.txt
  @copy ..\output\sony_classification_load_sequence_i2c.bin .\!dest_dir!\dnn_classification\dnn_classification_load_sequence_i2c.bin

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\dnn_classification\dnn_classification_system_memory.bin ^
      !dest_dir!\dnn_classification\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\dnn_classification\dnn_classification_dnn_memory.bin ^
      !dest_dir!\dnn_classification\BMC_PRELOAD.txt
  )

  @Rem ROI test names overlap and the ROI_POOL copy will copy out of date
  @REM files from earlier passes.
  @DEL /Q "..\output\test_roi_pool*"

  @mkdir "!dest_dir!\test_roi_pool"
  @echo ROI_POOL %%e
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_roi_pool_1_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_roi_pool* .\!dest_dir!\test_roi_pool

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_roi_pool\test_roi_pool_system_memory.bin ^
      !dest_dir!\test_roi_pool\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_roi_pool\test_roi_pool_dnn_memory.bin ^
      !dest_dir!\test_roi_pool\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\test_roi_pool_spatial_scale_0.5"
  @echo ROI_POOL_SS_05 %%e
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_roi_pool_2_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_roi_pool_spatial_scale_0.5* .\!dest_dir!\test_roi_pool_spatial_scale_0.5

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_roi_pool_spatial_scale_0.5\test_roi_pool_spatial_scale_0.5_system_memory.bin ^
      !dest_dir!\test_roi_pool_spatial_scale_0.5\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_roi_pool_spatial_scale_0.5\test_roi_pool_spatial_scale_0.5_dnn_memory.bin ^
      !dest_dir!\test_roi_pool_spatial_scale_0.5\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\test_roi_pool_spatial_scale_2"
  @echo ROI_POOL_SS_2 %%e
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_roi_pool_3_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_roi_pool_spatial_scale_2* .\!dest_dir!\test_roi_pool_spatial_scale_2

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_roi_pool_spatial_scale_2\test_roi_pool_spatial_scale_2_system_memory.bin ^
      !dest_dir!\test_roi_pool_spatial_scale_2\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_roi_pool_spatial_scale_2\test_roi_pool_spatial_scale_2_dnn_memory.bin ^
      !dest_dir!\test_roi_pool_spatial_scale_2\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\test_generate_proposals_1"
  @echo GENERATE_PROPOSALS_1 %%e
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_generate_proposals_1_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_generate_proposals_1* .\!dest_dir!\test_generate_proposals_1

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_generate_proposals_1\test_generate_proposals_1_system_memory.bin ^
      !dest_dir!\test_generate_proposals_1\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_generate_proposals_1\test_generate_proposals_1_dnn_memory.bin ^
      !dest_dir!\test_generate_proposals_1\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\test_generate_proposals_2"
  @echo GENERATE_PROPOSALS_2 %%e
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_generate_proposals_2_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_generate_proposals_2* .\!dest_dir!\test_generate_proposals_2

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_generate_proposals_2\test_generate_proposals_2_system_memory.bin ^
      !dest_dir!\test_generate_proposals_2\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_generate_proposals_2\test_generate_proposals_2_dnn_memory.bin ^
      !dest_dir!\test_generate_proposals_2\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\test_generate_proposals_3"
  @echo GENERATE_PROPOSALS_3 %%e
  @cd ..
  @python dnn_compiler.py configs\test\imx681_test_generate_proposals_3_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
  @cd tools
  @copy ..\output\test_generate_proposals_3* .\!dest_dir!\test_generate_proposals_3

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
        !dest_dir!\test_generate_proposals_3\test_generate_proposals_3_system_memory.bin ^
        !dest_dir!\test_generate_proposals_3\SYS_PRELOAD.txt
      @python bin_to_bmc_format.py ^
        !dest_dir!\test_generate_proposals_3\test_generate_proposals_3_dnn_memory.bin ^
        !dest_dir!\test_generate_proposals_3\BMC_PRELOAD.txt
  )

  @Rem Generate files for all remaining images
  @for /l %%x in (0, 1, 18) do @(
    if exist ..\configs\test\imx681_test_postproc_threshold_%%x_sim.cfg (
      @mkdir "!dest_dir!\test_postproc_threshold_%%x"
      @echo TEST_POSTPROC_THRESHOLD_%%x %%e
      @cd ..
      @python dnn_compiler.py configs\test\imx681_test_postproc_threshold_%%x_sim.cfg !endianness! FORCE_PATCHES=["189270"] SENSOR_VERSION=es2
      @cd tools
      @copy ..\output\test_postproc_threshold_%%x_* .\!dest_dir!\test_postproc_threshold_%%x

      @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
      if %%e NEQ LE (
        @python bin_to_bmc_format.py ^
          !dest_dir!\test_postproc_threshold_%%x\test_postproc_threshold_%%x_system_memory.bin ^
          !dest_dir!\test_postproc_threshold_%%x\SYS_PRELOAD.txt
        @python bin_to_bmc_format.py ^
          !dest_dir!\test_postproc_threshold_%%x\test_postproc_threshold_%%x_dnn_memory.bin ^
          !dest_dir!\test_postproc_threshold_%%x\BMC_PRELOAD.txt
      )
    )
  )
)

@endlocal
@exit /b
