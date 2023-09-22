@setlocal enabledelayedexpansion

@Rem Total number of test nets to generate files for
@set num_images=11

@Rem Delete existing output directory, then create a new one
@if exist ".\output" @RD /S /Q ".\output"
@mkdir ".\output"

@Rem generate both big endian and little endian versions of each
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
    @mkdir "!dest_dir!\test_pytorch_net_%%x"
    @echo TEST_PYTORCH_NET_%%x %%e
    @cd ..\sample_code\layer_parameter_tests
    @python load_and_compile_%%x.py %%e
    @cd ..\..\tools
    @copy ..\output\test_pytorch_summary.txt !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_summary.txt
    @copy ..\output\test_pytorch_dnn_memory.bin !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_dnn_memory.bin
    @copy ..\output\test_pytorch_system_memory.bin !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_system_memory.bin
    @copy ..\output\test_pytorch_registers.txt !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_registers.txt
    @copy ..\output\test_pytorch_inputs.txt !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_inputs.txt
    @copy ..\output\test_pytorch_outputs.txt !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_outputs.txt

    @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
    if %%e NEQ LE (
      @python bin_to_bmc_format.py ^
        !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_system_memory.bin ^
        !dest_dir!\test_pytorch_net_%%x\SYS_PRELOAD.txt
      @python bin_to_bmc_format.py ^
        !dest_dir!\test_pytorch_net_%%x\test_pytorch_net_%%x_dnn_memory.bin ^
        !dest_dir!\test_pytorch_net_%%x\BMC_PRELOAD.txt
    )
  )

  @mkdir "!dest_dir!\test_pytorch_fd"
  @echo TEST_PYTORCH_FD %%e
  @cd ..\sample_code\pytorch_rom_object_detect
  @python load_and_compile.py %%e
  @cd ..\..\tools
  @copy ..\output\test_pytorch_fd* !dest_dir!\test_pytorch_fd

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_pytorch_fd\test_pytorch_fd_system_memory.bin ^
      !dest_dir!\test_pytorch_fd\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_pytorch_fd\test_pytorch_fd_dnn_memory.bin ^
      !dest_dir!\test_pytorch_fd\BMC_PRELOAD.txt
  )

  @mkdir "!dest_dir!\test_pytorch_classification"
  @echo TEST_PYTORCH_CLASSIFICATION %%e
  @cd ..\sample_code\test_classification
  @python load_and_compile.py %%e
  @cd ..\..\tools
  @copy ..\output\test_pytorch_classification* !dest_dir!\test_pytorch_classification

  @Rem FW Simulation, indicated as LE build, doesn't use PRELOAD files so skip generating
  if %%e NEQ LE (
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_pytorch_classification\test_pytorch_classification_system_memory.bin ^
      !dest_dir!\test_pytorch_classification\SYS_PRELOAD.txt
    @python bin_to_bmc_format.py ^
      !dest_dir!\test_pytorch_classification\test_pytorch_classification_dnn_memory.bin ^
      !dest_dir!\test_pytorch_classification\BMC_PRELOAD.txt
  )
)
