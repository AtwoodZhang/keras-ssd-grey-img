#!/bin/sh

# Total number of test nets to generate files for
test_nets=11

# Delete existing output directory, then create a new one
output_dir="../output"
if [ -d $output_dir ]; then rm -Rf $output_dir; fi
mkdir $output_dir

# Generate files for all remaining images
num=1
last_net=$test_nets - $num
last_net=$((test_nets - num))
echo $last_net
for x in $(seq 0 $last_net)
    do 
        mkdir "../output/test_pytorch_net_$x"
        echo TEST_PYTORCH_NET_$x
        cd ../sample_code/layer_parameter_tests
        python3 load_and_compile_$x.py
        cd ../../tools
        cp "../output/test_pytorch_summary.txt" "../output/test_pytorch_net_"$x"/test_pytorch_net_$x""_summary.txt"
        cp "../output/test_pytorch_dnn_memory.bin" "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_dnn_memory.bin"
        cp "../output/test_pytorch_system_memory.bin" "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_system_memory.bin"
        cp "../output/test_pytorch_registers.txt" "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_registers.txt"
        cp "../output/test_pytorch_inputs.txt" "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_inputs.txt"
        cp "../output/test_pytorch_outputs.txt" "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_outputs.txt"
        python3 bin_to_bmc_format.py "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_system_memory.bin" \
        "../output/test_pytorch_net_"$x"/SYS_PRELOAD.txt"
        python3 bin_to_bmc_format.py "../output/test_pytorch_net_"$x"/test_pytorch_net_"$x"_dnn_memory.bin" \
        "../output/test_pytorch_net_"$x"/BMC_PRELOAD.txt"
    done

    
mkdir "../output/test_pytorch_fd"
echo TEST_PYTORCH_FD
cd ../sample_code/pytorch_rom_object_detect
python3 load_and_compile.py
cd ../../tools
cp ../output/test_pytorch_fd* ../output/test_pytorch_fd/
python3 bin_to_bmc_format.py "../output/test_pytorch_fd/test_pytorch_fd_system_memory.bin" \
"../output/test_pytorch_fd/SYS_PRELOAD.txt"
python3 bin_to_bmc_format.py "../output/test_pytorch_fd/test_pytorch_fd_dnn_memory.bin" \
"../output/test_pytorch_fd/BMC_PRELOAD.txt"

mkdir "../output/test_pytorch_classification"
echo TEST_PYTORCH_CLASSIFICATION
cd ../sample_code/test_classification
python3 load_and_compile.py
cd ../../tools
cp ../output/test_pytorch_classification* ../output/test_pytorch_classification/
python3 bin_to_bmc_format.py "../output/test_pytorch_classification/test_pytorch_classification_system_memory.bin" \
"../output/test_pytorch_classification/SYS_PRELOAD.txt"
python3 bin_to_bmc_format.py "../output/test_pytorch_classification/test_pytorch_classification_dnn_memory.bin" \
"../output/test_pytorch_classification/BMC_PRELOAD.txt"