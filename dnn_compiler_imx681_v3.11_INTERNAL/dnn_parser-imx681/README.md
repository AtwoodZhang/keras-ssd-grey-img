# DNN Compiler

The DNN Compiler is a python-based tool developed by the Sony ISDC firmware team
to convert a DNN from a Tensorflow TFLite File (.tflite) or PyTorch code
to auto-generated C source files and/or .bin files targeted for deployment on 
Sony image sensors (IMX681 or similar).

## Dependencies

The DNN Compiler as the following dependencies, and has been tested with the 
versions listed:

| Package                | Version      | Install Command                         |
| ---------------------- | ------------ | --------------------------------------- |
| python                 | 3.6.7        |                                         |
| flatbuffers            | 1.12         |  `pip install flatbuffers`              |
| numpy                  | 1.18.5       |  `pip install numpy`                    |
| pytorch                | 1.10, 1.12   |  `pip install torch torchvision`        | 
| OpenCV                 | 4.5.4        |  `pip install opencv-python`            |

The compiler has only been tested in a Windows and Linux command prompt.

| OS                     | Version      | 
| ---------------------- | ------------ | 
| Windows                | 10           |  
| Ubuntu                 | 20.04        |

If you intend to build a release of the DNN compiler, you will also need to
install the following:

| Package                | Version   | Install Command                         |
| ---------------------- | --------- | --------------------------------------- |
| PyArmor                | 6.7.2     |  `pip install pyarmor`                  |
| GitPython              | 3.1.18    |  `pip install GitPython`                |


## Usage


The DNN compiler's usage is different whether it is being using on a DNN that
originated from Tensorflow or a DNN that originated from Pytorch.

### Tensorflow

For Tensorflow, a pre-trained DNN is saved to a .tflite file. Then, the DNN 
compiler is run as a standalone tool as follows:

```
python dnn_compiler.py [config-file]
```

[config-file] is the path to a DNN Compiler configuration file that defines file
paths, output modes, memory locations, and other runtime parameters for the
compiler. Examples are included in the `configs` directory. 

For Tensorflow the `DNN_MODEL` configuration parameter specifies the path to the
.tflite file to load data from.

For complete usage and command line options, run the following:

```
python dnn_compiler.py --help
```

### Pytorch

For Pytorch, the DNN compiler is run inline in the python code that creates the
PyTorch model as a function call:

```
import dnn_compiler
dnn_compiler.run([config-file], [model-object])
```

[config-file] is the path to a DNN Compiler configuration file that defines file
paths, output modes, memory locations, and other runtime parameters for the
compiler. Examples are included in the `configs` directory. 

[model-object] is the object that represents the DNN model.

For sample programs that create a DNN and call the DNN compiler, see the
`sample_code` directory.

## Output Files

When the compiler runs, it will always generate the following file:

* `[dnn_name]_summary.txt`: a human readable file describing the size and memory
  location of each data structure, register settings, layers in the DNN, 
  estimated processing time, and more.

Based on the output mode, it will also generate one of the following sets of files:

OUTPUT_MODE = rom
* `dnn_common_auto.h`: header containing auto-generated constants and datatypes.
* `dnn_op_func_table_auto.c`: source file defining a table that maps operation IDs
  to function pointers.
* `[dnn_name]_optable_auto.c`: source file defining the optable
* `[dnn_name]_misc_data_auto.c`: source file defining the misc data array
* `[dnn_name]_quant_params_auto.c`: source file defining the quantization
  parameter array

OUTPUT_MODE = i2c
* `[dnn_name]_load_sequence_i2c.bin`: binary file containing a series of I2C writes
  to load the DNN and configure the sensor to use it
* `[dnn_name]_dnn_memory.bin`: binary file containing weights/biases and any other
  data structures that need to be loaded into DNN memory
* `[dnn_name]_system_memory.bin`: binary file containing the optable and any other
  data structures that need to be loaded into system RAM
* `[dnn_name]_registers.txt`: Text file containing register values and other 
  information needed by the DNN Simulator to run this DNN.
  
## Flat Buffer Setup

The DNN compiler using the FlatBuffers open source library from google to read the
.tflite file: <https://google.github.io/flatbuffers>

In order to run the DNN compiler, this library needs to be installed using the 
following command:

```
pip install flatbuffers
```

From a development point of view, there was some one-time setup that had to be 
done to prepare our repo to use flatbuffers. If we need to upgrade to a new
version of tensorflow in the future, this procedure may need to be repeated:
*  Install FlatBuffers for Windows: <https://google.github.io/flatbuffers/>
*  Clone the tensorflow git repo from github: <https://github.com/tensorflow/tensorflow>
*  Checkout the branch associated with the new version of tensorflow
*  Get the following file from the repo:  `tensorflow/lite/schema/schema.fbs`
*  From a command line, run the following command: `flatc.exe -python schema.fbs`
*  This will auto-generate a directory of python files called tflite. Copy this folder to `.\tflite`

## Release Procedure

To build an external release of the DNN compiler, run the following script:

```
python build_release.py [version-number]
```

This will do the following:
1. Prompt you to enter a version number
2. Prompt whether or not to automatically check in to git and create a tag
3. Modify `.\internal\constants.py`
  * Update the value of `DNN_COMPILER_VERSION`
     * Either set to `[version-number]`, or increment by 0.1 if `[version-number]` is omitted
  * Make sure that `DNN_COMPILER_INTERNAL_USE` is set to `False`
4. Use PyArmor to obfuscate source code
5. Package up the obfuscated source code and non-source files (e.g. configs, docs, data)
6. Create a final release archive called: `dnn_compiler_v[version-number].zip`
7. Optionally check in `constants.py` to git and create a tag for this release.

## Tools

A bin2def tool is provided in the `tools\bin2def` directory. This tool converts
DNN compiler output (I2C load sequence or memory images) to .def files that can be
used to load the DNN via SSP-300.

## Further Documentation

For more information on the design of the DNN Compiler, see: `doc\dnn_compiler_design.docx`
