
# IMX681 DNN Simulator
The IMX681 DNN Simulator (simulator) runs a DNN compiled for the IMX681 image sensor (sensor) in software to allow testing purely in software.

The simulator is a stand-alone program built from pieces of the IMX681 firmware and a software model of the IMX681 calculating hardware. It runs the compiled DNN on the test images and writes the sensors output, including GPO bits and IBI data, as a comma separated list.

The simulator's output is a reliable representation of the output you would get from the same DNN model running in in the sensor hardware.

This document refers to release version 1.01 of the DNN Simulator.

## Usage
The simulator is provided as a command line application, with versions for both Windows and Linux x86.

Invoke it with a reference to the output directory of the DNN Compiler (compiler), the name of the dnn, and one or more images to test.
  ```
  > imx681 -d dnn_compiler\output dnn_fd_sim images\test_0.pgm images\test_1.pgm
  ```

The as of version 1.1 the simulator can launch child processes to process images
in parallel.

The output is a Comma Separated list (CSV) of results. Results for each tested image on a separate line. 

Example results:
  ```
  "Image", "GPO 1", "GPO2", "GPO3", "IBI Data..."
  "images\test_0.pgm", 1, 0, 0, 2, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  "images\test_1.pgm", 0, 0, 0, 0
  ```
The first value on the line is the file that that was tested.

The next three values represent the hardware GPO pins, 1 indicating set, 0 indicating not set. The DNN sets these pins as configured in the compiler configuration files.

The remaining values are the data in the IBI data stream sent to the host processor. The DNN sets these bytes as configured in the compiler configuration files. The number of bytes of IBI data can vary, the simulator only writes the relevant data.
	
# Detailed Instructions
This section includes detailed instructions and discussion of each of the steps of simulating a DNN.

## Building for Simulation
### Design
The simulator requires features in the DNN compiler that were added in version 3.08. Earlier versions of the compiler do not provide complete information for the simulator and the simulator will produce an error message on trying to load the DNN image.

The simulator runs in the host computer's memory endianness, which is Little Endian for all currently released versions, while the sensor uses Big Endian memory format. The compiler builds a memory image of the compiled DNN so it must account for the endianness of the run environment. This means that the simulator cannot use exactly the same output files that the compiler produces for i2c control of the sensor.

The compiler and simulator can distinguish DNNs by name. This allows you to keep the two versions of the DNN model in same output directory.

### Process
Memory endianness is controlled by the OUTPUT_ENDIANNESS config parameter which is specified in the sensor config file.

The DNN simulator is built from a version of the FW with all patches pre-loaded. The DNN Compiler output must be built for a system with patches loaded in order to be compatible. To ensure that this is true, the FORCE_PATCHES=[170940_177827] config parameter must be supplied to the compiler.

It's also convenient to use a different DNN_NAME when compiling for simulation which is typically specified in the DNN config file. This allows both sets of output to be kept in the same compiler output directory. The alternatives are to only ever keep one DNN model or to separate the sensor and simulation models OUTPUT_DIRECTORY.

There are two ways to override these parameters that don't require you to modify the base config files. Inside config files the last read value for a parameter is the value that’s used. This allows you to make a top level config file that uses the same base config files but overrides the relevant parameters.

Example simulation config file, excerpt:
  ```
  # import base config files
  INC_CONFIG base\sensor\imx681.inc
  INC_CONFIG base\dnn\sony_example.inc

  # override name, endianness, and patches for simulation
  DNN_NAME dnn_fd_sim
  OUTPUT_ENDIANNESS little
  FORCE_PATCHES=[170940_177827]
  ```

Parameters can also be overridden at run time. The runtime override value is used no matter to value read from the config files. The format for this override is \<PARAM\>=\<value\>.

TensorFlow models override on the command-line:
  ```
  > python dnn_compiler.py configs/sony_example.cfg DNN_NAME=dnn_fd_sim OUTPUT_ENDIANNESS=little FORCE_PATCHES=[170940_177827]
  ```

PyTorch models override at the call to dnn_compiler.run: 
  ```
  dnn_compiler.run("../../configs/sony_example.cfg", model, config_overrides=["DNN_NAME=dnn_fd_sim", "OUTPUT_ENDIANNESS=little", "FORCE_PATCHES=[170940_177827]"])
  ```

## Test Image Details
### Design
Internally to the sensor the frame to process is provided to the DNN as a 160x120 8-bit per-pixel greyscale image. 

The simulator accepts a limited number of image formats, specifically:
* *.pgm and *.png files
* Images must be at the target resolution of 160x120. The simulator will not scale images
* Test images *may* be color. The simulator will convert them to gray-scale using an integer version of the BT.601 luminance calculation: Y'=0.299R'+0.587G'+0.114B'

### Preparing Images
Test images can be prepared by any process that makes them match the requirements above.

Sony provides an Image Converter tool that can batch process images to meet the requirements. This tool is documented separately.

### Providing Images to Test
Files to test are provided on the command line as one or more of: file name, directory containing files to process, or a file containing a list of files and directories to process.

See "Running The Simulation" below for specific details on providing these value on the command line.

## Running the Simulation
### Design
To run a simulation, the simulator requires:
* Compiled DNN artifacts
* Name of the DNN
* One or more image files to test
	
The compiled DNN artifacts are the files produced by the DNN Compiler. These files are placed in the directory specified by the compiler's `OUTPUT_DIRECTORY` parameter.

The DNN Name is a friendly name representing the DNN. It is specified by the compilers `DNN_NAME` parameter.

Image files to test are the filename of an image, the name of a directory containing image files to test, or -f<file> indicating a file containing a list of files or directories to process.

The simplest possible simulation run assumes that the compiler artifacts and image to test are in the working directory:
  ```
  > imx681 dnn_fd_sim test_0.pgm
  ```
### Details
The`--help` command line option produces an overview of the command line options. Each parameter and option will be discussed in detail below.
  ```
  > imx681.exe --help
  Usage: imx681.exe [OPTIONS] <dnn_name> <image specification>+
  Simulate DNN with image file or files.
    <dnn_name> is name as provided to DNN compiler's DNN_NAME config entry.

    Images for testing must be monochromatic, no alpha channel, PNG images.
    Images may be specified as one or more of:
      <file>     a single image file
      <dir>      a directory containing one or more image files
      -f <file>  a file containing a list of files to process, once file path per line.

  Options:
    -n <N>             Parallel simulation with N workers.
                       Spawns N additional child processes and spreads work through them.
                       (Default) When N = 1 all processing is in a single executable.
    -o[output file]    Direct output to file
                          With no additional filename it writes output to 
                            <dnn_name>_output_<date & time>.csv
                          (Default) When [output file] is "-" output is written to stdout
                          Otherwise this writes to the filename provided.
    -d <directory>     Directory containing binary DNN image files from DNN Compiler.
                          This defaults to the current directory.

    --help             This help information.
    --version          Simulator version information.
  ```
	
#### \<dnn_name\>
The DNN Name is a friendly name for the DNN to run. It is used by the simulator to choose which files to load when there is more than one compiled DNN in the compiler out directory.

The name of a DNN is provided to the compiler in its `DNN_NAME` parameter.

This is required on the command line and must be the first non-optional parameter on the command line.

#### \<image specification\>+
Images are specified on the command line. Every non-optional parameter after the `<dnn_name>` is treated as an image specifier. One image specifier is required, more may be provided. Specifier types may be mixed freely on the command line.

An image specifier is one of:
* filename of an image
* the name of a directory containing image files
* file containing a list of files and directories to process, indicated by `-f<file>`

Image are added to an internal list of images to process in the order they are encountered. Image specifiers are processed left to right on the command line and each specifier is fully expanding before processing the next specifier.

Specifiers are tested to ensure they reference a file system object. If it exists it is added to the list of images to process. Files that are not image files or not appropriate for simulation for other reasons are rejected when loading it during the simulation process.

When the specifier is a file name that that file is added to the internal list.

When the specifier is a directory all files in that directory are added to the list. Sub-directories are not recursively added to the list or processed in any way.

When the specifier is a file containing a list of files, as indicated by the `-f` flag, it is expected to be a file that contains files or directories to test, one entry per line. Files are added to the list of images to process. For directories all files in the directory are added to the list of files to process, sub-directories are not processed. There is no provision for comments.

#### [-d \<directory\>]
The simulator uses multiple files created by the DNN Compiler. It finds these files by combining the directory these files are in with the DNN_NAME that was provided on the command line.

The compiler artifact files can be moved but must be kept together. The simulator expects all relevant files to be kept in the same directory and won't be able to find them if they are separated.

The directory the simulator looks for these files defaults to the current working directory. That directory can be changed with the `-d <directory>` command line option. The directory path may be relative or absolute.

#### [-n \<NUM_WORKERS\>]
The simulator can process the list of specified images in parallel by launching child processes for a subset of the images. The number of worker processes is specified on the command line as `-n <N>`, this will spawn N children each processing a portion of the image list.

In this mode each worker is a separate process and not a thread inside the application. A similar result can be accomplished by manually starting multiple instances each with a subset of the total image list to process.

Each worker writes it's output to a short lived temp directory started in the run directory. The primary application merges these results into the requested output format. 

With one worker, without the `-n` parameter or with `-n1`, all processing happens in the single primary application just as it did in previous versions.

Note: when the output is directed to stdout and configured with 1 worker the results will be printed to stdout as they are processed. When configured with multiple workers each worker writes its output to a temporary file; results are only written to stdout by the primary instance after all the workers are finished.

#### [-o[output file]]
The output results are written to stdout by default. The `-o` command line options can be used to change that.
* `-o` with no additional parameter will cause the simulator to create a file named from the DNN_NAME, date, and time and write the results to that file
* `-o-` will write output to stdout. This is the default behavior
* `-o<filename>` will write the results to the file specified

## Interpreting Results
The results are written as comma separated values.

The first row is a header for the rest of the data.

Each additional row contain the results from a single run of the DNN on a single test image. 

Example:
  ```
  "images/test_0.pgm", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  ```

### Image Name 
The first column, "images/test_0.pgm," in the example, is the name of the image that was tested.

### GPO State
The next 3 columns, "1, 0, 0," from the example, represent the state of the sensor's 3 GPO pins. The results of a DNN run can set these pins to indicate certain results. 1 indicates the GPO is set, 0 indicates unset.

The specifics of how the GPO state is set based on the results is configured in the DNN Compiler config files. Documentation about configuring this is included with the DNN Compiler.

### IBI Results
The remaining columns are the DNN results data that is sent from the sensor as IBI data. The specifics of what this data indicates are how these results are calculated are configured in the DNN Compiler config files. Documentation about this configuration is included with the DNN Compiler.

### Error Case
In a case where there is a problem with the image file an error is reported in that images row. 

Example: 
  ```
  "images/test_0.jpg", "Error - image load failed"
  ```

The name of the image being tested is the first column.

The second column is an indication of an error and an minimal error message.
	
# Known Issues
Error messages for an image in the DNN results are not very descriptive.

# Command Line Examples
## File Specifiers
Examples of various file specifiers:
  ```
  > imx681 -d dnn_compiler/output dnn_fd_sim images/test_0.jpg images/test_0.pgm
  "Image", "GPO 1", "GPO2", "GPO3", "IBI Data..."
  "images/test_0.jpg", "Error - image load failed"
  "images/test_0.pgm", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  ```


  ```
  > ls -R images
  'images':
  is_dir  test_0.jpg  test_0.pgm  test_0.png  test_1.pgm  wrong_size.jpg
  	
  'images/is_dir':
  foo.txt  test_1.pgm

  > imx681 -d dnn_compiler/output dnn_fd_sim images
  "Image", "GPO 1", "GPO2", "GPO3", "IBI Data..."
  "images/test_0.jpg", "Error - image load failed"
  "images/test_0.pgm", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  "images/test_0.png", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  "images/test_1.pgm", 0, 0, 0
  "images/wrong_size.jpg", "Error - image load failed"
  ```

  ```
  > cat file.lst
  # every line is treated as a file, so no comments allowed
  images/is_dir
  images/test_0.pgm
  images/test_1.pgm

  > imx681 -d dnn_compiler/output dnn_fd_sim -f file.lst
  Unable to access # every line is treated as a file, so no comments allowed
  "Image", "GPO 1", "GPO2", "GPO3", "IBI Data..."
  "images/is_dir/foo.txt", "Error - image load failed"
  "images/is_dir/test_1.pgm", 0, 0, 0
  "images/test_0.pgm", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  "images/test_1.pgm", 0, 0, 0
  ```

  ```
  > echo mixed specifications are fine
  mixed specifications are fine

  > imx681 -d dnn_compiler/output dnn_fd_sim -f file.lst images/is_dir images/test_0.png
  Unable to access # every line is treated as a file, so no comments allowed
  "Image", "GPO 1", "GPO2", "GPO3", "IBI Data..."
  "images/is_dir/foo.txt", "Error - image load failed"
  "images/is_dir/test_1.pgm", 0, 0, 0
  "images/test_0.pgm", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  "images/test_1.pgm", 0, 0, 0
  "images/is_dir/foo.txt", "Error - image load failed"
  "images/is_dir/test_1.pgm", 0, 0, 0
  "images/test_0.png", 1, 0, 0, 3, 2, 14, 1, 106, 79, 63, 1, 71, 58, 24, 23, 63, 0, 61, 0, 20, 26, 52
  ```

# Versions
* 1.0 - First releases version
* 1.01 - Add parallel processing of images on the command line. 