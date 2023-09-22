## Sample code to quantize a floating point Pytorch DNN model using FX Graph Mode Quantization and run the Sony DNN Compiler 

This sample code includes a script "quantize.py" to quantize a FP32 lightweight classification model using PyTorch's FX Graph Mode Quantization. A validation set of CIFAR10 dataset will be downloaded to ./cifar10/test/ the first time the script is run, if it has not been downloaded already to calibrate the quantization parameters.

The script "quantize.py" loads the FP32 weights into the model from this file and demonstrates how to call the Sony DNN Compiler to generate the binary files for IMX681.

### To quantize a floating point model using FX Graph Mode Quantization and run Sony DNN Compiler:
```
python quantize.py /path/to/fp32_quantized_model.pth --batch_size 32 --num_classes 10
```
