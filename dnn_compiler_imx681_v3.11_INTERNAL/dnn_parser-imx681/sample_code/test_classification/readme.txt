ROM test classification model for DNN compiler testing.
One difference with the TFLite version is that SUB layer is replaced with ADD layer in Pytorch, 
since Pytorch quantization with FloatFunctional() does not support SUB.

Train:
$ python train.py

Parse:
$ python load_and_compile.py

