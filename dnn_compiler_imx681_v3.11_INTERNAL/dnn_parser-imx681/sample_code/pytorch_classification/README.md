## Sample code to convert a Pytorch DNN model with the Sony DNN Compiler 

This sample code includes a script "train.py" to train a lightweight classification model (with INT8 quantization) on the CIFAR10 dataset. The CIFAR10 dataset will be downloaded to ./cifar10/ the first time the script is run, if it has not been downloaded already.

After the model is trained, it will be saved to a file as a `state_dict` (https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html#save-and-load-the-model-via-state-dict). The script "load_and_compile.py" loads the model from this file and demonstrates how to call the Sony DNN Compiler to generate the binary files for IMX681.

### To train the model from scratch:
```
python train.py --batch_size 32 --num_epochs 10 --learning_rate 0.001 --use_cuda
```

### To run the Sony DNN Compiler on a trained model:
```
python load_and_compile.py
```

