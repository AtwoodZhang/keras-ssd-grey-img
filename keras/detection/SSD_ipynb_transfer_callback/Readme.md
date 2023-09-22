## Mobilenet-SSD：in Keras.


## Top News
**`2023-09`**:**build up the evaluation part and predict part**  

**`2022-12`**:**build the Mobilenet SSD structure, train the model.**   


# Name

## SSD_ipynb_transfer_callback

# Description

The project was build to explore the DNN tensorflow model building in IMX681.

SSD_ipynb_transfer_callback
 |-- Anchors.py     # to get the anchor;
 |-- callbacks.py   # evaluation part; 
 |-- Datasets.py    # Datasets, Dataloader;
 |-- get_anchors_no_normal.py    # get anchors according to Anchors.py, it will generate a anchor.txt file.
 |-- loss.py        # class loss between predict and GT;
 |-- Models.py      # build the model of SSD and Mobilenet Structure.
 |-- predict.ipynb  # inference according to the input image and output inferred image and bounding box. (write the predict script through jupyter)
 |-- predict.py     # inference according to the input image and output inferred image and bounding box. (write the predict script through py file)
 |-- SSD_Mobile_Det_good.ipynb  # Main function, you can train the project through this script.
 |-- ssd_pred.py    # the function used by predict.py, belongs to the part of the process of predict.
 |-- train.py       # same as SSD_Mobile_Det_good.ipynb, train script.
 |-- utils_bbox.py  # the function used by predict.py, belongs to the part of the process of predict.
 |-- utils_map.py   # mAP 
 |-- utils.py       # functions.


# Visuals
I cannot attach here, but you can see the file pred_offline.gif from the root path.

# Installation

pip install -r requirements.txt

# Usage

    * train:
    run SSD_Mobile_Det_good.ipynb, the output windows will shows following information. it is the same as train.py

        class_names: ['good'] num_classes: 2
        type: <class 'numpy.ndarray'> shape: (1242, 4)
        Train on 1603 samples, val on 179 samples, with batch size 32.
        Configurations:
        ----------------------------------------------------------------------
        |                     keys |                                   values|
        ----------------------------------------------------------------------
        |             classes_path |             ./model_data/voc_classes.txt|
        |               model_path |                                         |
        |              input_shape |                               [120, 160]|
        |                    Epoch |                                      150|
        |               batch_size |                                       32|
        |                       lr |                                    0.001|
        |           optimizer_type |                                     Adam|
        |                 momentum |                                    0.937|
        |                num_train |                                     1603|
        |                  num_val |                                      179|
        ----------------------------------------------------------------------
        Epoch 1/150
    
    * predict
    run predict.ipynb

    # get anchors.txt, same format as IMX681 requreiment.

# Support

email: youan.zhang@sony.com

# Roadmap


# Authors and acknowledgment

Author: 
1) YouanZhang build the project, train the model.
2) Yoward Hu generate the gif through computer PC.

# License

Sony internal confidential

# Project status

continue in comparing and modification.
