# keras-ssd-grey-img
将整体目录改为如下：
![image](https://user-images.githubusercontent.com/30472220/220815172-a5a41192-42b3-4198-8688-32b32579fb40.png)
数据集目录：
  VOCdevkit
     |---  VOC2007
              |------Annotations
              |------ImageSets
              |------JPEGImages
              |------cmp_img.ipynb
              |------split_train_test.ipynb
通过split_train_test.ipynb将数据集JPEGImages划分
通过voc_annotations.py生成2007_train.txt和2007_val.txt
