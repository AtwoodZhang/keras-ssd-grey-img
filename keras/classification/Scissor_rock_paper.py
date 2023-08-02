import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息；
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
from keras.preprocessing.image import ImageDataGenerator


"""step1. prepare dataset
"""
data_path = r"/home/zhangyouan/桌面/zya/dataset/681/srp/RockSecissorsPaper_enlarge/"
train_data_path = data_path + "train"
data_dir = pathlib.Path(train_data_path)
image_count = len(list(data_dir.glob('*/*.bmp')))
# print(image_count)

rock_dir = os.path.join(train_data_path+r"/rock/")
paper_dir = os.path.join(train_data_path+r"/paper/")
scissor_dir = os.path.join(train_data_path+r"/scissors/")

rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissor_files = os.listdir(scissor_dir)
# print(rock_files[:3])

pic_index = 2
next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index - 2:pic_index]]
next_scissors = [os.path.join(scissor_dir, fname) for fname in scissor_files[pic_index - 2:pic_index]]

# display
for i, img_path in enumerate(next_rock+next_paper+next_scissors):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    
  
    
"""step2. create dataset generator.
"""
training_datagen = ImageDataGenerator(
    # 数据增强
    rescale=1. / 255,
    rotation_range=40, # 旋转范围
    width_shift_range=0.2, # 宽平移
    height_shift_range=0.2,# 高平移
    shear_range=0.2, # 剪切
    zoom_range=0.2, # 缩放
    horizontal_flip=True,
    fill_mode='nearest'    
)
TRAINING_DIR = 




