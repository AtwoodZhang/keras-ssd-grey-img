#-----------------------------------------------------------
# Retrieve a list of list result on training and test data
# set for each training epoch
#-----------------------------------------------------------
import matplotlib.pyplot as plt


def visual_train(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc)) # Get number of epochs
    
    #-----------------------------------------------------------
    # Plot training and validation accuracy per epoch
    #-----------------------------------------------------------
    plt.plot(epochs, acc, 'r', label = "tra_acc")
    plt.plot(epochs ,val_acc, 'b', label = "val_acc")
    plt.title("training and validation accuracy")
    plt.legend(loc=0)
    plt.grid(ls='--')  # 生成网格
    plt.show()
    # 曲线呈直线是因为epochs/轮次太少
    #-----------------------------------------------------------
    # Plot training and validation loss per epoch
    #-----------------------------------------------------------
    plt.plot(epochs, loss, 'r', label = "train_loss")
    plt.plot(epochs ,val_loss, 'b', label = "val_loss")
    plt.title("training and validation loss")
    plt.legend(loc=0)
    plt.grid(ls='--')  # 生成网格
    plt.show()
    # 曲线呈直线是因为epochs/轮次太少
    


# 1. 获取类
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
