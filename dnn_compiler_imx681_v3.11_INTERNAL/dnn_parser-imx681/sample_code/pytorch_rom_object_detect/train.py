import os
import logging
import sys
import itertools
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from model.ssd import MatchPrior
from model.sony_mobilenet_ssdlite import create_sony_mobilenet_ssdlite
from model.config import sony_mobilenet_ssd_config
from datasets.custom_images import *
from model.multibox_loss import MultiboxLoss

sys.path.append("../../")
import dnn_compiler

logging.basicConfig(filename='train.log', level=logging.INFO) # log to file
logging.getLogger().addHandler(logging.StreamHandler()) # also print to console output

# Hard coded training parameters for initial testing.
net_name = 'sony-mb2-ssd-lite'
dataset_path = './datasets/train/'
validation_dataset = './datasets/validation/'
checkpoint_folder = './saved_models/checkpoint_folder/face_hand_person/'
annotation_format = 'yolo'
label_file = './saved_models/face-hand-person-model-labels.txt'
batch_size = 8
num_workers = 0
mb2_width_mult = 1.0
base_net_lr = None
lr = 0.01
extra_layers_lr = None
resume = None
base_net = None
use_cuda = False
momentum = 0.9
weight_decay = 5e-4
milestones = "80,100"
gamma = 0.1
num_epochs = 3 # just a few epochs for debugging
validation_epochs = 10 
debug_steps = 1
t_max = 200
quantize = True

# Set up GPU if available, otherwise use CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
if use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

# Prepare model for quantization by fusing conv/bn/relu layers.
# This is specifically hard-coded for the Sony-MobileNet-SSD model, and it will break for any other model.
def prepare_model(model):
    model.train() # set up model in training mode. this does not actually perform training.
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model.eval()
    model = torch.quantization.fuse_modules( # fuse conv/bn/relu layers
        model,
        [   # list of layers to fuse together (e.g. conv -> batchnorm -> relu)
            ['base_net.in_conv.conv', 'base_net.in_conv.bn', 'base_net.in_conv.relu'],
            *[
                *[[f'base_net.inv_conv_{i}.conv.conv1', 
                   f'base_net.inv_conv_{i}.conv.bn1', 
                   f'base_net.inv_conv_{i}.conv.relu1'] for i in range(1,22)],
                *[[f'base_net.inv_conv_{i}.conv.conv2', 
                   f'base_net.inv_conv_{i}.conv.bn2'] for i in [1,2,3,4,5,7,8,9,10,11,12,14,15,16,20,21]],
                *[[f'base_net.inv_conv_{i}.conv.conv2',
                   f'base_net.inv_conv_{i}.conv.bn2',
                   f'base_net.inv_conv_{i}.conv.relu2'] for i in [6,13,17,18,19]],
                *[[f'base_net.inv_conv_{i}.conv.conv3',
                   f'base_net.inv_conv_{i}.conv.bn3'] for i in [6,13,17,18,19]]
            ],
            *[
                *[[f'classification_headers.{i}.conv1',
                   f'classification_headers.{i}.bn1',
                   f'classification_headers.{i}.relu1'] for i in range(0,5)],
                *[[f'classification_headers.{i}.conv2',
                   f'classification_headers.{i}.bn2'] for i in range(0,5)],
                *[[f'regression_headers.{i}.conv1',
                   f'regression_headers.{i}.bn1',
                   f'regression_headers.{i}.relu1'] for i in range(0,5)],
                *[[f'regression_headers.{i}.conv2',
                   f'regression_headers.{i}.bn2'] for i in range(0,5)]
            ]
        ]
    )
    model.train()
    model = torch.quantization.prepare_qat(model) # convert fp32 model to quantize-aware model
    return model


def train(loader, model, criterion, optimizer, device, debug_steps=100, epoch=-1):
    model.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes) 
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            

def test(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        with torch.no_grad():
            confidence, locations = model(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    num = max(num, 1)
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()
    # set up
    config = sony_mobilenet_ssd_config

    # object classes
    class_names = tuple([name.strip() for name in open(label_file).readlines()])

    # data transform
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    
    # prep training data
    logging.info("Prepare training datasets.")
    datasets = []
    dataset = CustomImagesDataset(dataset_path, transform=train_transform,
                 target_transform=target_transform, is_yolo = True, class_names= class_names, is_gray=True)
    datasets.append(dataset)
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True, drop_last = True)
    num_classes = len(dataset.class_names)
    print("Num of classes: {}".format(num_classes))
    print("Num of training examples: {}".format(len(dataset.ids)))
    logging.info(train_dataset)
    
    # prep validation data
    logging.info("Prepare Validation datasets.")
    if annotation_format == 'voc':        
        valid_dataset = CustomImagesDataset(validation_dataset, transform=test_transform,
                         target_transform=target_transform, is_yolo=False, class_names= class_names, is_gray=True)
    elif annotation_format == 'yolo':
        valid_dataset = CustomImagesDataset(validation_dataset, transform=test_transform,
                         target_transform=target_transform, is_yolo=True, class_names= class_names, is_gray=True)
    else:
        raise ValueError(f"Annotation Format {annotation_format} is not supported.")
    logging.info(valid_dataset)
    logging.info("validation dataset size: {}".format(len(valid_dataset)))
    val_loader = DataLoader(valid_dataset, batch_size,
                            num_workers= num_workers,
                            shuffle=False, drop_last=True)
    
    # construct model
    logging.info("Build network.")
    model = create_sony_mobilenet_ssdlite(num_classes, width_mult=mb2_width_mult, quantize=quantize)
    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = base_net_lr if base_net_lr is not None else lr
    extra_layers_lr = extra_layers_lr if extra_layers_lr is not None else lr
    params = [ {'params': model.base_net.parameters(), 'lr': base_net_lr},
               {'params': itertools.chain(
                          model.source_layer_add_ons.parameters(),
                          model.extras.parameters()
                          ), 'lr': extra_layers_lr},
               {'params': itertools.chain(
                          model.regression_headers.parameters(),
                          model.classification_headers.parameters()
                          )}]
    timer.start("Load Model")
    model.init()
    if quantize:
        model = prepare_model(model) # add quantization
#    print(model.state_dict())
    
    # set up optimizer
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    if use_cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids = [DEVICE])
    model.to(DEVICE)    
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    logging.info(f"Learning rate: {lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")
    logging.info("Uses CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)
    
    # training loop
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, num_epochs):
        print(f"Starting epoch: {epoch}")
        scheduler.step()
        train(train_loader, model, criterion, optimizer,
              device=DEVICE, debug_steps=debug_steps, epoch=epoch)
#        print(model.state_dict())
        # perform validation every N epochs
        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, model, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
        # save intermediate checkpoints
        model_path = os.path.join(checkpoint_folder, f"Epoch-{epoch}-Loss{val_loss}.pth")
        if use_cuda and torch.cuda.is_available():
            model.module.save(model_path)
        else:
            model.save(model_path)
        logging.info(f"Saved model {model_path}")

    # final conversion to quantized model and save
    model_cpu = model.to('cpu') # move model to cpu if it is on gpu
    model_cpu.eval() # Set up model in eval mode.
    print('Converting the model (post-training)...')
    model_cpu = torch.quantization.convert(model_cpu) # apply quantization conversion
    print('Quantization done.')

    # save model
    print('Writing quantized model to disk.')
    if use_cuda and torch.cuda.is_available():
        model_out = model_cpu.module
    else:
        model_out = model_cpu
    torch.save(model_out.state_dict(), './model_quantized.pth')
#    print(model_out.state_dict())

    # Parse
    dnn_compiler.run("../../configs/test/imx681_test_pytorch_i2c.cfg", model_out)
