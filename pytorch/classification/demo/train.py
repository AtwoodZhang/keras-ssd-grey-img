import sys
sys.path.append('./')
import argparse
import os
import logging
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import SampleModel
from utils import prepare_model


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Sample Model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workders', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()
    return args


def train(loader, model, criterion, optimizer, epoch, device, debug_step=100):
    model.train()
    running_loss = 0.0
    

def main():
    logging.basicConfig(filename="./train.log", level=logging.INFO)  # log to file
    logging.getLogger().addHandler(logging.StreamHandler())  # also print to console output
    args = get_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if args.use_cuda and torch.cuda.is_available():
        logging.info("Using CUDA")
        
    # Data Transform
    train_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(), 
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # Datasets
    train_data = datasets.CIFAR10("/home/zhangyouan/桌面/zya/dataset/cifar10/train/", train=True, transform=train_transform, download=True)
    val_data = datasets.CIFAR10("/home/zhangyouan/桌面/zya/dataset/cifar10/test/", train=False, transform=val_transform, download=True)
    
    # Training Data Loading
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = args.batch_size,
        shuffle = True,
    )
    # Validation Data Loading
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size = args.batch_size, 
        shuffle = False,
    )
    # Create model
    model = SampleModel(
        num_classes = args.num_classes,
        input_size = (args.height, args.width),
        quantize = False,
        mode = 'train',
    )
    model.init()
    model = prepare_model(model)  # add quantization
    logging.info("Model created")
    if args.use_cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids = [DEVICE])
        model.to(DEVICE)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda and torch.cuda.is_available():
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    for epoch in range(args.num_epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, DEVICE)
    
if __name__ == "__main__":
    main()