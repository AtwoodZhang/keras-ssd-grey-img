# ------------------------------------------------------------------------------
# Copyright 2022 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------

import argparse
import os
import logging
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ROMClassification
from utils import prepare_model

def get_args():
  parser = argparse.ArgumentParser(description='PyTorch Test Model')
#  parser.add_argument('--train_dir', type=str, default='./dataset/train/')
#  parser.add_argument('--val_dir', type=str, default='./dataset/validation/')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--num_workers', type=int, default=1)
  parser.add_argument('--num_epochs', type=int, default=10)
  parser.add_argument('--learning_rate', type=float, default=0.01)
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
  for batch_idx, data in enumerate(loader):
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    logits, predictions = model(images) # forward pass
    loss = criterion(logits, labels) # loss calculation
    loss.backward() # calculate gradients
    optimizer.step()
    running_loss += loss.item()
    if batch_idx % debug_step == 0:
      avg_loss = running_loss / debug_step
      logging.info(
        f"Epoch: {epoch}, Step: {batch_idx}, " +
        f"Average Loss: {avg_loss:.4f}")
      running_loss = 0.0
  return avg_loss

def val(loader, model, criterion, device):
  model.eval()
  running_loss = 0.0
  num_iter = 1
  for _, data in enumerate(loader):
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
      logits, predictions = model(images)
      loss = criterion(logits, labels)
    running_loss += loss.item()
    num_iter += 1
  return running_loss / num_iter

def main():
  logging.basicConfig(filename='train.log', level=logging.INFO) # log to file
  logging.getLogger().addHandler(logging.StreamHandler()) # also print to console output
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
  train_data = datasets.CIFAR10("./cifar10/train/", train=True, transform=train_transform, download=True)
  val_data = datasets.CIFAR10("./cifar10/test/", train=False, transform=val_transform, download=True)
  
  # Training Data Loading
  train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = args.batch_size,
    shuffle = True)

  # Validation Data Loading
  val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size = args.batch_size,
    shuffle = False)

  # Create model
  model = ROMClassification(num_classes=args.num_classes,
                            input_size=(args.height, args.width),
                            quantize=True)
  model.init()
  model = prepare_model(model) # add quantization
  print(model)
  if args.use_cuda and torch.cuda.is_available():
    model = torch.nn.DataParallel(model, device_ids=[DEVICE])
    model.to(DEVICE)
  
  # Loss function
  criterion = torch.nn.CrossEntropyLoss()
  if args.use_cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
  optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

  # Training loop
  for epoch in range(args.num_epochs):
    train_loss = train(train_loader, model, criterion, optimizer, epoch, DEVICE)
    val_loss = val(val_loader, model, criterion, DEVICE)
    logging.info(
      f"Epoch: {epoch}, " + 
      f"Training Loss: {train_loss:.4f}, " +
      f"Validation Loss: {val_loss:.4f}, ")

  # Save model
  model_cpu = model.to('cpu') # move model to cpu if it is on gpu
  model_cpu.eval()
  print('Converting the model (post-training)...')
  model_cpu = torch.quantization.convert(model_cpu)
  print('Quantization done.')
  if args.use_cuda and torch.cuda.is_available():
    model_out = model_cpu.module
  else:
    model_out = model_cpu
  torch.save(model_out.state_dict(), './model_quantized.pth')
  print(model_out.state_dict())


if __name__ == "__main__":
  main()





