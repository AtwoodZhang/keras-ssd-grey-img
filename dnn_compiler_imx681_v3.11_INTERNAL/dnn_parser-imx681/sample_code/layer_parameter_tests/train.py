import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pdb
import yaml
import argparse
from helpers import load_config, save_model, prepare_data, prepare_model, model_import
import os



def train_net(config, use_cuda, DEVICE):
    batch_size = config["training"]["batch_size"]
    trainloader = prepare_data(config, batch_size)
    

    base_model = model_import(config['name'], config)

  
    prepared_model = prepare_model(base_model)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(prepared_model.parameters(), lr=0.001, momentum=0.9)

    if use_cuda and torch.cuda.is_available():
        prepared_model = torch.nn.DataParallel(prepared_model, device_ids=[DEVICE])
        prepared_model.to(DEVICE)
        criterion.cuda()

    epochs = config["training"]["epochs"]
    for epoch in range(epochs):  # loop over the dataset multiple times
        prepared_model.train()
        running_loss = 0.0 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['image'], data['classes']
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
            optimizer.zero_grad()

            # Prep the model for quantization
            outputs = prepared_model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0


    print('Finished Training')
    save_model(config, prepared_model, use_cuda)
    
    
  

if __name__ == "__main__":

    layer_parser = argparse.ArgumentParser(description='Layer Parameter Tests')
    layer_parser.add_argument('config', type=str, help='the path to config.yaml file')
    args = layer_parser.parse_args()
    
    
    config = load_config(args.config)
    use_cuda = config["training"]["use_cuda"]
    device = config["training"]["devices"]
  

    DEVICE = torch.device("cuda:"+str(device) if torch.cuda.is_available() and use_cuda else "cpu")


    train_net(config, use_cuda, DEVICE)

    