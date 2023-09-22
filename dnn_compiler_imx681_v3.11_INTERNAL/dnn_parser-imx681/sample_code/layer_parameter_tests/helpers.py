import yaml
import torch
import os
from dataset import ImageNetDataset 
from torch.utils.data import DataLoader
from torch.quantization import QConfig, FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
import torchvision.transforms as transforms
import pdb

def load_config(config_file) -> dict:
    with open(config_file, "r", encoding="utf-8") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
    return cfg

def prepare_data(config, batch_size):

    transform = transforms.Compose(
    [ transforms.ToTensor(),
        transforms.Resize((config["model"]["input_height"], config["model"]["input_width"])),
      transforms.Grayscale(num_output_channels=1),
           transforms.Normalize((0.5), (0.5))])


    train_dataset = ImageNetDataset(config=config, batch_size=batch_size, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
    return train_loader

def save_model(config, prepared_model,use_cuda):
    model_cpu = prepared_model.to('cpu') # move model to cpu if it is on gpu
    model_cpu.eval()

 
    quant_model = torch.quantization.convert(model_cpu)
    quant_path = os.path.join(config["training"]["save_quantized_root_path"], config["name"])

    if not os.path.exists(quant_path):
        os.makedirs(quant_path)


    quant_path = os.path.join(quant_path, "model_quantized.pth")

    if use_cuda and torch.cuda.is_available():
        quant_model = quant_model.module
    else:
        quant_model = quant_model
    torch.save(quant_model.state_dict(), quant_path)
    print(quant_model.state_dict())

    torch.save(quant_model.state_dict(), quant_path)
    print('Quantization done.')

# Custom INT8 quantization configuration (per-tensor)
def quant_config_per_tensor():
    qconfig = QConfig(activation=FakeQuantize.with_args(
                        observer=MovingAverageMinMaxObserver,
                        quant_min=0,
                        quant_max=255,
                        reduce_range=True),
                      weight=FakeQuantize.with_args(
                        observer=MovingAverageMinMaxObserver,
                        quant_min=-128,
                        quant_max=127,
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric,
                        reduce_range=True))
    return qconfig

# Custom INT8 quantization configuration (per-channel)
def quant_config_per_channel():
    qconfig = QConfig(activation=FakeQuantize.with_args(
                        observer=MovingAverageMinMaxObserver,
                        quant_min=0,
                        quant_max=255,
                        reduce_range=True),
                      weight=FakeQuantize.with_args(
                        observer=MovingAveragePerChannelMinMaxObserver,
                        quant_min=-128,
                        quant_max=127,
                        dtype=torch.qint8,
                        qscheme=torch.per_channel_symmetric,
                        reduce_range=True,
                        ch_axis=0))
    return qconfig



def prepare_model(model):
    # pdb.set_trace()
    model.train() # set up model in training mode. this does not actually perform training.
    model.qconfig = quant_config_per_channel() # set up all layers with per-channel quantization
    # for m in model.children(): # change any FC layers to use per-tensor quantization
    for m in model.modules(): # change any FC layers to use per-tensor quantization
        if isinstance(m, torch.nn.Linear):
            m.qconfig = quant_config_per_tensor()
    model.eval()
    model = torch.quantization.fuse_modules(model, model.all_layer_conv_names)   # fuse conv/bn/relu layers
    model.train()
    model = torch.quantization.prepare_qat(model) # convert fp32 model to quantize-aware model
    return model 

def model_import(name, config, debug=False):
    if name == "layer_parameter_test_0":
        from layer_test_models.model_0 import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model

    elif name == "layer_parameter_test_1":
        from layer_test_models.model_1  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_2":
        from layer_test_models.model_2  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_3":
        from layer_test_models.model_3  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_4":
        from layer_test_models.model_4  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_5":
        from layer_test_models.model_5  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model

    elif name == "layer_parameter_test_6":
        from layer_test_models.model_6  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_7":
        from layer_test_models.model_7  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_8":
        from layer_test_models.model_8  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
    
    elif name == "layer_parameter_test_9":
        from layer_test_models.model_9  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model

    elif name == "layer_parameter_test_10":
        from layer_test_models.model_10  import build_model
        # Create Model    
        base_model  = build_model(config, debug)
        return base_model
