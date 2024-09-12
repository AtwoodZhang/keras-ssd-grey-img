import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# 定义图像转换步骤
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((120, 160)),  # 调整大小为 160x120
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图，输出通道数为1
        transforms.ToTensor(),  # 转换为 PyTorch 张量
        transforms.Normalize([0.5], [0.5])  # 对灰度图进行标准化，均值和标准差为0.5
    ]),
    'val': transforms.Compose([
        transforms.Resize((120, 160)),  # 调整大小为 160x120
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图，输出通道数为1
        transforms.ToTensor(),  # 转换为 PyTorch 张量
        transforms.Normalize([0.5], [0.5])  # 对灰度图进行标准化，均值和标准差为0.5
    ])
}

# 数据集目录
data_dir = r"/home/zhangyouan/桌面/zya/dataset/681/PCScreen_Book_PhoneScreen"

# 加载数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 设备配置 (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Classes: {class_names}")

import torch.nn.functional as F

class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=2, stride=1):
        super(MV2Block, self).__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expansion_factor
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)

        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)

        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu6(self.expand_bn(self.expand_conv(x)))
        out = F.relu6(self.depthwise_bn(self.depthwise_conv(out)))
        out = self.project_bn(self.project_conv(out))
        if self.use_residual:
            out = x + out
        return out

class MobileNetV2Style(nn.Module):
    def __init__(self):
        super(MobileNetV2Style, self).__init__()
        # 输入层
        self.input_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)  # 单通道输入 (灰度图)
        self.input_bn = nn.BatchNorm2d(3)
        
        # MobileNetV2 架构
        self.mv2_block1 = MV2Block(3, 16, stride=1)
        self.mv2_block2 = MV2Block(16, 32, stride=2)
        self.mv2_block3 = MV2Block(32, 32, stride=1)
        self.mv2_block4 = MV2Block(32, 32, stride=2)
        self.mv2_block5 = MV2Block(32, 16, stride=1)
        self.mv2_block6 = MV2Block(16, 16, stride=1)
        
        # 池化层和全连接层
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 15 * 20, 64)  # 输入尺寸根据池化后的大小调整
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        x = self.mv2_block1(x)
        x = self.mv2_block2(x)
        x = self.mv2_block3(x)
        x = self.mv2_block4(x)
        x = self.mv2_block5(x)
        x = self.mv2_block6(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x

# 创建模型实例并将其移到设备
model = MobileNetV2Style().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个 epoch 都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和正确的预测数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

# 保存模型的状态字典
torch.save(model.state_dict(), 'model_test_batch16.pth')
print("模型已保存到 'model.pth'")
