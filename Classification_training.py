import os
import cv2
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import matplotlib.image as mpimg
from torchvision import transforms
from torchvision import datasets
from torchvision import transforms, models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights


def data_loader(train_data_path, test_data_path, batchsize):
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=train_data_path, transform=transform_img)
    train_data_loader = data.DataLoader(train_data, batch_size=batchsize, shuffle=True)

    test_data = datasets.ImageFolder(root=test_data_path, transform=transform_img)
    test_data_loader = data.DataLoader(test_data, batch_size=batchsize, shuffle=True)

    return train_data_loader, test_data_loader

def evaluate_accuracy(data_loader, model, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    return accuracy


def calculate_color_richness(img_np):
    unique_colors = np.unique(img_np.reshape(-1, img_np.shape[2]), axis=0)
    richness = len(unique_colors)
    return richness


def calculate_color_richness_gpu(img_tensor):
    B, C, H, W = img_tensor.size()
    richness = torch.zeros(B, device=img_tensor.device)

    for i in range(C):
        img_channel = img_tensor[:, i, :, :] * 255
        for b in range(B):
            hist = torch.histc(img_channel[b].view(-1), bins=256, min=0, max=255)
            richness[b] += hist[hist > 0].numel()

    richness /= C
    return richness


class HueAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(HueAttentionModule, self).__init__()
        mid_channels = max(in_channels // 2, 1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class RichnessAttentionModule(nn.Module):
    def __init__(self):
        super(RichnessAttentionModule, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(x)
        return attention


def rgb_to_hsv(input):
    device = input.device
    input_np = input.cpu().numpy().transpose(0, 2, 3, 1)
    hsv_images = []
    for rgb_image in input_np:
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        hsv_images.append(hsv_image[:, :, 0])
    hsv_images = np.stack(hsv_images, axis=0)
    hsv_tensor = torch.from_numpy(hsv_images).float().unsqueeze(1)
    return hsv_tensor.to(device)


class ResNetWithHARichness(nn.Module):
    def __init__(self, num_classes=2, alpha=3.0, beta=2.0):
        super(ResNetWithHARichness, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()

        self.classifier = nn.Linear(num_ftrs + 2, num_classes)
        self.hue_attention = HueAttentionModule(in_channels=1)
        self.richness_attention = RichnessAttentionModule()


    def forward(self, x):
        h_component = rgb_to_hsv(x)
        h_attended = self.hue_attention(h_component)
        global_h_feature = torch.mean(h_attended, dim=[2, 3]).unsqueeze(1) * self.alpha
        if global_h_feature.dim() > 2:
            global_h_feature = global_h_feature.squeeze(-1)

        richness_features = calculate_color_richness_gpu(x).unsqueeze(1) * self.beta
        if richness_features.dim() > 2:
            richness_features = richness_features.squeeze(-1)

        features = self.resnet50(x)
        features = torch.cat((features, global_h_feature, richness_features), dim=1)
        x = self.classifier(features)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    device = next(model.parameters()).device
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            progress_bar.set_postfix(loss=running_loss / (batch_idx + 1),
                                     acc=running_corrects.double() / (batch_idx + 1) * inputs.size(0))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

        val_loss = 0.0
        val_corrects = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        model.train()






def main():
    batch_size = 32

    train_data_path = 'dataset8/train'
    test_data_path = 'dataset8/test'

    train_data_loader, test_data_loader = data_loader(train_data_path, test_data_path, batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetWithHARichness().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.1)
    train_model(model, train_data_loader, test_data_loader, criterion, optimizer, num_epochs=10)
    torch.save(model, 'weights/model_complete.pth')

    train_accuracy = evaluate_accuracy(train_data_loader, model, device)
    print(f'Train Accuracy: {train_accuracy:.4f}')

    test_accuracy = evaluate_accuracy(test_data_loader, model, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
