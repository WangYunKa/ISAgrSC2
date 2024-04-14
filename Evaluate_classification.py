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


def random_classification_validation(model, data_path, transform, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(17, 10))
    class_names = ["field", "nonfield"]
    image_files = os.listdir(data_path)

    for i in range(num_images):
        index = random.randint(0, len(image_files) - 1)
        image_path = os.path.join(data_path, image_files[index])
        image = Image.open(image_path)
        img = mpimg.imread(image_path)

        image_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            pred_class = class_names[output.argmax().item()]

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {pred_class}", color='blue')
        plt.axis('off')
    plt.show()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('weights/model_complete.pth', map_location=device)
    model.to(device)
    model.eval()

    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_data_path = "dataset8/val"
    random_classification_validation(model, val_data_path, transform_img, num_images=5)

if __name__ == '__main__':
    main()
