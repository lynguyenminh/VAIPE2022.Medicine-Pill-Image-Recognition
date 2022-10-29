import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from operator import itemgetter
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

# Load model efficientnetb7
def load_model_classify_efficientnetb7(weight):
    # 1. init model
    model = models.efficientnet_b7(pretrained=False)
    num_class = 107
    model.classifier = nn.Linear(2560, num_class)

    for param in model.parameters():
        param.requires_grad = False

    # 2. Load weight for model
    checkpoint = torch.load(weight, map_location=torch.device('cpu')) # if using cpu
    # checkpoint = torch.load(weight) # if using gpu
    model.load_state_dict(checkpoint['model_state_dict'])

    return model




# predict data da chuan bi phia tren
def predict_dataloader(model, dataloaders, class_names,  device):
    '''
    This function is using to predict pills which are predicted into class 107 by YOLO.
    '''
    phase = 'crop_107_class'
    model.eval()

    predict_list = []
    print('Start using classifier re-predict pill in class 107...')
    for inputs, _ in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        predict_list += [class_names[i] for i in list(preds.cpu().detach().numpy())]
    return predict_list


def predict_img(weight, img, CLASS_NAMES):
    # Create the model
    model = models.efficientnet_b7(pretrained=False)
    num_ftrs = 2560
    num_class = len(CLASS_NAMES)
    model.classifier = nn.Linear(num_ftrs, num_class)

    # Load state_dict
    checkpoint = torch.load(weight, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create the preprocessing transformation here
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # convert cv2 to PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Apply transform
    img = transform(img)

    img = img.reshape((1, 3, 224, 224))
    model.eval()
    output = model(img).cpu().detach().numpy().argmax(axis=1)[0]

    return CLASS_NAMES[output]