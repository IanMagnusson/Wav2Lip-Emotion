import torch
import torch.nn as nn
from torchvision import models

def affect_net():
    model_ft = models.densenet121()
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, len(EMOTION_DICT))
    input_size = 224
    model_ft.load_state_dict(torch.load("./densenet121_rot30_2019Nov11_14.23")['net'])
    return model_ft
