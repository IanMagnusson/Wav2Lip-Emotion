import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

def diferentiable_greyscale(X):
    """
    Inspired by https://kornia.readthedocs.io/en/latest/_modules/kornia/color/gray.html#rgb_to_grayscale
    :param X: A tensor ([channels, height, width]) of a cropped face image in color
    :return: A tensor ([channels, height, width]) of a cropped face image in greyscale
    """
    rgb_weights = torch.tensor([0.299, 0.587, 0.114]).to(X.device, X.dtype)
    rgb_weights = rgb_weights.view(-1, 1, 1)    # A tensor ([channels, height, width])
    X_grey = torch.sum(X * rgb_weights, dim=-3) # A tensor ([height, width])
    X_grey_3c = torch.stack([X_grey] * 3, dim=-3)
    return X_grey_3c

CUDA_VISIBLE_DEVICES=1

class AffectObjective(nn.Module):
    EMOTION_DICT = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }
    INPUT_SIZE = 224

    def __init__(self, pretrain_path, desired_affect):
        super(AffectObjective, self).__init__()

        assert desired_affect in self.EMOTION_DICT
        self.pretrain_path = pretrain_path
        self.desired_affect = desired_affect

        self.model = models.densenet121()
        num_ftrs = self.model.classifier.in_features
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda(1)

        self.model.classifier = nn.Linear(num_ftrs, len(self.EMOTION_DICT))

        self.model.load_state_dict(torch.load(pretrain_path, map_location=device)['net'])

        # like on syncnet we only need gradients wrt the inputs not wrt the parameters so I think this saves compute
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, X):
        """
        :param X: A tensor ([channels, height, width]) of a cropped face image
        :return: A tensor ([]) of the desired class likelihood of the image
        """

        X_grey = diferentiable_greyscale(X)                # X_transformed ([batch X temporal, channels, height, width])
        X_resized = F.interpolate(X_grey, self.INPUT_SIZE)      # todo experiment with interp modes
        logits = self.model(X_resized)                          # logits ([batch X temporal, classes])
        likelihoods = F.softmax(logits.squeeze(0), dim=-1)      # likelihoods ([classes])
        desired_likelihoods = likelihoods[...,self.desired_affect]  # desired_likelihoods ([])

        return desired_likelihoods



