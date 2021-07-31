import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

def diferentiable_greyscale(X):
    """
    Inspired by https://kornia.readthedocs.io/en/latest/_modules/kornia/color/gray.html#rgb_to_grayscale
    :param X: A tensor ([..., channels, height, width]) of a cropped face image in color
    :return: A tensor ([..., channels, height, width]) of a cropped face image in greyscale
    """
    rgb_weights = torch.tensor([0.299, 0.587, 0.114]).to(X.device, X.dtype)
    rgb_weights = rgb_weights.view(-1, 1, 1)    # A tensor ([channels, height, width])
    X_grey = torch.sum(X * rgb_weights, dim=-3) # A tensor ([height, width])
    X_grey_3c = torch.stack([X_grey] * 3, dim=-3)
    return X_grey_3c

def diferentiable_normalize(data: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Inspired by https://kornia.readthedocs.io/en/latest/_modules/kornia/augmentation/augmentation.html#Normalize
    Normalize a tensor image with mean and standard deviation.
    """
    shape = data.shape

    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std = std[..., :, None]

    data_temp = data.view(-1,shape[-3],shape[-2],shape[-1])
    data_temp = data_temp.view(data_temp.shape[0],data_temp.shape[1], -1)
    out: torch.Tensor = (data_temp - mean) / std

    return out.view(shape)

CUDA_VISIBLE_DEVICES=1

class AffectObjective(nn.Module):
    INPUT_SIZE = 224

    def __init__(self, pretrain_path, desired_affect: int, emotion_idx_to_label: dict,
                 greyscale=False, normalize=False, ):
        super(AffectObjective, self).__init__()

        self.emotion_idx_to_label = emotion_idx_to_label
        assert desired_affect in self.emotion_idx_to_label
        self.pretrain_path = pretrain_path
        self.desired_affect = desired_affect


        self.greyscale = greyscale
        self.normalize = normalize

        self.model = models.densenet121()
        num_ftrs = self.model.classifier.in_features
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.to('cuda:2')

        self.model.classifier = nn.Linear(num_ftrs, len(self.emotion_idx_to_label)).to(device)
        self.model.load_state_dict(torch.load(pretrain_path, map_location=device)['net'])

        # like on syncnet we only need gradients wrt the inputs not wrt the parameters so I think this saves compute
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, X):
        """
        :param X: A tensor ([channels, height, width]) of a cropped face image
        :return: A tensor ([]) of the desired class likelihood of the image
        """
        if self.normalize:
            X = diferentiable_normalize(X, [0.4306, 0.3199, 0.2652], [0.1722, 0.1150, 0.0941])
        if self.greyscale:
            X = diferentiable_greyscale(X)                # X_transformed ([batch X temporal, channels, height, width])
        X_resized = F.interpolate(X, self.INPUT_SIZE, mode='bilinear')
        logits = self.model(X_resized)                          # logits ([batch X temporal, classes])
        likelihoods = F.softmax(logits.squeeze(0), dim=-1)      # likelihoods ([classes])
        desired_likelihoods = likelihoods[...,self.desired_affect]  # desired_likelihoods ([])

        return desired_likelihoods



