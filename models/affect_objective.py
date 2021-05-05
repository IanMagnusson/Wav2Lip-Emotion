import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

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
        self.model.classifier = nn.Linear(num_ftrs, len(self.EMOTION_DICT))

        self.model.load_state_dict(torch.load(pretrain_path, map_location='cpu')['net']) #TODO make this adapt to the device

        self.input_transform = transforms.Compose([
            transforms.ToPILImage(mode=None), # todo confirm that mode should be none or find a way to run transforms w/o conversion
            transforms.Resize(self.INPUT_SIZE),
            transforms.CenterCrop(self.INPUT_SIZE),
            transforms.Grayscale(num_output_channels=3),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), TODO confirm if this pretrained densenet requires normalization
            transforms.ToTensor()
        ])

    def forward(self, X):
        """
        :param X: A tensor ([batch, channels, height, width]) of cropped face images
        :return: A tensor ([batch]) of the desired class likelihood for each image
        """

        X_transformed = self.input_transform(X)                     # X_transformed ([batch, channels, height, width])
        logits = self.model(X_transformed)                          # logits ([batch, classes])
        likelihoods = F.softmax(logits, dim=1)                      # likelihoods ([batch, classes])
        desired_likelihoods = likelihoods[:,self.desired_affect]    # desired_likelihoods ([batch])

        return desired_likelihoods



