# setup data

EMOTION_DICT = {
  0: "angry",
  1: "disgust",
  2: "fear",
  3: "happy",
  4: "sad",
  5: "surprise",
  6: "neutral",
}

import os
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset

class Fer2013(Dataset):
  def __init__(self, stage, configs):
      self._stage = stage
      self._configs = configs

      self._image_size = (configs["image_size"], configs["image_size"])

      self._data = pd.read_csv(
          os.path.join(configs["data_path"], "{}.csv".format(stage))
      )

      self._pixels = self._data["pixels"].tolist()
      self._emotions = pd.get_dummies(self._data["emotion"])

      self._transform = transforms.Compose(
          [
              transforms.ToPILImage(),
              transforms.ToTensor(),
          ]
      )

  def __len__(self):
      return len(self._pixels)

  def __getitem__(self, idx):
      pixels = self._pixels[idx]
      pixels = list(map(int, pixels.split(" ")))
      image = np.asarray(pixels).reshape(48, 48)
      image = image.astype(np.uint8)

      image = cv2.resize(image, self._image_size)
      image = np.dstack([image] * 3)

      image = self._transform(image)
      target = self._emotions.iloc[idx].idxmax()
      return image, target

# load test data

conf = {
"data_path": "./",
"image_size": 224
}
test_set = Fer2013("test", conf)
