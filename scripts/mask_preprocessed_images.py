import sys
from os import listdir, path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp
from imutils import face_utils
import dlib
import skimage
from skimage.draw import polygon
import scipy
from scipy.spatial import ConvexHull
import numpy as np

import face_detection


MODEL_PATH = "/u/ianmag/emotion-synthesis/shape_predictor_81_face_landmarks.dat"

parser = argparse.ArgumentParser()

parser.add_argument("--input", help="image file (.jpg) containing already cropped face to mask", required=True)
parser.add_argument("--output", help="output path for masked image file", required=True)

args = parser.parse_args()

def mask_img(infile, outfile):
        img = cv2.imread(infile)
        predictor = dlib.shape_predictor(MODEL_PATH)
        height, width, channels = img.shape
        sp = predictor(img, dlib.rectangle(0, 0, width -1,  height - 1))
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        mask_outline_landmarks = landmarks[[*range(17), 78, 74, 79, 73, 72, 80, 71, 70, 69, 68, 76, 75, 77]]
        Y, X = skimage.draw.polygon(mask_outline_landmarks[:,1], mask_outline_landmarks[:,0])
        Y[Y >= img.shape[0]] = img.shape[0] - 1
        X[X >= img.shape[1]] = img.shape[1] - 1
        cropped_img = np.zeros(img.shape, dtype=np.uint8)
        cropped_img[Y, X] = img[Y, X]
        final = cv2.subtract(img, cropped_img)
        img_masked = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        cv2.imwrite(outfile, cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))

mask_img(args.input, args.output)
