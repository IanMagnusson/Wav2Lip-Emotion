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


def process_video_file(vfile, args, gpu_id):
	print('process_video_file', vfile, args, gpu_id)
	video_stream = cv2.VideoCapture(vfile)
	
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)

	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	maskdir = path.join(args.preprocessed_root, dirname, f'{vidname}-m')
	os.makedirs(fulldir, exist_ok=True)
	os.makedirs(maskdir, exist_ok=True)

	batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
	i = -1
	for fb in batches:
		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
		dlib.cuda.set_device(gpu_id)
		try:
			masks = set_face_zero(fb)
		except IndexError:
			masks = None
			if os.path.exists(args.bad_video_filelist):
				mode = 'a'
			else:
				mode = 'w'
			with open(args.bad_video_filelist, mode) as fout:
				fout.write(vfile + '\n')
		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue
			x1, y1, x2, y2 = f
			cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
			if masks:
				cv2.imwrite(path.join(maskdir, '{}.jpg'.format(i)), cv2.cvtColor(masks[j][y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

def set_face_zero(img):
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

	return img_masked

def main(args):
	img = cv2.imread(args.input)

	img_masked = set_face_zero(img)
	
	cv2.imwrite(args.output, cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
	main(args)
