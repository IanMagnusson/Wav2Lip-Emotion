import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

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

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()
args.ngpu = [1,2,3]

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in args.ngpu ]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

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
		masks = set_face_zero(fb)
		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue
			x1, y1, x2, y2 = f
			cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
			cv2.imwrite(path.join(maskdir, '{}.jpg'.format(i)), cv2.cvtColor(masks[j][y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

def set_face_zero(window):
	masked_face = []
	for w in window:
		p = "shape_predictor_81_face_landmarks.dat"
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(p)

		# print(f'W {w.shape}')
		rect = detector(w)[0]
		sp = predictor(w, rect)
		landmarks = np.array([[p.x, p.y] for p in sp.parts()])
		mask_outline_landmarks = landmarks[[*range(17), 78, 74, 79, 73, 72, 80, 71, 70, 69, 68, 76, 75, 77]]
		Y, X = skimage.draw.polygon(mask_outline_landmarks[:,1], mask_outline_landmarks[:,0])
		Y[Y >= w.shape[0]] = w.shape[0] - 1
		X[X >= w.shape[1]] = w.shape[1] - 1
		cropped_img = np.zeros(w.shape, dtype=np.uint8)
		cropped_img[Y, X] = w[Y, X]
		final = cv2.subtract(w, cropped_img)
		masked_face.append(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

	return masked_face

def process_audio_file(vfile, args):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')

	command = template.format(vfile, wavpath)
	subprocess.call(command, shell=True)


def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	filelist = glob(path.join(args.data_root, '*/*.mp4'))
	jobs = [(vfile, args, i % len(args.ngpu)) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(len(args.ngpu))
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	print('Dumping audios...')

	for vfile in tqdm(filelist):
		try:
			process_audio_file(vfile, args)
		except KeyboardInterrupt:
			exit(0)
		except:
			traceback.print_exc()
			continue

if __name__ == '__main__':
	main(args)
