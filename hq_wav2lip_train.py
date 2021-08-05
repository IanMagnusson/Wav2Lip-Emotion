from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import AffectObjective
from models import Wav2Lip, Wav2Lip_disc_qual
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

import datetime
from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
parser.add_argument("--dest_affect_root",
    help="Root folder of the preprocessed data for the destination affect you are trying to generate", required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)

parser.add_argument('--affect_checkpoint_path', help='Load the pre-trained affect objective', default=None, type=str)
parser.add_argument('--gpu_id', help='index of gpu to use', default=0, type=int)

args = parser.parse_args()

if not args.dest_affect_root and (hparams.gt_dest_affect or hparams.mask_dest_affect):
    parser.error('destination affect data must be provided at --dest_affect_root if using dest_affect hparams')
    
if args.affect_checkpoint_path is None and hparams.affect_wt > 0.:
    parser.error('affect_checkpoint_path cannot be used when hparams.affect_wt is set to 0')

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)
        self.affect_videos = get_image_list(args.dest_affect_root, split)
        self.image_cache = {}
        self.audio_cache = {}

    def get_img(self, fname):
        if not hparams.dataset_cache:
            return cv2.imread(fname)
        if fname in self.image_cache:
            img = self.image_cache[fname].copy()
        else:
            img = cv2.imread(fname)
            self.image_cache[fname] = img.copy()
        return img

    def get_audio(self, fname):
        if not hparams.dataset_cache:
            wav = audio.load_wav(fname, hparams.sample_rate)
            return audio.melspectrogram(wav).T
        if fname in self.audio_cache:
            orig_mel = self.audio_cache[fname].copy()
        else:
            wav = audio.load_wav(fname, hparams.sample_rate)
            orig_mel = audio.melspectrogram(wav).T
            self.audio_cache[fname] = orig_mel.copy()
        return orig_mel

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames, masked = False):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = self.get_img(join(dirname(fname) + '-m', basename(fname)) if masked else fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return self.prepare_window(window)

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # returns 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)
            target_img_name = img_name.replace(args.data_root, args.dest_affect_root) if hparams.gt_dest_affect else img_name
            mask_img_name = img_name.replace(args.data_root, args.dest_affect_root) if hparams.mask_dest_affect else img_name

            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            target_window_fnames = self.get_window(target_img_name)
            mask_window_fnames = self.get_window(mask_img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if target_window_fnames is None or  mask_window_fnames is None or wrong_window_fnames is None:
                continue
            
            target_window = self.read_window(target_window_fnames)
            wrong_window = self.read_window(wrong_window_fnames)
            if hparams.full_masked:
                masked_window = self.read_window(mask_window_fnames, masked=True)
            else:
                masked_window = self.read_window(mask_window_fnames) if (hparams.gt_dest_affect != hparams.mask_dest_affect) \
                    else target_window.copy()
                masked_window[:, :, masked_window.shape[2]//2:] = 0.
            if target_window is None or wrong_window is None or masked_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                if hparams.gt_dest_affect: wavpath = wavpath.replace(args.data_root, args.dest_affect_root)
                orig_mel = self.get_audio(wavpath)
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), target_img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), target_img_name)
            if indiv_mels is None:
                continue

            x = np.concatenate([masked_window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(target_window)
            return x, indiv_mels, mel, y

def get_grad_norm(model):
    # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2
    norm_type = 2
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), 2.0).item()

    return total_norm

def clamp_grad_norm_(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    clip_coef = hparams.disc_max_grad_norm / (total_norm + 1e-6)
    stretch_coef = hparams.disc_min_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
            for p in parameters:
                        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    elif stretch_coef > 1.0:
            for p in parameters:
                        p.grad.detach().mul_(stretch_coef.to(p.grad.device))

def save_sample_images(x, g, gt, global_step, sample_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    if not os.path.exists(sample_dir): os.mkdir(sample_dir)
    folder = join(sample_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


syncnet = SyncNet().to(device)
syncnet.eval()
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

if hparams.affect_wt:
    affect_objective = AffectObjective(args.affect_checkpoint_path, hparams.desired_affect, hparams.emotion_idx_to_label,
                                   greyscale=hparams.greyscale_affect, normalize=hparams.normalize_affect).eval()
    affect_objective.to(device)

def get_affect_loss(X):
    """
    :param X: A tensor ([batch, channels, temporal, height, width]) of cropped face images TODO figure out mystery dim
    :return: A tensor ([batch]) of the desired class logit for each image
    """

    # merge temporal and batch
    X = X.permute(0, 2, 1, 3, 4).clone().to(device)
    X = X.view(-1, *X.shape[2:])

    desired_likelihoods = affect_objective(X)   # desired_likelihoods ([batch X ???])
    affect_loss = 1 - desired_likelihoods     # affect_loss ([batch X ???])
    avg_affect_loss = affect_loss.mean()       # avg_affect_loss ([])
    return avg_affect_loss

def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        running_affect_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            if disc: disc.train()
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            ### Train generator now. Remove ALL grads.
            optimizer.zero_grad()
            if disc_optimizer: disc_optimizer.zero_grad()

            g = model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = torch.zeros(1).to(device)

            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = torch.zeros(1).to(device)

            if hparams.affect_wt > 0.:
                affect_loss = get_affect_loss(g)
            else:
                affect_loss = torch.zeros(1).to(device)

            if hparams.l1_wt > 0:
                l1loss = recon_loss(g, gt)
            else:
                l1loss = torch.zeros(1).to(device)

            loss = hparams.syncnet_wt * sync_loss + \
                   hparams.disc_wt * perceptual_loss + \
                   hparams.l1_wt * l1loss + \
                   hparams.affect_wt * affect_loss

            loss.backward()
            optimizer.step()

            if disc:
                ### Remove all gradients before Training disc
                disc_optimizer.zero_grad()

                pred = disc(gt)
                disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
                disc_real_loss.backward()

                pred = disc(g.detach())
                disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                disc_fake_loss.backward()

                if hparams.disc_max_grad_norm and hparams.disc_min_grad_norm: 
                    #torch.nn.utils.clip_grad_norm_(disc.parameters(), hparams.disc_max_grad_norm)
                    clamp_grad_norm_(disc.parameters())

                disc_optimizer.step()

                disc_grad_norm = get_grad_norm(disc)

                running_disc_real_loss += disc_real_loss.item()
                running_disc_fake_loss += disc_fake_loss.item()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, join(checkpoint_dir,'train_samples'))

            # Logs
            global_step += 1
            session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            running_sync_loss += sync_loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_affect_loss += affect_loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                if disc: save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')


            if global_step % hparams.eval_interval == 0:
                with open(join(args.checkpoint_dir,"train.log"), "a") as train_log:
                    train_log.write('###### Now at global epoch {} and global step{:09d} #####\n'.format(global_epoch, global_step))
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)

                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', hparams.syncnet_wt + hparams.syncnet_warmup_wt_increase)


            report = 'L1: {}, Sync: {}, Percep: {} Affect: {} | Fake: {}, Real: {}, norm {}'.format(
                    running_l1_loss / (step + 1),
                    running_sync_loss / (step + 1),
                    running_perceptual_loss / (step + 1),
                    running_affect_loss / (step + 1),
                    running_disc_fake_loss / (step + 1),
                    running_disc_real_loss / (step + 1),
                    disc_grad_norm
                )
            prog_bar.set_description(report)

            with open(join(args.checkpoint_dir,"train.log"), "a") as train_log:
                train_log.write(report + '\n')

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, disc):
    count = min(300, len(test_data_loader))
    print('Evaluating for {} steps'.format(count))
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss, running_affect_loss = 0., 0., 0., 0., 0., 0.

    for step, (x, indiv_mels, mel, gt) in tqdm(enumerate(test_data_loader)):
        model.eval()
        if disc: disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        g = model(indiv_mels, x)
        if step  == 0:
            save_sample_images(x, g, gt, global_step, join(checkpoint_dir,'eval_samples'))

        if disc:
            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

        sync_loss = get_sync_loss(mel, g)

        if hparams.disc_wt > 0.:
            perceptual_loss = disc.perceptual_forward(g)
        else:
            perceptual_loss = torch.zeros(1).to(device)

        if hparams.affect_wt > 0.:
            affect_loss = get_affect_loss(g)
        else:
            affect_loss = torch.zeros(1).to(device)

        if hparams.l1_wt > 0.:
            l1loss = recon_loss(g, gt)
        else:
            l1loss = torch.zeros(1).to(device)
        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                hparams.l1_wt * l1loss + \
                                hparams.affect_wt * affect_loss

        running_l1_loss += l1loss.item()
        running_sync_loss += sync_loss.item()
        running_perceptual_loss += perceptual_loss.item()
        running_affect_loss += affect_loss.item()

        if step > count: break

    report = 'L1: {}, Sync: {}, Percep: {}, Affect: {} | Fake: {}, Real: {}'.format(running_l1_loss / count,
                                                       running_sync_loss / count,
                                                       running_perceptual_loss / count,
                                                        running_affect_loss / count,
                                                        running_disc_fake_loss / count,
                                                        running_disc_real_loss / count)
    print(report)
    with open(join(args.checkpoint_dir, "eval.log"), "a") as eval_log:
        eval_log.write('Evaluating at global epoch {} and  step{:09d}\n'.format(global_epoch, global_step))
        eval_log.write(report + '\n')

    return running_sync_loss / count


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    load_log = open(join(args.checkpoint_dir, "load.log"), 'a')

    print("Load checkpoint from: {}".format(path))
    load_log.write("Load checkpoint from: {}\n".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            load_log.write("Load optimizer state from {}\n".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    load_log.close()
    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)


    # Model
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if hparams.disc_wt:
        disc = Wav2Lip_disc_qual(hparams.disc_bn, hparams.disc_residual).to(device)
        disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))
        print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))
    else:
        disc = None
        disc_optimizer = None

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    with open(join(args.checkpoint_dir,"train.log"), "a") as train_log:
        train_log.write(f'###############  Starting new run at {datetime.datetime.now()} #################\n')
    with open(join(args.checkpoint_dir,"eval.log"), "a") as eval_log:
        eval_log.write(f'###############  Starting new run at {datetime.datetime.now()} #################\n')
    with open(join(args.checkpoint_dir,"load.log"), "a") as load_log:
        load_log.write(f"hparams:\n{hparams}\n")
        load_log.write(f'###############  Starting new run at {datetime.datetime.now()} #################\n')
    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    if hparams.disc_wt and args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer,
                                reset_optimizer=False, overwrite_global_states=False)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True,
                                overwrite_global_states=False)

    # Train!
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
