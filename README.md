# emotion-synthesis
A term project for 6.869 2021 by Aruna and Ian

## setup
### download
This setup assumes you are on a matlaber machine
```
# (note capitalization)
kinit -l 48h -r 10d -f user_name@MEDIA.MIT.EDU
ssh -K username@matlaber1.media.mit.edu
git clone https://github.com/jagnusson/emotion-synthesis.git
```

### docker

```
# get YOUR_UID (you will need this later)
id -u $(whoami)

# inside repo top directory so that it uses the Dockerfile there
docker build -t <image name>:<version tag> .
docker run --shm-size 16g -dit -v /mas:/mas -v /u:/u -v /dtmp:/dtmp -v /tmp:/tmp --name $(whoami)-<image name>-<version tag> <image name>:<version tag>
docker exec -it $(whoami)-<image name>-<version tag> bash

# You will now be inside the container and complete some installs that require user input
# enter MEDIA.MIT.EDU when prompted for default realm 
apt-get install krb5-user kstart

# make the mlusers group in the container
groupadd -g 2000 mlusers
# make your user in the container (you need to use the uid that you got earlier)
useradd -g 2000 -u YOUR_UID -m -s /bin/bash YOUR_USERNAME
# now become yourself and do some work
sudo -iu YOUR_USERNAME
cd /u/YOUR_USERNAME/emotion-synthesis
```
### python dependencies
```
pip3 install -r requirements.txt
```

### saveing and loading the docker

```
# after all of that you probably want to save your container image so you don't have to do it again or can run it on another matlaber
# make sure you are outside of your container for this
docker commit $(whoami)-<image name>-<version tag> $(whoami)/<image name>-<version tag>
docker save $(whoami)/<image name>-<version tag> | gzip > ~/<image name>-<version tag>.tar.gz
# I recommend removing your image immediately after
docker rmi $(whoami)/<image name>-<version tag>
```

```
# now you can move your ~tensorflow-1.tar.gz export to another machine or just import it on another matlaber
docker load -i ~/<image name>-<version tag>.tar.gz
docker run --shm-size 16g -dit -v /mas:/mas -v /u:/u -v /dtmp:/dtmp -v /tmp:/tmp --name $(whoami)-<image name>-<version tag> $(whoami)/<image name>-<version tag>
# I recommend removing your image immediately after
docker rmi $(whoami)/<image name>-<version tag>
```

## Setup on EC2
Most of the setup is similar on EC2, but you need to expose the GPUs and you don't want to run the userspace setup.

To expose the GPUs run the container like this:
```
docker run --gpus all -dit --name $(whoami)-<image name>-<version tag> <image name>:<version tag>
```
Confusingly this doesn't work on matlaber, and the gpus run fine without it. This was taken from https://towardsdatascience.com/how-to-properly-use-the-gpu-within-a-docker-container-4c699c78c6d1

## Demo to verify
Try out simple wav2lip inference to test that everything is working alright. Sample data and model weights are from [this google drive](https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k)

```
pip3 install gdown
# gdown ids are gotten from the sharing link
gdown --id 1bYX03oZHuvNKHWeQ46xLqTrs7miGvIno
gdown --id 1XIHCyHdP6YH1f2ShtaF9BfV3BTeLWD-V
gdown --id 1_OvqStxNxLc7bXzlaVG5sz695p-FVfYY
python3 inference.py --checkpoint_path wav2lip_gan.pth --face input_vid.mp4 --audio input_audio.wav
```
This will output a file `results/result_voice.mp4`. Use `scp -o GSSAPIAuthentication=yes username@matlaber*.media.mit.edu:/u/username/emotion-synthesis/results/result_voice.mp4` to get this locally to look at. Inference should complete quickly if gpu is working correctly (check this to make sure that its not running on CPU; the device uses is printed out at the beginning of the command).

