# Instructions for use on matlaber compute cluster
based on https://nuwiki.media.mit.edu/bin/view/Main/NvidiaDocker

## Login and clone
### (note capitalization)
```
kinit -l 48h -r 10d -f user_name@MEDIA.MIT.EDU
ssh -K username@matlaber*.media.mit.edu
git clone https://github.com/jagnusson/Wav2Lip-Emotion.git
```

## docker

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
# now you can move your ~t<image name>-<version tag>.tar.gz export to another machine or just import it on another matlaber
docker load -i ~/<image name>-<version tag>.tar.gz
docker run --shm-size 16g -dit -v /mas:/mas -v /u:/u -v /dtmp:/dtmp -v /tmp:/tmp --name $(whoami)-<image name>-<version tag> $(whoami)/<image name>-<version tag>
# I recommend removing your image immediately after
docker rmi $(whoami)/<image name>-<version tag>
```
