FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y sudo vim wget
RUN apt-get install -y --allow-change-held-packages libcudnn7 libcudnn7-dev
RUN apt-get install -y libgl-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install -y libsndfile-dev
RUN apt-get install -y less
RUN apt-get install -y ffmpeg
RUN apt-get install -y git
RUN apt-get install -y python3-pip python3-dev python3-wheel
# required to fix opencv fail with skbuild (https://stackoverflow.com/questions/63448467/installing-opencv-fails-because-it-cannot-find-skbuild)
RUN pip3 install --upgrade pip setuptools

# required to make dlib work (https://stackoverflow.com/questions/48503646/where-should-i-install-cmake/51856073)
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libgtk-3-dev
RUN apt-get install -y libboost-all-dev

# addditonal dependencies for scripts
RUN apt-get install -y parallel
RUN apt-get install -y pv
RUN apt-get install -y imagemagick
