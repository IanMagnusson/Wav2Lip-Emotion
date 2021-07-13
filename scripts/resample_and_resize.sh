#!/bin/bash
# a script to use ffmpeg to resample the MEAD dataset and resize it
# options selected based on https://stackoverflow.com/questions/11004137/re-sampling-h264-video-to-reduce-frame-rate-while-maintaining-high-image-quality

mead_root=$1

FRAME_RATE=25
RESOLUTION=480x270
CODEX=libx264
BITRATE=3M

# paths of from root/actor/videochunk/orientation/affect/intensity/
file_count=$(ls -1q $mead_root/*/*/front/*/*/*.mp4 | wc -l)
for videopath in $mead_root/*/*/front/*/*/*.mp4
do
	new_videopath=${videopath%.mp4}_resampsize.mp4
	ffmpeg -i $videopath -r $FRAME_RATE -s $RESOLUTION -c:v $CODEX -b:v $BITRATE -movflags faststart $new_videopath
	rm $videopath
done | pv -l -s $file_count > /dev/null 
