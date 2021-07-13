#!/bin/bash
# a script to untar all the files in the mead dataset at the provided root diri

mead_root_dir=$1

shopt -s nullglob # to handle empty actor_paths without .tars

for actor_path in $mead_root_dir/*
do
	for video_file in $actor_path/*.tar
	do
		tar --extract --verbose --file $video_file --directory $actor_path
	done
done
