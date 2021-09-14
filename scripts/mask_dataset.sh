#!/bin/bash
data_dir=$1
tmp_dir=$2

process_image () {
	# if output doesn't already exist
	if [ ! -f "$2" ]
	then
		python3 emotion-synthesis/mask_preprocessed_images.py --input $1 --output $2
	fi
}

export -f process_image


for f in $(ls $data_dir)
do
	ls -d $data_dir/${f}/?????/*.jpg | parallel --tmpdir $tmp_dir --jobs -4 --bar process_image {} {//}-m/{/}
done
