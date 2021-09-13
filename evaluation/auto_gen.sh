short_name=$1
source_affect=$2
device=$3
for f in ../checkpoints/mead_${short_name}2?_full/
do
	base=`echo $f | cut -d "/" -f3`
	echo python3 real_videos_inference.py --mode affect --filelist test_filelists/mead/$source_affect.txt --data_root ../../mead_resampled/$source_affect/ --results_dir ../results/mead_results/$base --checkpoint_path $f/checkpoint_canonical_*  --face_res 96 --min_frame_res 160 --gpu_id $device --full_mask
	python3 real_videos_inference.py --mode affect --filelist test_filelists/mead/$source_affect.txt --data_root ../../mead_resampled/$source_affect/ --results_dir ../results/mead_results/$base --checkpoint_path $f/checkpoint_canonical_*  --face_res 96 --min_frame_res 160 --gpu_id $device --full_mask
done
