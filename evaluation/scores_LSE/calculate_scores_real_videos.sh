data_dir=$1
yourfilenames=`ls $1`
score_output_file=$2
FID_output_dir=$3

if [ -f $score_output_file ]; then
   echo "score file already exists, aborting"
   exit 1
fi

for eachfile in $yourfilenames
do
   python run_pipeline.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir >> $score_output_file
   for img in `ls tmp_dir/pytmp/wav2lip/ | grep .jpg`
   do
     cp tmp_dir/pytmp/wav2lip/$img $FID_output_dir/${eachfile}${img}
   done
done