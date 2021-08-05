data_dir=$1
yourfilenames=`ls $1 | grep .mp4`
score_output_file=$2
export CUDA_VISIBLE_DEVICES=$3

counter=0
tmp_dir_name=tmp_dir$counter
while [ -d $tmp_dir_name ]
do
    counter=$(( $counter + 1 ))
    tmp_dir_name=tmp_dir$counter
done
echo using $tmp_dir_name
mkdir $tmp_dir_name

if [ -f $score_output_file ]; then
   echo "score file already exists, aborting"
   exit 1
fi
filecount=`ls $1 | grep .mp4 | wc -l`
echo $filecount
for eachfile in $yourfilenames
do
   python run_pipeline.py --videofile $1/$eachfile --reference wav2lip --data_dir $tmp_dir_name --min_track 50 --min_face_size 50 > /dev/null 2>&1
   python calculate_scores_real_videos.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir --batch_size 64 >> $score_output_file 2> /dev/null
   echo processing $eachfile
done | pv -l -s $filecount > /dev/null
echo removing $tmp_dir_name
rm -r $tmp_dir_name
