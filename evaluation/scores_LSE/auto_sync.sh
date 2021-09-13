short_name=$1
for f in ../../results/mead_results/mead_${short_name}2?_full/
do
	base=`echo $f | cut -d "/" -f5`
	echo "python calculate_scores_LRS.py --data_root /u/ianmag/emotion-synthesis/results/mead_results/$base/ > /u/ianmag/emotion-synthesis/results/mead_results/$base/sync_scores.txt"
	python calculate_scores_LRS.py --data_root /u/ianmag/emotion-synthesis/results/mead_results/$base/ > /u/ianmag/emotion-synthesis/results/mead_results/$base/sync_scores.txt
done
