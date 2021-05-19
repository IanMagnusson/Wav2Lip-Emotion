# Novel Evaluation Framework, new filelists, and using the LSE-D and LSE-C metric.

Our paper also proposes a novel evaluation framework (Section 4). To evaluate on LRS2, LRS3, and LRW, the filelists are present in the `test_filelists` folder. Please use `gen_videos_from_filelist.py` script to generate the videos. After that, you can calculate the LSE-D and LSE-C scores using the instructions below. Please see [this thread](https://github.com/Rudrabha/Wav2Lip/issues/22#issuecomment-712825380) on how to calculate the FID scores. 

The videos of the ReSyncED benchmark for real-world evaluation will be released soon. 

### Steps to set-up the evaluation repository for LSE-D and LSE-C metric:
We use the pre-trained syncnet model available in this [repository](https://github.com/joonson/syncnet_python). 

* Clone the SyncNet repository.
``` 
git clone https://github.com/joonson/syncnet_python.git 
```
* Follow the procedure given in the above linked [repository](https://github.com/joonson/syncnet_python) to download the pretrained models and set up the dependencies. 
    * **Note: Please install a separate virtual environment for the evaluation scripts. The versions used by Wav2Lip and the publicly released code of SyncNet is different and can cause version mis-match issues. To avoid this, we suggest the users to install a separate virtual environment for the evaluation scripts**
```
cd syncnet_python
pip install -r requirements.txt
sh download_model.sh
```
* The above step should ensure that all the dependencies required by the repository is installed and the pre-trained models are downloaded.
### Generating videos from filelists
(written by Ian)

There are two scripts for generating videos en-mass: `gen_videos_from_filelist.py` and `real_videos_inference.py`.
I believe `gen_videos_from_filelist.py` is just for LRS data, so we will use `real_videos_inference.py`.
```
python3 real_videos_inference.py --mode dubbed 
                                 --filelist <path to filelist>
                                 --results_dir <path to where you want videos to go>
                                 --data_root <path to actual videos to input to generator model>
                                 --checkpoint_path <path to model weights>
                                 --face_res 96
                                 --min_frame_res 160
```
The filelist needs to contain line separated filenames of the videos as they are within the `--data_root`, for example
```
001.mp4
002.mp4
```
The script is invoked in `dubbed` mode because this will run the nodel on the video and audio from each file without swapping
the audio around. Thus the outputs will just be the same audio but with affect changed.

The face_res automatically adjusts the face video resolution so that the detected face (as judged by the first video frame)
is approximately face_res size in its longest dimension. Meanwhile min_frame_res sets the minimum size that the smallest
dimension of the video can drop to as a result of adjusting for the face size.

### Running the evaluation scripts:
* Copy our evaluation scripts given in this folder to the cloned repository.
```  
    cd Wav2Lip/evaluation/scores_LSE/
    cp *.py syncnet_python/
    cp *.sh syncnet_python/ 
```
**Note: We will release the test filelists for LRW, LRS2 and LRS3 shortly once we receive permission from the dataset creators. We will also release the Real World Dataset we have collected shortly.**

* Our evaluation technique does not require ground-truth of any sorts. Given lip-synced videos we can directly calculate the scores from only the generated videos. Please store the generated videos (from our test sets or your own generated videos) in the following folder structure.
```
video data root (Folder containing all videos)
├── All .mp4 files
```
* Change the folder back to the cloned repository. 
```
cd syncnet_python
```

* To run evaluation on the ReSynced dataset or your own generated videos, please run the following command:
```
sh calculate_scores_real_videos.sh /path/to/video/data/root /path/to/file/to/place/scores /path/to/dir/to/place/FID/imgs
```
* This will generate LSE-D and LSE-C scores on the videos in the first arg and output them in that order, with a pair of
scores per video on a line, in a file at the second arg
* This command will also dump cropped faces from all the videos into the directory at the 3rd argument; make a different
folder for the videos of each model to be evaluated as well as the ground truth videos, as these will be compared by FID next.

#### Notes on calculate_scores_real_videos.sh
(written by Ian)

some assumptions set inside the syncnet repo code invoked by this script may cause the script to silently fail and output
an empty file of scores. These assumed arguments are set in `syncnet_python/run_pipeline.py`. Of note are: 
`--min_track` which should be lowered if input videos are short. `--min_face_size` which should be lowered if input videos
are small size.

# Evaluation of image quality using FID metric.
First install the metric
```
pip install pytorch-fid
```

Then to get the FID score between two sets of videos, we use the folders of cropped face images generated in the last step
and compare them. These folders are the ones that were give as the 3rd argument to `calculate_scores_real_videos.sh`
```
python -m pytorch_fid /path/to/face/crops/dir1 /path/to/face/crops/dir1 --device <your gpu such as cuda:0>
```


# Opening issues related to evaluation scripts
* Please open the issues with the "Evaluation" label if you face any issues in the evaluation scripts. 

# Acknowledgements
Our evaluation pipeline in based on two existing repositories. LSE metrics are based on the [syncnet_python](https://github.com/joonson/syncnet_python) repository and the FID score is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid) repository. We thank the authors of both the repositories for releasing their wonderful code.



