# Lit review

## [Our ADGD @ ACMMM 2021 paper](https://github.com/jagnusson/Wav2Lip-Emotion/blob/main/literature/Wav2Lip-Emotion_updated.pdf)
- We extend wav2lip to modify facial expressions of emotions via l1 reconstruction and pre-trained emotion objectives.
- We propose a novel automatic evaluation for emotion modification corroborated with a user study. 

## Deepfake Synthesis

### [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild](PrajwalLipsync2020.pdf) Prajwal et al 2020
- use an expert descriminator model pretrained on real videos only rather than a descriminator trained in tandem with generator
- propose a new evaluation approach using the pretrained expert model rather than using masked real data which requires model to also
unrealistically have to generate pose changes at test time

## Temporal Consistency / Video-to-Video

### [World-Consistent Video-to-Video Synthesis](https://arxiv.org/pdf/2007.08509.pdf) Mallya et al 2020
- recent work from NVIDA on "world consistency" which is some kind of fancy long range temporal consistency
- TODO: mine related works section for recent work on ordinary temporal consistency

## Deepfake Detection

### [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](Tolosana2020Deepfakes.pdf) Tolosana et al 2020
- Survey of supervised approaches to DeepFake video detection (as of early 2020)
- Covers four types of deepfakes: 1) entire face synthesis, 2) attribute manipulation, 3) identity swap, 4) expression sawp

### [Emotions Don’t Lie: An Audio-Visual Deepfake Detection Method using Affective Cues](MittalEmotionsDeepfakeDetection2020.pdf) Mittal et al 2020
- Uses discrepencies between visual and audio emotion features to detect Deepfakes
- TODO finish reading

## Datasets

## [CMU-MOSEI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) Zadeh et al 2018
- emotion labeled videos in the wild (from youtube)
- labeled by crowd workers at the sentence level
- also contains visemes and sentiment

## [MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation](https://wywu.github.io/projects/MEAD/MEAD.html) Wang et al 2020
- largest controled emotion video dataset
- 60 actors each with about 40 mins recoreded over 8 emotions and multiple viewing angles
- TODO figure out what the number of emotion annotions per video is (probably not sentnece level)

### [The Socio-Moral Image Database (SMID): A novel stimulus set for the study of social, moral and affective processes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190954) Crone et al 2018
- 2,941 freely available photographic images
- annoted for valence, arousal, and various moral theory metrics 

### [EmoSen: Generating Sentiment and Emotion Controlled Responses in a Multimodal Dialogue System](https://ieeexplore.ieee.org/document/9165162) Firdaus et al 2020
- Sentiment and Emotion Multimodal Dialogue Dataset of 10 tv shows annotated (at sentence level?) for sentiment and emotion
- not yet available because journal hasn't been published!
