# Lit review

## Deepfake Synthesis

### [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild](PrajwalLipsync2020.pdf) Prajwal et al 2020
- use an expert descriminator model pretrained on real videos only rather than a descriminator trained in tandem with generator
- propose a new evaluation approach using the pretrained expert model rather than using masked real data which requires model to also
unrealistically have to generate pose changes at test time

## Deepfake Detection

### [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](Tolosana2020Deepfakes.pdf) Tolosana et al 2020
- Survey of supervised approaches to DeepFake video detection (as of early 2020)
- Covers four types of deepfakes: 1) entire face synthesis, 2) attribute manipulation, 3) identity swap, 4) expression sawp

### [Emotions Don’t Lie: An Audio-Visual Deepfake Detection Method using Affective Cues](MittalEmotionsDeepfakeDetection2020.pdf) Mittal et al 2020
- Uses discrepencies between visual and audio emotion features to detect Deepfakes
- TODO finish reading

## Datasets

### [The Socio-Moral Image Database (SMID): A novel stimulus set for the study of social, moral and affective processes](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190954) Crone et al 2018
- 2,941 freely available photographic images
- annoted for valence, arousal, and various moral theory metrics 
