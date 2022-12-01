# Brainharmonic - Generating music from brain signal data using deep learning
## Description
Wouldn't it be great to hear your thoughts in musical form? 

For years, neuroscientists have focused their research in understanding how the brain works. Non-invasive techniques, such as fMRI and xEG (electroencephalography variants), have been widely use to probe the brain to monitor neuron activity through imaging and electric signal acquistion. Thus, through these techniques we can study brain activity as a set of signals through time. Could we use that information to create songs?

This project aims to develop a tool to generate music from brain activity using deep learning models. There has been a lot of work on generating new music from a large collection of music using deep AI models (usually with adversarial networks), as well as some work on generating music from brain xEG/fMRI signal through algorithmic or rule based approaches. However, there has not been any approach developed using deep learning models to generate music from EEG/fMRI signal.

In this project, we will use deep generative models to allow xEG/fMRI data to be used as an input to generate music. Signals, extracted from either xEG or fMRI, will be used as a starting point to extract features into a latent space. In parallel, we will use self-supervised techniques to link that latent space to the generation of new songs.
Furthermore, we would also like to look into linking brain signal from different regions of the brain to different instruments to create a symphony. Different brain signals, could be linked to a specific piece of a song played by an instrument and at a later stage, different instruments could be fused together into a harmonic song.


## Background
While this is a novel idea and not many works deal with similar topics, we have collected a list of resources to take ideas that will help for each different task:
- [Carlos Hernandez-Olivan, Jose R. Beltran, "Music Composition with Deep Learning: A Review"](https://arxiv.org/abs/2108.12290)
- [Jing Lu, et al. "Scale-Free Brain-Wave Music from Simultaneously EEG and fMRI Recordings"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0049773)
- [Sageev Oore, et al. "This Time with Feeling: Learning Expressive Musical Performance"](https://arxiv.org/abs/1808.03715)
- [Cheng-Zhi Anna Huang et al. "Music transformer"](https://arxiv.org/abs/1809.04281)

We would also like to include some repositories on works related to music generation:
[Synthesizer](https://github.com/irmen/synthesizer)
[MuseGAN](http://github.com/salu133445/musegan)
[Deep Symphony](https://hackmd.io/@zCouBXpGTjeQekat74moFw/rkZsNt9xf?type=view) (report with links to useful repositories and training data)

This is by no means an exhaustive list and expertise from different points of view is valued.

Extra resources on MIDI files and music theory:
- [MIDI note information](https://newt.phys.unsw.edu.au/jw/notes.html)
- [Reddit post on the middle MIDI note](https://www.reddit.com/r/musictheory/comments/8fwoti/why_does_middle_c_midi_note_60/)
- [MIDI tutorial](https://www.cs.cmu.edu/~music/cmsip/readings/MIDI%20tutorial%20for%20programmers.html)
- [MIDI file standard](https://www.cs.cmu.edu/~music/cmsip/readings/MIDI%20tutorial%20for%20programmers.html)
- [Frequency and pitch](https://www.animations.physics.unsw.edu.au/jw/frequency-pitch-sound.htm)
- [Introduction to tempo](https://courses.lumenlearning.com/musicappreciation_with_theory/chapter/introduction-to-tempo/)
- [MIDI programs](https://en.wikipedia.org/wiki/General_MIDI)

## Methodology/Approach
This project consists of a few different part, some of them optional depending on how the progress advances:
- Processing fMRI sequences into a set of signals for each subject (Optional): fMRI is a highly-dimensional image sequence. For each subject thousands of voxels are acquired and each one of them represents a signal. To reduce the complexity, conventional techniques would need to be used to have a small set of signals that represent some major areas in the brain. This task would involve preprocessing the 4D sequence, parcellating the brain into regions, and obtaining a representative signal (mean) for each region. Conventional tools, such as [fmriprep](https://fmriprep.org/en/stable/) and [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)/[Fastsurfer](https://github.com/Deep-MI/FastSurfer) can help with that task. Since this is not a research project, rough, easy and fast solutions are welcomed for this step.
- Converting a brain signal (either from xEG or preprocessed fMRI) into a sequence of "tokens" in a latent space: The goal of this task is to simplify the input signals into a more manageable space we can define for the music generation network. Several solutions can be employed here, ranging from basic patient clustering (to define subject similarity) followed by feature extraction on the signals, to deep learning techniques to classify subject based on their categories. Solutions for this task should aim at generating tokens that are close in the latent space if the subjects or features are similar and distant tokens for dissimilar ones.
- Training (or reusing a pretrained network) in a self-supervised manner to define a musical latent space: The goal of this task is to generate a musical latent space that follows the same properties as the brain signal space, while also providing a good starting point for music generation. Thus, this latent space should group musical features together if they are similar (for example, songs with a similar style) and separate dissimilar musical features. Furthermore, the dimensions and ranges of the dimensions should match those of the brain signal latent space.
- Converting model predictions into music files: Depending on how the output of the model is predicted (musical notes, cords, signal values, etc.), these predictions will need to be converted into songs.
- Creating different tracks for a multi-track song and fusing them together (Optional): xEG and fMRI sequences, can provide multiple signals per subject. Therefore, why not use them as different instruments for a harmonic composition? While the models developed on the previous two tasks could be used independently for each different signal, harmony should be key. There is a limit to how many instruments should overlap at each time. The goal of this task is to ensure that harmony is kept when multiple parallel instrumental tracks are generated.


## Data
This project is purely open source. Therefore, we will be using publicly available data following open policies and the appropriate privacy and ethical regulations. For that reason, our data we will be gathered from [OpenNeuro](https://openneuro.org/). Ideally, we would like to use data from studies were groups are defined, or where subjects can be clustered, to represent different styles of music and to map them into a latent space based on similarity.
EEG datasets currently included as samples:
- [Participants listening to different genres of music (ds003774)](https://openneuro.org/datasets/ds003774/versions/1.0.0)
- [During sleep (ds003768)](https://openneuro.org/datasets/ds003768/versions/1.0.2)
- [Emotions with naturalistic stimuli (ds003751)](https://openneuro.org/datasets/ds003751/versions/1.0.3)



Since this project deals both with brain signal files and musical files. In order to provide training samples, we have included MIDI files from the [DeepSymphony repository](https://github.com/Shaofanl/DeepSymphony). ideally, we should be able to include the [Magenta datasets](https://magenta.tensorflow.org/datasets).
- [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro#dataset)
- [GiantMIDI dataset](https://github.com/bytedance/GiantMIDI-Piano)

## Data loading and pre-processing
### EEG data
Another Brainhack project to standardise EEG data formats and processing pipeline that may be useful for this:
https://www.repronim.org/reproschema-ui/#/?url=https%3A%2F%2Fraw.githubusercontent.com%2FRemi-Gau%2FeCobidas%2Fhack_artemis_202110%2Fschemas%2Fartemis%2Fprotocols%2Fartemis_schema.jsonld

## Collaborator set-up requirements (git clone)
The whole project will be based on python 3.x coding. While other preprocessing tools not based on python for fMRI/xEG can be used to clean or prepare the input signals for the model, the models will be implemented in python, preferrably using pytorch. Some useful python packages to start hacking include:
- [Numpy](https://numpy.org/) and [Scipy](https://scipy.org/) (basic matrix operations)
- [Bottleneck](https://bottleneck.readthedocs.io/) (moving window operations, useful for timeseries)
- [Pytorch](https://pytorch.org/docs/stable/tensors.html) (deep learning and differential methods)
- [Nibabel](https://nipy.org/nibabel/) (nifti file loading for fMRI)
- [scikit-learn](https://scikit-learn.org/stable/) (classical machine learning techniques such as clustering)
- [MNE](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html) (xEG data loading)
- [Pooch](https://pypi.org/project/pooch/) (companion to MNE to download xEG datasets)
- [Mido](https://mido.readthedocs.io/en/latest/) (MIDI object processing package)
- [music21](https://web.mit.edu/music21/) (music encoding and other music-related tools)
- [pygame](https://www.pygame.org/news) (needed to play midi files with music21)
- [Pyo AJAX](http://ajaxsoundstudio.com/software/pyo/) (signal processing library)
- [Jupyter](https://jupyter.org/) (not strictly necessary, but helpful for prototyping code)
- [MuseScoe](https://musescore.org/) (show music sheets from MIDI files in your Jupyer Notebooks!)
- [AudioLazy](https://pypi.org/project/audiolazy/) ()


# For MuseMorphose, make sure to download all the datasets given in that repository's readme into its main directory.
```
cd ./MuseMorphose
# download REMI-pop-1.7K dataset
wget -O remi_dataset.tar.gz https://zenodo.org/record/4782721/files/remi_dataset.tar.gz?download=1
tar xzvf remi_dataset.tar.gz
rm remi_dataset.tar.gz
python attributes.py
wget -O musemorphose_pretrained_weights.pt https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1
```