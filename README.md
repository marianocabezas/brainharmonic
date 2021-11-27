# Brainharmonic - Generating music from brain signal data using deep learning
## Description
Wouldn't it be great to hear your thought in musical form? 

For years, neuroscientists have focused their research in understanding how the brain works. Non-invasive techniques, such as fMRI and xEG (electroencephalography variants), have been widely use to probe the brain to monitor neuron activity through imaging and electric signal acquistion. Thus, through these techniques we can study brain activity as a set of signals through time. Could we use that information to create songs?

This project aims to develop a tool to generate music from brain activity using deep learning models. There has been a lot of work on generating new music from a large collection of music using deep AI models (usually with adversarial networks), as well as some work on generating music from brain xEG/fMRI signal through algorithmic or rule based approaches. However, there has not been any approach developed using deep learning models to generate music from EEG/fMRI signal.

In this project, we will use deep generative models to allow xEG/fMRI data to be used as an input to generate music. Signals, extracted from either xEG or fMRI, will be used as a starting point to extract features into a latent space. In parallel, we will use self-supervised techniques to link that latent space to the generation of new songs.
Furthermore, we would also like to look into linking brain signal from different regions of the brain to different instruments to create a symphony. Different brain signals, could be linked to a specific piece of a song played by an instrument and at a later stage, different instruments could be fused together into a harmonic song.


## Background
While this is a novel idea and not many works deal with similar topics, we have collected a list of resources to take ideas that will help for each different task:
[Carlos Hernandez-Olivan, Jose R. Beltran, "Music Composition with Deep Learning: A Review"](https://arxiv.org/abs/2108.12290)
[Jing Lu, et al. "Scale-Free Brain-Wave Music from Simultaneously EEG and fMRI Recordings"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0049773)

We would also like to include some repositories on works related to music generation:
[Synthesizer](https://github.com/irmen/synthesizer)
[MuseGAN](http://github.com/salu133445/musegan)
[Deep Symphony](https://hackmd.io/@zCouBXpGTjeQekat74moFw/rkZsNt9xf?type=view) (report with links to useful repositories and training data)

This is by no means an exhaustive list and expertise from different points of view is valued.

## Methodology/Approach
This project consists of a few different part, some of them optional depending on how the progress advances:
- Processing fMRI sequences into a set of signals for each subject (Optional): fMRI is a highly-dimensional image sequence. For each subject thousands of voxels are acquired and each one of them represents a signal. To reduce the complexity, conventional techniques would need to be used to have a small set of signals that represent some major areas in the brain. This task would involve preprocessing the 4D sequence, parcellating the brain into regions, and obtaining a representative signal (mean) for each region. Conventional tools, such as [fmriprep](https://fmriprep.org/en/stable/) and [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)/[Fastsurfer](https://github.com/Deep-MI/FastSurfer) can help with that task. Since this is not a research project, rough, easy and fast solutions are welcomed for this step.
- Converting a brain signal (either from xEG or preprocessed fMRI) into a sequence of "tokens" in a latent space: The goal of this task is to simplify the input signals into a more manageable space we can define for the music generation network. Several solutions can be employed here, ranging from basic patient clustering (to define subject similarity) followed by feature extraction on the signals, to deep learning techniques to classify subject based on their categories. Solutions for this task should aim at generating tokens that are close in the latent space if the subjects or features are similar and distant tokens for dissimilar ones.
- Training (or reusing a pretrained network) in a self-supervised manner to define a musical latent space: The goal of this task is to generate a musical latent space that follows the same properties as the brain signal space, while also providing a good starting point for music generation. Thus, this latent space should group musical features together if they are similar (for example, songs with a similar style) and separate dissimilar musical features. Furthermore, the dimensions and ranges of the dimensions should match those of the brain signal latent space.
- Converting model predictions into music files: Depending on how the output of the model is predicted (musical notes, cords, signal values, etc.), these predictions will need to be converted into songs.
- Creating different tracks for a multi-track song and fusing them together (Optional): xEG and fMRI sequences, can provide multiple signals per subject. Therefore, why not use them as different instruments for a harmonic composition? While the models developed on the previous two tasks could be used independently for each different signal, harmony should be key. There is a limit to how many instruments should overlap at each time. The goal of this task is to ensure that harmony is kept when multiple parallel instrumental tracks are generated.


## Data
This project is purely open source. Therefore, we will be using publicly available data following open policies and the appropriate privacy and ethical regulations. For that reason, our data we will be gathered from [OpenNeuro](https://openneuro.org/). Ideally, we would like to use data from studies were groups are defined, or where subjects can be clustered, to represent different styles of music and to map them into a latent space based on similarity.

## Collaborator set-up requirements
The whole project will be based on python 3.x coding. While other preprocessing tools not based on python for fMRI/xEG can be used to clean or prepare the input signals for the model, the models will be implemented in python, preferrably using pytorch. Some useful python packages to start hacking include:
- [Numpy](https://numpy.org/) and [Scipy](https://scipy.org/) (basic matrix operations)
- [Pytorch](https://pytorch.org/docs/stable/tensors.html) (deep learning and differential methods)
- [Nibabel](https://nipy.org/nibabel/) (nifti file loading for fMRI)
- [scikit-learn](https://scikit-learn.org/stable/) (classical machine learning techniques such as clustering)
- [MNE](https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html) (xEG data loading)
- [Pooch](https://pypi.org/project/pooch/) (companion to MNE to download xEG datasets)
- [Pyo AJAX](http://ajaxsoundstudio.com/software/pyo/) (signal processing library)
