import sys
import os
sys.path.append(".")
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from math import floor
import random
from eeg_midi import File
from eeg_midi import Track
from eeg_midi import Note
from mido import MidiFile
import glob
import os
import music21
import copy
from mido import Message, MidiFile, MidiTrack
from scipy import signal

from scipy.ndimage import uniform_filter1d, sobel, prewitt, fourier_gaussian, fourier_shift
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from pyts.decomposition import SingularSpectrumAnalysis # SSA for timeseries
import heapq # used in the sort of the n-largest index of a vector
    

#ranking of different midi channel (instruments)'s timbre
timbre_ranking = np.array([[1,73],
                           [2,19],
                           [3,68],
                           [4,71],
                           [5,31],
                           [6,40],
                           [7,56],
                           [8,42],
                           [9,58]])


def get_spectrum_spread(signal):
    signal = signal - np.mean(signal)
    signal = signal / np.std(signal)
    f = abs(np.fft.fft(signal))
    spectrum = f.real * f.real + f.imag * f.imag
    nspectrum = spectrum / spectrum[0]
    spread = np.std(nspectrum)
    return spread

def map_three_channels(X):
    midi_channels = [1, 5, 9]
    spreads = [get_spectrum_spread(x) for x in X]
    inds = np.argsort(spreads)
    return [midi_channels[i] for i in inds]

def map_eeg_channel_signal_to_midi_channel(eeg_channel_signal,eeg_global_signal):
    channel_spectrum_spread = get_spectrum_spread(eeg_channel_signal)
    global_spectrum_spread = get_spectrum_spread(eeg_global_signal)
    spectrum_ratio = channel_spectrum_spread/global_spectrum_spread
    if spectrum_ratio >= 1:
        midi_channel_rank = 9
    else:
        midi_channel_rank = floor(spectrum_ratio*10)%10
    midi_channel = timbre_ranking[midi_channel_rank-1,1]
    return midi_channel


def get_transpose_steps(eeg_channel_signal,eeg_global_signal):
    channel_sd = np.std(eeg_channel_signal)
    global_sd = np.std(eeg_global_signal)
    sd_ratio = channel_sd / channel_sd
    specific_intervals = [2,4,5,7,9,11,12]
    if sd_ratio >= 1:
        transpose_steps = 12
    else:
        transpose_steps = specific_intervals[floor(sd_ratio * 10) % 7 - 1]
    return transpose_steps


def get_time_delay(eeg_channel_signal,eeg_global_signal):
    correlation = signal.correlate(eeg_channel_signal - np.mean(eeg_channel_signal), eeg_global_signal - np.mean(eeg_global_signal), mode="full")
    lags = signal.correlation_lags(len(eeg_channel_signal), len(eeg_global_signal), mode="full")
    lag = floor(lags[np.argmax(abs(correlation))])
    return lag


def transpose_song(song,transpose_steps,new_instrument,new_channel):
    track = song.tracks[0]
    new_track = copy.deepcopy(track)
    new_track.append(Message('program_change', program=new_instrument, time=0, channel=new_channel))
    for message in new_track:
        if message.type in ('note_on', 'note_off'):
            message.note += transpose_steps
            message.channel = new_channel
    return new_track


def add_delayed_track(song, delay_time,new_instrument,new_channel):
    track = song.tracks[0]
    new_track = copy.deepcopy(track)
    new_track.append(Message('program_change', program=new_instrument, time=0, channel=new_channel))
    for message in new_track:
        if message.type in ('note_on', 'note_off'):
            message.time += delay_time*4*48
            message.channel = new_channel
    return new_track


def add_skipped_notes_track(song, number_notes_skipped, new_instrument, new_channel):
    track = song.tracks[0]
    new_track = copy.deepcopy(track)
    new_track.append(Message('program_change', program=new_instrument, time=0, channel=new_channel))

    for message in new_track:
        if message.type=='note_on':
            new_note = message.note
            break
    skipped_notes = 0
    for message in new_track:
        if message.type =='note_on':
            if message.note != new_note & skipped_notes <= number_notes_skipped:
                new_note = message.note
                message.velocity = 0
                skipped_notes += 1
            else:
                message.channel = new_channel
    return new_track


def get_max_time(song):
    # get the ending time
    max_time = 0
    for track in song.tracks:
        for message in track:
            if message.type in ('note_on', 'note_off') and message.time > max_time:
                max_time = message.time
    return max_time


def add_special_effects(song, special_effect, channel):
    max_time = get_max_time(song)
    last_track = song.tracks[-1]
    last_track.append(Message('program_change', program=special_effect, time=max_time-48*4, channel=channel))


# adjust track loudness
def change_track_velocity(track, velocity_factor):
    for message in track:
        if message.type in ('note_on', 'note_off'):
            message.velocity = floor(message.velocity*velocity_factor)
    return track


def create_multitrack(song,global_signal,first_signal, second_signal, third_signal,
                      transpose_up=True, transpose_down=True, add_delay=True,
                      add_skipped_notes=True, add_special_effect=False, change_track_velocity=False):

    # first_signal_instrument = map_eeg_channel_signal_to_midi_channel(first_signal,global_signal)
    # second_signal_instrument = map_eeg_channel_signal_to_midi_channel(second_signal, global_signal)
    # third_signal_instrument = map_eeg_channel_signal_to_midi_channel(third_signal, global_signal)
    instruments = map_three_channels([first_signal, second_signal, third_signal])
    first_signal_instrument, second_signal_instrument, third_signal_instrument = instruments

    if first_signal_instrument == second_signal_instrument:
        # convert second signal to a different instrument
        condition = timbre_ranking[:, 1] == second_signal_instrument
        instrument_ranking = timbre_ranking[condition][0][0]
        new_instrument_ranking = 9 - instrument_ranking
        new_condition = timbre_ranking[:,0] == new_instrument_ranking
        second_signal_instrument = timbre_ranking[new_condition][0][1]

    channel = 0

    # transpose first and second signal
    first_signal_transpose_steps = get_transpose_steps(first_signal,global_signal)
    second_signal_transpose_steps = get_transpose_steps(second_signal, global_signal)

    if transpose_up:
        first_additional_track = transpose_song(song,first_signal_transpose_steps,first_signal_instrument,channel+1)
        if change_track_velocity:
            velocity_factor = min(np.mean(first_signal)/np.mean(global_signal),1)
            first_additional_track = change_track_velocity(first_additional_track,velocity_factor)
        channel += 1
        song.tracks.append(first_additional_track)
    if transpose_down:
        second_additional_track = transpose_song(song,second_signal_transpose_steps,second_signal_instrument,channel+1)
        if change_track_velocity:
            velocity_factor = min(np.mean(second_signal) / np.mean(global_signal), 1)
            second_additional_track = change_track_velocity(second_additional_track, velocity_factor)
        channel += 1
        song.tracks.append(second_additional_track)

    if add_delay:
        delay = get_time_delay(third_signal,global_signal)
        third_additional_track = add_delayed_track(song,delay,third_signal_instrument,channel+1)
        if change_track_velocity:
            velocity_factor = min(np.mean(second_signal) / np.mean(global_signal), 1)
            third_additional_track = change_track_velocity(third_additional_track, velocity_factor)
        channel += 1
        song.tracks.append(third_additional_track)

    if add_skipped_notes:
        #can add more to this based on the global signal, right now the number of notes skipped and instrument is fixed
        skipped_notes_track = add_skipped_notes_track(song, number_notes_skipped=5, new_instrument=6, new_channel=channel+1)
        channel += 1
        song.tracks.append(skipped_notes_track)
    print(song)
    if add_special_effect:
        random_number = random.randint(0,9)
        if random_number%2 == 0:
            add_special_effects(song,126,channel)
        else:
            add_special_effects(song,127,channel)

    return song

# # Tested some functions
# path = "samples/music/"
# mid = MidiFile(path+'35064.mid', clip=True)
# transposed_track = transpose_song(mid,transpose_steps=12,new_instrument=42,new_channel=1)
# delayed_track = add_delayed_track(mid,delay_time=3,new_instrument=0,new_channel=2)
# skipped_notes_track = add_skipped_notes_track(mid,number_notes_skipped=5,new_instrument=6,new_channel=3)
# mid.tracks.append(transposed_track)
# mid.tracks.append(delayed_track)
# mid.tracks.append(skipped_notes_track)
# output_path = "outputs/"
# mid.save(output_path+"35064_added_3.mid")




def three_dominant_components(X):
    #input X is the EEG timeseries from one subject
    #return three_channels,TSeriesMatrix[three_channels] 
    #return to the No. of the three dominant channels and the main component of the three dominant channels 
    
    # We decompose the time series into three subseries
    
    ## obtain the main part of each channel
    #using the first component of SSA.
    window_size = 15
    groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

    # Singular Spectrum Analysis
    ssa = SingularSpectrumAnalysis(window_size=15, groups=groups)
    X_ssa = ssa.fit_transform(X)

    TSeriesMatrix=np.zeros([X.shape[0],X.shape[1]])

    for i in range(X.shape[0]):
        TSeriesMatrix[i,:]=X_ssa[i,1]
    ### 1.divided the channel into three parts——by Hierarchical Clustering
    hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    channels=range(X.shape[0])
    labels = hierarchical_cluster.fit_predict(TSeriesMatrix) 
    channels_classify = list(zip(channels, labels))


    ### 2.give the most important channel in each class: 
    ### shows the most important channel for each class
    ###give a score of the channel for each class
    ### give a sort of the channel/brain region
    ### define the most important channel as most corr with each channel
    ### for channel i, add the correlation between timeseries xi and xj (j does not equal i) 
    class1_channel=np.where(labels==0)
    class2_channel=np.where(labels==1)
    class3_channel=np.where(labels==2)
    
    Timeseries_class1=TSeriesMatrix[class1_channel[0],:]
    Timeseries_class2=TSeriesMatrix[class2_channel[0],:]
    Timeseries_class3=TSeriesMatrix[class3_channel[0],:]

    FCM_class1 = np.corrcoef(Timeseries_class1)
    FCM_class2 = np.corrcoef(Timeseries_class2)
    FCM_class3 = np.corrcoef(Timeseries_class3)

    FCM_1 = FCM_class1
    FCM_2 = FCM_class2
    FCM_3 = FCM_class3
    
    corr_sum1=sum(np.nan_to_num(FCM_class1),1)
    main_channel=np.where(corr_sum1==np.max(corr_sum1))
    main_channel1=class1_channel[0][main_channel]

    corr_sum2=sum(np.nan_to_num(FCM_class2),1)
    main_channel=np.where(corr_sum2==np.max(corr_sum2))
    main_channel2=class2_channel[0][main_channel]
    
    corr_sum3=sum(np.nan_to_num(FCM_class3),1)
    main_channel=np.where(corr_sum3==np.max(corr_sum3))
    main_channel3=class3_channel[0][main_channel]

    three_channels=[main_channel1[0],main_channel2[0],main_channel3[0]]
    return three_channels,X[three_channels]
    