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


path = "samples/music/"

mid = MidiFile(path+'35064.mid', clip=True)
mid2 = MidiFile(path+'albinoni1.mid')
print(mid)

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
    f = abs(np.fft.fft(signal))
    spectrum = f.real * f.real + f.imag * f.imag
    nspectrum = spectrum / spectrum[0]
    spread = np.std(nspectrum)
    return spread


def map_eeg_channel_signal_to_midi_channel(eeg_channel_signal,eeg_global_signal):
    channel_spectrum_spread = get_spectrum_spread(eeg_channel_signal)
    global_spectrum_spread = get_spectrum_spread(eeg_global_signal)
    spectrum_ratio = channel_spectrum_spread/global_spectrum_spread
    if spectrum_ratio>=1:
        midi_channel_rank = 9
    else:
        midi_channel_rank = floor(spectrum_ratio*10)%10
    midi_channel = timbre_ranking[midi_channel_rank-1,1]
    return midi_channel


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

def add_special_effects(song,special_effects):


transposed_track = transpose_song(mid,transpose_steps=12,new_instrument=42,new_channel=1)
delayed_track = add_delayed_track(mid,delay_time=3,new_instrument=0,new_channel=2)
skipped_notes_track = add_skipped_notes_track(mid,number_notes_skipped=5,new_instrument=6,new_channel=3)
mid.tracks.append(transposed_track)
mid.tracks.append(delayed_track)
mid.tracks.append(skipped_notes_track)

mid.save("/Users/arianguyen/Development/brainharmonic/outputs/35064_added_3.mid")

