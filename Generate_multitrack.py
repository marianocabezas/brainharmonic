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
    lag = lags[np.argmax(abs(correlation))]
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
            message.velocity  = message.velocity*velocity_factor
    return track



def create_multitrack(song,global_signal,first_signal, second_signal, third_signal,
                      transpose_up=True, transpose_down=True, add_delay=True,
                      add_skipped_notes=True, add_special_effect=False, change_track_velocity=False):

    first_signal_instrument = map_eeg_channel_signal_to_midi_channel(first_signal,global_signal)
    second_signal_instrument = map_eeg_channel_signal_to_midi_channel(second_signal, global_signal)
    third_signal_instrument = map_eeg_channel_signal_to_midi_channel(third_signal, global_signal)

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

    if add_special_effect:
        random_number = random.randint(0,9)
        if random_number%2 == 0:
            add_special_effects(song,126,channel)
        else:
            add_special_effects(song,127,channel)

    return(song)

# Tested some functions
path = "samples/music/"
mid = MidiFile(path+'35064.mid', clip=True)
transposed_track = transpose_song(mid,transpose_steps=12,new_instrument=42,new_channel=1)
delayed_track = add_delayed_track(mid,delay_time=3,new_instrument=0,new_channel=2)
skipped_notes_track = add_skipped_notes_track(mid,number_notes_skipped=5,new_instrument=6,new_channel=3)
mid.tracks.append(transposed_track)
mid.tracks.append(delayed_track)
mid.tracks.append(skipped_notes_track)
output_path = "outputs/"
mid.save(output_path+"35064_added_3.mid")

