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

path = 'samples/music/motifs'
#if not os.path.exist(path):
#    os.mkdir(path)

# parameters
number_of_notes_in_motif = 5

# PREPROCESS EEG DATA
# artifact removal

# GENERATE NOTES
def find_most_dominant_freq(signal):
    # apply FFT and take absolute value
    f = abs(np.fft.fft(signal))

    # get the list of frequencies
    x = list(range(len(signal)))
    num = np.size(x)
    freq = [i / num for i in list(range(num))]

    # get the list of spectrums
    spectrum = f.real * f.real + f.imag * f.imag
    nspectrum = spectrum / spectrum[0]

    results = pd.DataFrame({'freq': freq, 'nspectrum': nspectrum})
    indx_max = np.argmax(results['nspectrum'])
    most_dominant_freq = results.iloc[indx_max]['freq']
    return most_dominant_freq


def generate_notes(raw_signal, number_of_notes_in_motif):
    #this will return a specified number of notes in a 13-notes range.
    # The range is determined by the overall frequency of the whole signal sequence
    notes = []

    mean_signal_cross_channels = np.mean(raw_signal, axis=0)

    # find most dominant freg on the whole signal sequence
    most_dominant_freq = find_most_dominant_freq(mean_signal_cross_channels)

    # map to a relevant range of notes - the higher the frequency, the higher the note.
    # From this note we identify a corresponding range of 12 notes with this note as the centre
    corresponding_note = floor(most_dominant_freq * 128)
    min_note_in_motif = max(0, min(corresponding_note - 6, 115))

    sds = [] #array of standard deviation for each segment
    sd = np.std(mean_signal_cross_channels)

    # generate the note for each sequence
    for i in range(number_of_notes_in_motif):
        segment_i = np.array_split(mean_signal_cross_channels, number_of_notes_in_motif)[i]
        most_dominant_freq_in_segment = find_most_dominant_freq(segment_i)
        note_for_segment = floor(min_note_in_motif + most_dominant_freq_in_segment * 12)
        notes.append(note_for_segment)
        sd_signal_strength = np.std(segment_i)
        sds.append(sd_signal_strength)

    #if the signal is too static and there's only one or two notes in the whole motif,
    # generate the sequence around on the dominant note, with the distance of the surrounding notes proportional to the standard deviation of the sequence
    most_frequent_note = max(notes, key=notes.count)
    if notes.count(most_frequent_note)/len(notes) > 0.5:
        for i in range(number_of_notes_in_motif-1):
            i = i + 1 #keep the first note as it is
            print(i)
            if sds[i] < sd:
                notes[i] = max(0,notes[i] - floor(sds[i]/sd * 13))
            elif sds[i] > sd:
                notes[i] = min(127,notes[i] + floor(sds[i] / sd * 13))

    return notes

# GENERATE LOUDNESS (= velocity for midi)
def generate_velocities(raw_signal, number_of_notes_in_motif):
    # this returns the loudness of each note in the motif.
    # The loudness is evaluated based on the signal strength for each segment
    velocities = []
    mean_signal_cross_channels = np.mean(raw_signal, axis=0)
    mean_signal_cross_channels_scaled  = (mean_signal_cross_channels - np.min(mean_signal_cross_channels))
    capped_signal_strength = np.quantile(mean_signal_cross_channels_scaled,0.95) #cap signal strength at 0.95 quantile to avoid extreme ones

    for i in range(number_of_notes_in_motif):
        segment_i = np.array_split(mean_signal_cross_channels_scaled, number_of_notes_in_motif)[i]
        median_signal_strength = np.quantile(segment_i,0.5)
        # if the median signal strength for this segment is greater than the 0.95 quantile of the whole sequence strength,
        # cap the velocity at 127
        velocity_for_segment = min(127, round(128 * median_signal_strength/capped_signal_strength))
        velocities.append(velocity_for_segment)

    return velocities

# GENERATE DURATIONS

def generate_durations(raw_signal, number_of_notes_in_motif):
    durations = []
    mean_signal_cross_channels = np.mean(raw_signal, axis=0)
    sd_whole_sequence = np.std(mean_signal_cross_channels)
    sds = [] # array of standard deviations for each segment

    for i in range(number_of_notes_in_motif):
        segment_i = np.array_split(mean_signal_cross_channels, number_of_notes_in_motif)[i]
        sd_signal_strength = np.std(segment_i)
        sds.append(sd_signal_strength)

    sd_sds = np.std(sds)
    high_volatilities = []

    for i in range(number_of_notes_in_motif):
        if sd_whole_sequence - sd_sds <= sds[i] <= sd_whole_sequence + sd_sds:
            duration_for_segment = 1
        elif sds[i] > sd_whole_sequence + sd_sds:
            duration_for_segment = 0.5
        elif sds[i] < sd_whole_sequence - sd_sds:
            duration_for_segment = 2

        if sds[i] > sd_whole_sequence + 2*sd_sds:
            high_volatility = 1
        else:
            high_volatility = 0
        durations.append(duration_for_segment)
        high_volatilities.append(high_volatility)

    return durations, high_volatilities

# Other experiments
def add_random_fast_notes(note,number_added_notes):
    notes = [note] * number_added_notes + random.sample(range(-6,6),number_added_notes)
    durations = [0.25] * number_added_notes
    return notes, durations

def add_ending_note_for_long_motif(notes):
    most_frequent_note = max(notes,key=notes.count)
    notes.append(most_frequent_note)


# clustering for timbre groups

# testing
subjects = range(108)
runs = [1]
tickspeed = 480

for subject in subjects:
    subject = subject + 1
    print("Subject %s" %subject)

    # Get data and locate in to given path
    files = eegbci.load_data(subject, runs, '../datasets/')
    raws = [read_raw_edf(f, preload=True) for f in files]
    # Combine all loaded runs
    raw_obj = concatenate_raws(raws)
    open_eyes_data = raw_obj.get_data()

    midi_file = File(tickspeed=tickspeed)
    track_name = "subject" + str(subject) + "_run" + str(runs)
    track = midi_file.add_track(track_name)

    notes = generate_notes(open_eyes_data,number_of_notes_in_motif)
    print("Notes: ",notes)
    velocities = generate_velocities(open_eyes_data,number_of_notes_in_motif)
    print("Velocities: ", velocities)
    durations = generate_durations(open_eyes_data,number_of_notes_in_motif)
    print("Durations: ",durations)

    cumulative_time = np.cumsum(durations[0])
    cumulative_time = cumulative_time * tickspeed
    cumulative_time = [int(x) for x in cumulative_time]
    print("Cumulative time: ",cumulative_time)

    for i in range(number_of_notes_in_motif):
        track.add_note(note=notes[i],time=cumulative_time[i],length=durations[0][i],velocity=velocities[i])

    file_name = path + "/subject" + str(subject) + "_run" + str(runs[0]) + ".mid"
    midi_file.save(file_name)