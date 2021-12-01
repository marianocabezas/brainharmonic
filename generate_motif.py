import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from math import floor

# parameters
number_of_notes_in_motif = 5

# LOAD DATA
subject = 1  # use data from subject 1
runs = [1]  # use only hand and feet motor imagery runs

# Get data and locate in to given path
files = eegbci.load_data(subject, runs, '../datasets/')
raws = [read_raw_edf(f, preload=True) for f in files]
# Combine all loaded runs
raw_obj = concatenate_raws(raws)
open_eyes_data = raw_obj.get_data()
stats.describe(open_eyes_data)


# PREPROCESS EEG DATA
# artifact removal

# GENERATE NOTES

def find_most_dominant_freg(signal):
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
    most_dominant_freq = find_most_dominant_freg(mean_signal_cross_channels)

    # map to a relevant range of notes - the higher the frequency, the higher the note.
    # From this note we identify a corresponding range of 12 notes with this note as the centre
    corresponding_note = floor(most_dominant_freq * 128)
    min_note_in_motif = corresponding_note - 6

    # generate the note for each sequence
    for i in range(number_of_notes_in_motif):
        segment_i = np.array_split(mean_signal_cross_channels, number_of_notes_in_motif)[i]
        most_dominant_freq_in_segment = find_most_dominant_freg(segment_i)
        note_for_segment = floor(min_note_in_motif + most_dominant_freq_in_segment * 12)
        notes.append(note_for_segment)

    return notes

# GENERATE LOUDNESS (= velocity for midi)
def generate_velocities(raw_signal, number_of_notes_in_motif):
    # this returns the loudness of each note in the motif.
    # The loudness is evaulated based on the signal strength for each segment
    velocities = []
    mean_signal_cross_channels = np.mean(raw_signal, axis=0)
    capped_signal_strength = np.quantile(mean_signal_cross_channels,0.95) #cap signal strength at 0.95 quantile to avoid extreme ones

    for i in range(number_of_notes_in_motif):
        segment_i = np.array_split(mean_signal_cross_channels, number_of_notes_in_motif)[i]
        median_signal_strength = np.quantile(segment_i,0.5)
        # if the median signal strength for this segment is greater than the 0.95 quantile of the whole sequence strength,
        # cap the velocity at 127
        velocity_for_segment = min(127, 128 * median_signal_strength/capped_signal_strength)
        velocities.append(velocity_for_segment)

    return velocities

# GENERATE DURATIONS

def generate_durations(raw_signal, number_of_notes_in_motif):
    durations = []
    mean_signal_cross_channels = np.mean(raw_signal, axis=0)
    sd_whole_sequence = np.stf(mean_signal_cross_channels)
    sds = [] # array of standard deviations for each segment

    for i in range(number_of_notes_in_motif):
        segment_i = np.array_split(mean_signal_cross_channels, number_of_notes_in_motif)[i]
        sd_signal_strength = np.std(segment_i)
        sds.append(sd_signal_strength)

    sd_sds = np.stf(sds)
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
def add_random_fast_notes(motif):
    pass

def add_ending_note_for_long_motif(notes):
    most_frequent_note = max(notes,key=notes.count)
    notes.append(most_frequent_note)


# clustering for timbre groups
