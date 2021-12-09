from mido import MidiFile
import numpy as np
import os
import pandas as pd

def generate_midi_stats(midi_file):
    notes = []
    for i in midi_file:
        if i.type == 'note_on':
            notes.append(i.dict()["note"])

    note_sd = np.std(notes)
    number_unique_notes = len(np.unique(notes))
    unique_note_freq = number_unique_notes / len(notes)
    stats = {"note_sd": note_sd, "number_unique_notes": number_unique_notes, "unique_note_freq": unique_note_freq}

    return stats

selected_dir = "samples/music/motifs"
mid_stats_table = pd.DataFrame(columns=['note_sd', 'number_unique_notes', 'unique_note_freq'])

for filename in os.listdir(selected_dir):
    mid_file_name = selected_dir + "/" + filename
    mid = MidiFile(mid_file_name, clip=True)
    mid_stats = generate_midi_stats(mid)
    mid_stats_table = mid_stats_table.append(mid_stats, ignore_index=True)




