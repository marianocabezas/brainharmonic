import os
from mido import MidiFile
import numpy as np
from torch.utils.data.dataset import Dataset


class MotifDataset(Dataset):
    def __init__(
            self, paths=None, motif_size=64, notespbeat=12
    ):
        # Init
        if paths is None:
            paths = ['samples/music/jazz/', 'samples/music/classical/']
        self.motif_size = motif_size
        self.rolls = []
        min_len = self.motif_size + 1
        beat = 0
        for path in paths:
            files = sorted(os.listdir(path))
            for f in files:
                t = 0
                discard = False
                mpb_i = None
                note_found = False
                try:
                    mid_temp = MidiFile(os.path.join(path, f), clip=True)
                    notes = {
                        n: {'start': [], 'end': [], 'velocity': []}
                        for n in range(128)
                    }
                    tpb = mid_temp.ticks_per_beat
                    for track in mid_temp.tracks:
                        for msg in track:
                            if not msg.is_meta:
                                if note_found:
                                    t += msg.time
                                if msg.type == 'note_on':
                                    if not note_found:
                                        t = 0
                                        note_found = True
                                    beat = t // tpb
                                    if msg.velocity > 0:
                                        notes[msg.note]['start'].append(beat)
                                        notes[msg.note]['velocity'].append(
                                            msg.velocity
                                        )
                                    else:
                                        notes[msg.note]['end'].append(beat)
                                elif msg.type == 'note_off':
                                    beat = t // tpb
                                    notes[msg.note]['end'].append(beat)
                            else:
                                if msg.type == 'set_tempo':
                                    if mpb_i is None:
                                        mpb_i = msg.tempo
                                    else:
                                        discard = True
                                        break
                                elif msg.type == 'time_signature':
                                    num = msg.numerator
                                    den = msg.denominator
                                    if num != 4 and den != 4:
                                        discard = True
                                        break

                    if not discard:
                        piano_roll = np.zeros((128, beat))
                        for n, events in notes.items():
                            if len(events['start']) > 0:
                                for n_ini, n_end, v in zip(
                                        events['start'], events['end'],
                                        events['velocity']
                                ):
                                    # piano_roll[n, n_ini:n_end] = v / 127
                                    piano_roll[n, n_ini:n_end] = 1
                        max_notes = np.max(
                            np.sum(piano_roll, axis=0)
                        ).astype(int)
                        seq_len = piano_roll.shape[1]
                        if seq_len > min_len and max_notes < notespbeat:
                            self.rolls.append(piano_roll)
                except EOFError:
                    print('Unreadable', f)

        max_notes = [
            np.max(np.sum(roll, axis=0)).astype(int)
            for roll in self.rolls
        ]
        print(
            '{:d} piano rolls loaded with '
            '[{:02d}, {:02d}] - {:5.3f} ± {:5.3f}'.format(
                len(self.rolls), np.min(max_notes), np.max(max_notes),
                np.mean(max_notes), np.std(max_notes)
            )
        )
        # self.samples = [
        #     (
        #         roll_i,
        #         slice(
        #             ini, ini + self.motif_size
        #         ),
        #         slice(
        #             ini + self.motif_size, ini + self.motif_size + 1
        #         )
        #     )
        #     for roll_i, roll in enumerate(self.rolls)
        #     for ini in range(roll.shape[1] - 2 * self.motif_size)
        # ]

    def __getitem__(self, index):
        # roll_i, input_slice, output_idx = self.samples[index]

        # roll = self.rolls[roll_i]
        roll = self.rolls[index]
        max_ini = roll.shape[1] - self.motif_size - 1
        data_ini = np.random.randint(0, max_ini)
        target_ini = data_ini + self.motif_size
        # data = roll[:, input_slice].astype(np.float32)
        # target = roll[:, output_idx].astype(np.float32)
        data = roll[:, data_ini:target_ini].astype(np.float32)
        target = roll[:, target_ini:target_ini + 1].astype(np.float32)

        return data, target

    def __len__(self):
        # return len(self.samples)
        return len(self.rolls)
