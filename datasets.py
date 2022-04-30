import os
from mido import MidiFile
import numpy as np
from torch.utils.data.dataset import Dataset


def shift_tokens(shift, bits=8, notes=32):
    remainder = shift
    tokens = []
    for shift_bit in range(bits - 1, -1, -1):
        token = np.zeros(2 * notes + bits)
        token[2 * notes + shift_bit] = 1
        n_tokens = remainder // 2 ** shift_bit
        remainder = remainder - n_tokens * 2 ** shift_bit
        tokens += n_tokens * [token]

    return tokens


def on_tokens(note_list, bits=8, notes=32):
    tokens = []
    for note in note_list:
        token = np.zeros(2 * notes + bits)
        token[note] = 1
        tokens.append(token)

    return tokens


def off_tokens(note_list, bits=8, notes=32):
    tokens = []
    for note in note_list:
        token = np.zeros(2 * notes + bits)
        token[notes + note] = 1
        tokens.append(token)

    return tokens


def roll_to_state(roll, bits=8):
    notes = len(roll)
    shift = 0
    notes_active = np.array([])
    token_list = []
    for beat in roll.transpose():
        if np.sum(beat) > 0:
            played_notes = np.where(beat > 0)[0]
            new_replayed = np.isin(played_notes, notes_active)
            old_replayed = np.isin(notes_active, played_notes)
            if new_replayed.all() and old_replayed.all():
                # We are repeating everything
                # print('Repeat', notes_active, played_notes, shift)
                pass
            else:
                # Changes
                on = played_notes[np.logical_not(new_replayed)]
                off = notes_active[np.logical_not(old_replayed)]
                # print(
                #     'OFF', off, 'ON', on, 'Data', notes_active, old_replayed,
                #     played_notes, new_replayed, shift
                # )
                token_list += shift_tokens(shift, bits, notes)
                token_list += off_tokens(off, bits, notes)
                token_list += on_tokens(on, bits, notes)
                shift = 0
            notes_active = played_notes
        else:
            if len(notes_active) == 0:
                # Silence
                # print('Silence', shift)
                pass
            else:
                # Notes go off
                # print('OFF', notes_active, shift)
                token_list += shift_tokens(shift, bits, notes)
                token_list += off_tokens(notes_active, bits, notes)
                notes_active = np.array([])
                shift = 0

        shift += 1

    token_list += shift_tokens(shift, bits, notes)

    return np.stack(token_list, axis=1)


def state_to_roll(states, bits=8, notes=32):
    roll = []
    active_notes = []
    for state in states.transpose():
        state_code = np.where(state)[0][0]
        if state_code < notes:
            # print('ON', state_code)
            active_notes.append(state_code)
        elif state_code < (2 * notes):
            # print('OFF', state_code - notes)
            active_notes.remove(state_code - notes)
        else:
            # print('Shift', 2 ** (state_code - 2 * notes))
            shift = 2 ** (state_code - 2 * notes)
            beat = np.zeros(notes)
            for note in active_notes:
                beat[note] = 1
            roll += shift * [beat]

    return np.stack(roll, axis=1)


class MotifDataset(Dataset):
    def __init__(
            self, paths=None, motif_size=64, notespbeat=12,
            multitokens=True
    ):
        # Init
        if paths is None:
            paths = ['samples/music/jazz/', 'samples/music/classical/']
        self.multitokens = multitokens
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
        if not self.multitokens:
            roll = roll_to_state(roll)
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
