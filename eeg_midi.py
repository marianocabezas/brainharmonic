import mido
from audiolazy import lazy_midi

# This is a framework for easily using mido.
# Create a file:
# midi_file = eeg_midi.File(tickspeed, bpm, collapse_tracks=False)

# Add a track to a file:
# track = midi_file.add_track(track_name)

# Add notes to a track:
# track.add_note(note, time, length, velocity, pitch)
# See the comments on the Note object for details on what inputs it accepts.
# You can add notes at any time - they don't have to be sequential.

# Save a completed file:
# midi_file.save("filepath/filename.mid")


class File(object):
    def __init__(self, tickspeed, bpm=60, collapse_tracks=False):
        self.tickspeed = tickspeed  # ticks per beat. It's convenient to synchronise this with the EEG sample rate.
        self.bpm = bpm  # beats per minute.
        self.tracks = []
        self.collapse_tracks = collapse_tracks  # whether the final output file has all tracks collapsed into one.

    def add_track(self, name):
        track = Track(name=name, bpm=self.bpm, tickspeed=self.tickspeed)
        self.tracks.append(track)

        return track

    def to_mido(self):
        midi_file = mido.MidiFile(ticks_per_beat=self.tickspeed)

        if self.collapse_tracks:
            midi_file.tracks.append(mido.merge_tracks([track.to_mido() for track in self.tracks]))
        else:
            for track in self.tracks:
                midi_file.tracks.append(track.to_mido())

        return midi_file

    def save(self, filename):
        self.to_mido().save(filename)


class Track(object):
    def __init__(self, name, bpm, tickspeed):
        self.notes = []
        self.bpm = bpm
        self.tickspeed = tickspeed
        self.name = name

    def add_note(self, note, time, length, velocity, pitch=0, note_type="int"):
        self.notes.append(Note(note=note,
                               time=time,
                               velocity=velocity,
                               length=length,
                               tickspeed=self.tickspeed,
                               pitch=pitch,
                               note_type=note_type))

    def to_mido(self):
        track = mido.MidiTrack()
        track.name = self.name

        messages = []
        for note in self.notes:
            messages.append(note.pitch_message())
            messages.append(note.on_message())
            messages.append(note.off_message())

        messages.sort(key=lambda m: m.time, reverse=True)  # sort messages in increasing time order

        previous_pitch = 0
        previous_time = 0
        for message in messages:

            # clean any unnecessary pitchwheel messages, and messages with negative time values
            if message.type == 'pitchwheel':
                if message.pitch == previous_pitch:
                    messages.remove(message)
                    continue
                else:
                    previous_pitch = message.pitch
            if message.time < 0:
                messages.remove(message)
                continue

            # MIDI uses the delta time since the last message rather than absolute time, so we convert:
            delta_time = message.time - previous_time

            # MIDI can't support multiple messages in the same tick, so we push some messages forward a tick.
            # this can cause weirdness if there are many simultaneous messages, but works okay otherwise.
            if delta_time <= 0:
                delta_time = 1
            else:
                previous_time = message.time

            message.time = delta_time

            track.append(message)
        return track


class Note(object):
    def __init__(self, note, time, tickspeed, length=1, velocity=64, pitch=0, note_type="int"):
        self.note = note  # musical note. reads string, int and float differently, beware data type!
        self.time = time  # tick number since the start of the track. Integer.
        self.velocity = velocity  # 'loudness' value, integer 1-128.
        self.pitch = pitch  # pitch bend, integer -8192 to 8192 corresponding to +-2 semitones
        self.tick_length = int(length * tickspeed)  # length is in beats, eg. a crotchet is 1, quaver is 0.5
        self.note_type = note_type

        # read strings as "Note-(Modifier)-Octave", eg. "F#4", "C6", "Bb2"
        if self.note_type == "str":
            self.parsed_note = lazy_midi.str2midi(self.note)
        # read floats as frequencies, rounded to the nearest semitone
        if self.note_type == "float":
            self.parsed_note = int(round(lazy_midi.freq2midi(self.note)))
        # read integers as midi note numbers directly
        if self.note_type == "int":
            self.parsed_note = self.note

    def pitch_message(self):
        return mido.Message('pitchwheel',
                            channel=1,
                            pitch=self.pitch,
                            time=self.time - 1)  # we bend the pitch just before starting the new note

    def on_message(self):
        return mido.Message('note_on',
                            channel=1,
                            note=self.parsed_note,
                            velocity=self.velocity,
                            time=self.time)

    def off_message(self):
        return mido.Message('note_off',
                            channel=1,
                            note=self.parsed_note,
                            velocity=self.velocity,
                            time=self.time + self.tick_length)
