import rtmidi
from pretty_midi import Note, PrettyMIDI, Instrument
from helpers import vectorize

class MidiInputError(Exception):
    pass

def read(midis, n_velocity_events=32, n_time_shift_events=125):

    note_sequence = []
    i = 0

    for m in midis:
        if m.instruments[0].program == 0:
            piano_data = m.instruments[0]
        else:
            raise PreprocessingError("Non-piano midi detected")
        note_sequence = self.apply_sustain(piano_data)
        note_sequence = sorted(note_sequence, key=lambda x: (x.start, x.pitch))
        note_sequences.append(note_sequence)



    live_notes = {}
    while i < len(midis):
        info , time_delta = midis[i]
        if i == 0:
            #start time tracking from zero
            time = 0
        else:
            #shift forward
            time = time + time_delta
        pitch = info[1]
        velocity = info[2]
        if velocity > 0:
            #(pitch (on), velocity, start_time (relative)
            live_notes.update({pitch: (velocity, time)})
            #how to preserve info ...?
        else:
            note_info = live_notes.get(pitch)
            if note_info is None:
                raise MidiInputError("what?")
            note_sequence.append(Note(pitch=pitch, velocity = note_info[0],
                start = note_info[1], end = time))
            live_notes.pop(pitch)

        i += 1

    note_sequence = quantize(note_sequence, n_velocity_events, n_time_shift_events)

    note_sequence = vectorize(note_sequence)
    return note_sequence

def quantize(note_sequence, n_velocity_events, n_time_shift_events):

    timestep = 1 / n_time_shift_events
    velocity_step = 128 // n_velocity_events

    for note in note_sequence:
        note.start = (note.start * n_time_shift_events) // 1 * timestep
        note.end = (note.end * n_time_shift_events) // 1 * timestep

        note.velocity = (note.velocity // velocity_step) * velocity_step + 1


    return note_sequence

if __name__ == "__main__":
    read()



