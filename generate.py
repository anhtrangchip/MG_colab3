import argparse, uuid, subprocess
import torch
from model import MusicTransformer
from preprocess import SequenceEncoder
from preprocess import PreprocessingPipeline
from helpers import sample, write_midi
import pretty_midi
from pretty_midi import ControlChange
import six
from pretty_midi import Note, PrettyMIDI, Instrument
import numpy as np


class GeneratorError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--checkpoint", type=str, default="./saved_models/tf_04152021_e4",
                        help="Optional path to saved model, if none provided, the model is trained from scratch.")
    parser.add_argument("--sample_length", type=int, default=500,
                        help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float,
                        default=[1.0],
                        help="space-separated list of temperatures to use when sampling")
    parser.add_argument("--n_trials", type=int, default=3,
                        help="number of MIDI samples to generate per experiment")
    parser.add_argument("--live_input", action='store_true', default=False,
                        help="if true, take in a seed from a MIDI input controller")

    parser.add_argument("--play_live", action='store_true', default=False,
                        help="play sample(s) at end of script if true")
    parser.add_argument("--keep_ghosts", action='store_true', default=False)
    parser.add_argument("--stuck_note_duration", type=int, default=0)
    parser.add_argument("--condition_file", type=str, default="./data/2018/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi")

    args = parser.parse_args()

    # generate model
    sampling_rate = 125
    n_velocity_bins = 32
    seq_length = 1024
    n_tokens = 256 + sampling_rate + n_velocity_bins
    model = MusicTransformer(n_tokens, seq_length,
                             d_model=64, n_heads=8, d_feedforward=256,
                             depth=4, positional_encoding=True, relative_pos=True)
    if torch.cuda.is_available():
        model.cuda()
        print("GPU is available")
    else:
        print("GPU not available, CPU used")

    # load state
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        model.load_state_dict(state)
        print(f"Successfully loaded checkpoint at {args.checkpoint}")
    else:
        print(f"NOT FOUND checkpoint")

    n_velocity_events = 32
    n_time_shift_events = 125

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events,
                              min_events=0)

    pipeline = PreprocessingPipeline(input_dir="data", stretch_factors=[0.975, 1, 1.025],
                                     split_size=30, sampling_rate=sampling_rate, n_velocity_bins=n_velocity_bins,
                                     transpositions=range(-2, 3), training_val_split=0.9,
                                     max_encoded_length=seq_length + 1,
                                     min_encoded_length=257)

    if args.condition_file:
        pretty_midis = []
        print("Expecting a midi input...")
        # print(f"Parsing {len(midis)} midi files in {os.getcwd()}...")
        with open(args.condition_file, "rb") as f:
            try:
                midi_str = six.BytesIO(f.read())
                pretty_midis.append(pretty_midi.PrettyMIDI(midi_str))
                # print("Successfully parsed {}".format(m))
            except:
                print("Could not parse {}".format(m))
        print(pretty_midis)
        note_sequences = pipeline.get_note_sequences(pretty_midis)
        # print(note_sequence)
        note_sequence = note_sequences[0]
        note_sequence = vectorize(note_sequence)
        # note_sequence = midi_input.read(pretty_midis, n_velocity_events, n_time_shift_events)
        prime_sequence = decoder.encode_sequences([note_sequence])[0]

    else:
        prime_sequence = []

    temps = args.temps


    n_trials = args.n_trials

    keep_ghosts = args.keep_ghosts
    stuck_note_duration = None if args.stuck_note_duration == 0 else args.stuck_note_duration

    for temp in temps:
        print(f"sampling temp={temp}")
        note_sequence = []
        for i in range(n_trials):
            print("generating sequence")
            output_sequence = sample(model, prime_sequence=prime_sequence, sample_length=args.sample_length,
                                     temperature=temp)
            note_sequence = decoder.decode_sequence(output_sequence,
                                                    verbose=True, stuck_note_duration=stuck_note_duration)

            output_dir = f"output/"
            file_name = f"sample{i + 1}_{temp}"
            write_midi(note_sequence, output_dir, file_name)

    for temp in temps:
        try:
            subprocess.run(['timidity', f"output/sample{i + 1}_{temp}.midi"])
        except KeyboardInterrupt:
            continue


def quantize(note_sequence, n_velocity_events, n_time_shift_events):

    timestep = 1 / n_time_shift_events
    velocity_step = 128 // n_velocity_events

    for note in note_sequence:
        note.start = (note.start * n_time_shift_events) // 1 * timestep
        note.end = (note.end * n_time_shift_events) // 1 * timestep

        note.velocity = (note.velocity // velocity_step) * velocity_step + 1


    return note_sequence
def vectorize(sequence):
    """
    Converts a list of pretty_midi Note objects into a numpy array of
    dimension (n_notes x 4)
    """
    print(sequence[0])
    array = [[note.start, note.end, note.pitch, note.velocity] for
            note in sequence]
    return np.asarray(array)

if __name__ == "__main__":
    main()