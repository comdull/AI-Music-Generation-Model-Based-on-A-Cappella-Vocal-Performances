# tokenizer.py
# -*- coding: utf-8 -*-
import pretty_midi

class MidiTokenizer:
    def __init__(self):
        # === 基础配置 ===
        self.pitch_range = 128   # 0~127
        self.dur_bins = [0.05 * i for i in range(1, 65)]  # 64 bins: 0.05~3.2s
        self.vel_bins = [i * 4 for i in range(1, 33)]     # 32 bins: 4~128

        self.vocab_size = self.pitch_range + len(self.dur_bins) + len(self.vel_bins) + 4  # PAD, SOS, EOS, MASK

        # === 各类 offset ===
        self.PITCH_OFFSET = 0
        self.DUR_OFFSET = self.pitch_range
        self.VEL_OFFSET = self.DUR_OFFSET + len(self.dur_bins)

        self.SOS_TOKEN = self.vocab_size - 4
        self.EOS_TOKEN = self.vocab_size - 3
        self.PAD_TOKEN = self.vocab_size - 2
        self.MASK_TOKEN = self.vocab_size - 1

    # === PrettyMIDI → Events ===
    def midi_to_events(self, pm: pretty_midi.PrettyMIDI):
        if not pm.instruments:
            return []
        melody = pm.instruments[0]
        notes = sorted(melody.notes, key=lambda n: n.start)
        return notes

    # === Events → PrettyMIDI ===
    def events_to_midi(self, notes, program=0):
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program)
        inst.notes.extend(notes)
        pm.instruments.append(inst)
        return pm

    # === 单个 note → tokens ===
    def encode_note(self, note):
        pitch_token = note.pitch  # 0~127

        duration = note.end - note.start
        dur_token = min(range(len(self.dur_bins)), key=lambda i: abs(self.dur_bins[i] - duration))

        vel_token = min(range(len(self.vel_bins)), key=lambda i: abs(self.vel_bins[i] - note.velocity))

        return [
            pitch_token + self.PITCH_OFFSET,
            dur_token + self.DUR_OFFSET,
            vel_token + self.VEL_OFFSET
        ]

    # === notes → token 序列 ===
    def encode(self, notes, max_len=512):
        tokens = [self.SOS_TOKEN]
        for note in notes:
            tokens.extend(self.encode_note(note))
        tokens.append(self.EOS_TOKEN)

        if len(tokens) < max_len:
            tokens.extend([self.PAD_TOKEN] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]

        return tokens

    # === token 序列 → notes ===
    def decode(self, tokens, start_time=0.0):
        notes = []
        idx = 0
        current_time = start_time

        # 跳过 SOS
        if tokens[idx] == self.SOS_TOKEN:
            idx += 1

        while idx + 2 < len(tokens):
            # 如果遇到 EOS 或 PAD，提前停止
            if tokens[idx] in [self.EOS_TOKEN, self.PAD_TOKEN]:
                break

            pitch_token = tokens[idx]
            dur_token = tokens[idx + 1]
            vel_token = tokens[idx + 2]

            pitch = pitch_token - self.PITCH_OFFSET
            pitch = max(0, min(127, pitch))

            dur_idx = dur_token - self.DUR_OFFSET
            duration = self.dur_bins[dur_idx] if 0 <= dur_idx < len(self.dur_bins) else 0.1

            vel_idx = vel_token - self.VEL_OFFSET
            velocity = self.vel_bins[vel_idx] if 0 <= vel_idx < len(self.vel_bins) else 64
            velocity = max(0, min(127, velocity))

            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=current_time,
                end=current_time + duration
            )
            notes.append(note)
            current_time += duration

            idx += 3

        return notes
