# preprocess.py
# -*- coding: utf-8 -*-
import os
import pretty_midi
import pandas as pd
import pickle
from tokenizer import MidiTokenizer

DATASET_DIR = r"D:\Program -AI music\data\POP909"
OUTPUT_DIR = r"D:\Program -AI music\data\processed"
INDEX_XLSX = r"D:\Program -AI music\data\index.xlsx"

def extract_track_pairs(task):
    """
    task: MELODY2BRIDGE, MELODY2PIANO, BRIDGE2PIANO
    """
    tokenizer = MidiTokenizer()
    pairs = []

    index_list = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    for idx in index_list:
        midi_path = os.path.join(DATASET_DIR, idx, f"{idx}.mid")
        if not os.path.exists(midi_path):
            continue

        pm = pretty_midi.PrettyMIDI(midi_path)
        melody, bridge, piano = None, None, None

        for inst in pm.instruments:
            if inst.name.upper() == "MELODY":
                melody = inst
            elif inst.name.upper() == "BRIDGE":
                bridge = inst
            elif inst.name.upper() == "PIANO":
                piano = inst

        if task == "MELODY2BRIDGE" and melody and bridge:
            src, tgt = melody.notes, bridge.notes
        elif task == "MELODY2PIANO" and melody and piano:
            src, tgt = melody.notes, piano.notes
        elif task == "BRIDGE2PIANO" and bridge and piano:
            src, tgt = bridge.notes, piano.notes
        else:
            continue

        pairs.append((src, tgt))

    print(f"Extracted {len(pairs)} pairs for {task}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 使用变量和os.path.join构建路径
    output_path = os.path.join(OUTPUT_DIR, f"{task}_pairs.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(pairs, f)

if __name__ == "__main__":
    extract_track_pairs("MELODY2BRIDGE")
    extract_track_pairs("MELODY2PIANO")
    extract_track_pairs("BRIDGE2PIANO")