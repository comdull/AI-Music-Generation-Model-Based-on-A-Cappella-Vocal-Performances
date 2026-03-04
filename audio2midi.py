import librosa
import numpy as np
import pretty_midi


def audio_to_midi(audio_path, output_midi_path):
    # 1. 预处理：加载音频并标准化
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # 2. 分帧与加汉宁窗
    frame_length = 2048
    hop_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    window = np.hanning(frame_length)
    frames_windowed = frames * window.reshape(-1, 1)

    # 3. STFT提取频谱
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)

    # 4. 音高提取（使用pYIN算法）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )

    # 5. 节奏检测（Onset Detection）
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # 6. 音符量化与MIDI生成
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # 钢琴音色

    for time in onset_times:
        # 计算f0数组的索引
        f0_idx = int(time * sr / hop_length)
        # 修复：检查索引是否越界以及f0值是否有效
        if 0 <= f0_idx < len(f0) and not np.isnan(f0[f0_idx]):
            note_number = librosa.hz_to_midi(f0[f0_idx])
            # 量化到最近的八分之一音符
            quantized_time = round(time * 8) / 8
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(note_number),
                start=quantized_time,
                end=quantized_time + 0.5  # 假设每个音符持续半拍
            )
            instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    midi_data.write(output_midi_path)


# 示例调用 audio_to_midi("input_vocal.wav", "vocal.mid")
if __name__ == "__main__":
    # 测试音频转MIDI功能
    audio_to_midi("input_vocal.wav", "vocal.mid")
    print("Audio to MIDI conversion completed.")