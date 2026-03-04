# generater_tracks.py
# -*- coding: utf-8 -*-
import torch
import pretty_midi
from tokenizer import MidiTokenizer
from model import TransformerSeq2Seq
import os

# === 路径配置 ===
BASE_DIR = r"D:\Program -AI music"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# === 你的外部输入 MIDI 文件 ===

def generate_from_midi(input_mid_path, task="MELODY2BRIDGE"):
    MODEL_PATH = os.path.join(BASE_DIR, "Transformer", f"{task}.pth")

    tokenizer = MidiTokenizer()

    # === 1) 读取输入 MIDI 并提取轨道 ===
    pm = pretty_midi.PrettyMIDI(input_mid_path)

    if not pm.instruments:
        raise ValueError("No instruments found in input MIDI.")
    
    melody_inst = pm.instruments[0]
    notes = sorted(melody_inst.notes, key=lambda n: n.start)

    # === 2) encode 成 tokens ===
    src_tokens = tokenizer.encode(notes)

    # === 3) 转成 tensor，batch_size=1 ===
    src = torch.tensor(src_tokens).unsqueeze(0).cuda()

    # === 4) 加载模型 ===
    model = TransformerSeq2Seq(vocab_size=tokenizer.vocab_size).cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # === 5) 贪心生成目标 tokens ===
    max_len = 512
    tgt_tokens = [tokenizer.SOS_TOKEN]
    for _ in range(max_len):
        tgt_input = torch.tensor(tgt_tokens).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = model(src, tgt_input)
        next_token = logits[0, -1].argmax(dim=-1).item()
        if next_token == tokenizer.EOS_TOKEN:
            break
        tgt_tokens.append(next_token)

    print("Generated tokens:", tgt_tokens)

    # === 6) decode 回 notes ===
    # 生成完成后，过滤SOS和EOS，再解码
    valid_tokens = [t for t in tgt_tokens if t not in [tokenizer.SOS_TOKEN, tokenizer.EOS_TOKEN]]
    if not valid_tokens:
        print("Warning: No valid tokens generated after filtering SOS/EOS.")
    else:
        print("Valid tokens (after filtering):", valid_tokens)

    # 解码有效token为音符
    generated_notes = tokenizer.decode(valid_tokens)

    # === 7) 重新打包输出 MIDI ===
    out_pm = pretty_midi.PrettyMIDI()

    # 保留原 MELODY
    melody_inst_out = pretty_midi.Instrument(program=0, name="MELODY")
    melody_inst_out.notes.extend(notes)
    out_pm.instruments.append(melody_inst_out)

    # 添加生成的 BRIDGE
    bridge_inst_out = pretty_midi.Instrument(program=0, name="BRIDGE")
    bridge_inst_out.notes.extend(generated_notes)
    out_pm.instruments.append(bridge_inst_out)

    # === 8) 保存输出文件 ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_mid = os.path.join(OUTPUT_DIR, f"{task}_generated_from_input.mid")
    out_pm.write(output_mid)

    print(f"Saved generated MIDI: {output_mid}")

     
# === 主流程封装函数 ===
def main_pipeline(melody_midi_path, sentiment_label, output_dir="outputs"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    generate_from_midi(melody_midi_path, "MELODY2BRIDGE")
    generate_from_midi(melody_midi_path, "MELODY2PIANO")

    print(f"All tracks have been generated and saved to the {output_dir} folder.")


if __name__ == "__main__":
    input_midi_full_path = os.path.join(OUTPUT_DIR, "input_vocal.mid")  # 核心改动：结合OUTPUT_DIR和文件名
    main_pipeline(
        melody_midi_path=input_midi_full_path,  # 主旋律MIDI路径
        sentiment_label="POSITIVE",           # 情感标签
        output_dir="outputs",                 # 输出文件夹
    )
