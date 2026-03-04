# train.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tokenizer import MidiTokenizer
from model import TransformerSeq2Seq
import os

# 定义基础路径（原始字符串）
BASE_DIR = r"D:\Program -AI music"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

class MidiDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_notes, tgt_notes = self.pairs[idx]
        src_tokens = self.tokenizer.encode(src_notes, self.max_len)
        tgt_tokens = self.tokenizer.encode(tgt_notes, self.max_len)
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def train(task="MELODY2BRIDGE"):
    tokenizer = MidiTokenizer()
    with open(os.path.join(PROCESSED_DIR, f"{task}_pairs.pkl"), "rb") as f:
        pairs = pickle.load(f)

    dataset = MidiDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TransformerSeq2Seq(vocab_size=tokenizer.vocab_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_TOKEN)

    for epoch in range(3):
        for src, tgt in loader:
            src, tgt = src.cuda(), tgt.cuda()
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item()}")
        # 保存模型时（先确保输出目录存在）
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # 关键：创建outputs目录（若不存在）
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{task}_epoch{epoch}.pth"))

if __name__ == "__main__":
    # 直接调用 train 函数开始训练
    train("MELODY2BRIDGE")  # 可以根据需要修改任务类型