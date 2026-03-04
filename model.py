# model.py
import torch
import torch.nn as nn

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_enc[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_enc[:, :tgt.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        out = self.transformer(src_emb, tgt_emb)
        out = self.fc_out(out).permute(1, 0, 2)
        return out

    def generate(self, src, max_len=512, bos_token_id=1, eos_token_id=2):
        self.eval()
        src_emb = self.embedding(src) + self.pos_enc[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (S, N, E)
        memory = self.transformer.encoder(src_emb)

        ys = torch.ones(1, 1).fill_(bos_token_id).type_as(src)  # (N, T)
        for _ in range(max_len):
            tgt_emb = self.embedding(ys) + self.pos_enc[:, :ys.size(1), :]
            tgt_emb = tgt_emb.permute(1, 0, 2)
            out = self.transformer.decoder(tgt_emb, memory)
            out = self.fc_out(out)
            prob = out[-1, 0, :].softmax(dim=-1)
            next_token = torch.argmax(prob).unsqueeze(0).unsqueeze(0)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token == eos_token_id:
                break
        return ys
