# 定义模型
from torch import nn


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, device, emb_dim=512, d_model=512, num_heads=8, num_layers=6,
                 dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, emb_dim)

        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(d_model, trg_vocab_size)
        self.device = device

    def forward(self, src, trg):
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(self.device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(self.device)

        src_emb = self.src_embedding(src)
        trg_emb = self.trg_embedding(trg)

        out = self.transformer(src_emb, trg_emb, src_mask=src_mask, tgt_mask=trg_mask)

        return self.fc(out)
