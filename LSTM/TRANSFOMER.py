import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (src_seq_len, batch_size)
        src = self.embedding(src)  # (src_seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # (src_seq_len, batch_size, d_model)
        return output


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # tgt: (tgt_seq_len, batch_size)
        # memory: (src_seq_len, batch_size, d_model)
        tgt = self.embedding(tgt)  # (tgt_seq_len, batch_size, d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)  # (tgt_seq_len, batch_size, d_model)
        return output


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(input_vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=dropout)
        self.decoder = Decoder(output_vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=dropout)
        self.fc = nn.Linear(d_model, output_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src: (src_seq_len, batch_size)
        # tgt: (tgt_seq_len, batch_size)
        memory = self.encoder(src)  # (src_seq_len, batch_size, d_model)
        output = self.decoder(tgt, memory)  # (tgt_seq_len, batch_size, d_model)
        output = self.fc(output)  # (tgt_seq_len, batch_size, output_vocab_size)
        output = self.softmax(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
