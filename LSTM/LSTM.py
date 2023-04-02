import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim   # 词汇表大小
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch_size]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        batch_size = trg.shape[1]
        print(batch_size)
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features
        print(src)
        print(src.shape)
        print(trg.shape)
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        print(outputs.shape)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        print(input.shape)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
