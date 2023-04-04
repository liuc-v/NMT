import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_size, embedding_dim):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(self.num_layers, self.batch_size, self.hidden_size).zero_().cuda())
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout, batch_size, embedding_dim):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded, hidden)
        output = self.output_layer(output.squeeze(1))
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = input.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.output_layer.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        encoder_output, hidden = self.encoder(input)

        decoder_input = target[:, 0]
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_output)
            outputs[:, t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1

        return outputs
