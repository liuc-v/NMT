import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input):
        # input size: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(input))
        # embedded size: (batch_size, seq_len, hidden_size)
        output, hidden = self.lstm(embedded)
        # output size: (batch_size, seq_len, hidden_size)
        # hidden size: (num_layers, batch_size, hidden_size)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, context):
        # input size: (batch_size, 1)
        # hidden size: (num_layers, batch_size, hidden_size)
        # context size: (batch_size, seq_len, hidden_size)
        embedded = self.dropout(self.embedding(input))
        # embedded size: (batch_size, 1, hidden_size)
        lstm_input = torch.cat((embedded, context), dim=2)
        # lstm_input size: (batch_size, 1, hidden_size+context_size)
        output, hidden = self.lstm(lstm_input, hidden)
        # output size: (batch_size, 1, hidden_size)
        # hidden size: (num_layers, batch_size, hidden_size)
        output = self.fc(output)
        # output size: (batch_size, 1, output_size)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        # input size: (batch_size, input_seq_len)
        # target size: (batch_size, target_seq_len)
        batch_size = input.size(0)
        target_seq_len = target.size(1)
        target_vocab_size = self.decoder.fc.out_features

        encoder_output, hidden = self.encoder(input)
        # encoder_output size: (batch_size, input_seq_len, hidden_size)
        # hidden size: (num_layers, batch_size, hidden_size)

        decoder_input = torch.tensor([[0]] * batch_size, dtype=torch.long, device=input.device)
        # decoder_input size: (batch_size, 1)
        context = torch.zeros(batch_size, 1, self.encoder.hidden_size, device=input.device)
        # context size: (batch_size, 1, hidden_size)

        outputs = torch.zeros(batch_size, target_seq_len, target_vocab_size, device=input.device)
        for i in range(target_seq_len):
            output, hidden = self.decoder(decoder_input, hidden, context)
            # output size: (batch_size, 1, target_vocab_size)
            outputs[:, i:i+1, :] = output
            decoder_input = target[:, i:i+1]
            context = encoder_output[:, i:i+1, :]

        return outputs