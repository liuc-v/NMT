import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim=100, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return output, (hidden, cell)

    def init_hidden_cell(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(3 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_output):
        # Compute attention score
        attention_score = self.v(torch.tanh(self.W(torch.cat((decoder_hidden.expand(encoder_output.size(1), -1, -1).transpose(0, 1), encoder_output), dim=-1))))

        # Compute attention weights
        attention_weights = torch.softmax(attention_score, dim=1)

        # Compute context vector
        context_vector = torch.sum(attention_weights * encoder_output, dim=1)

        return context_vector


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_dim=100, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.attention = Attention(hidden_size)  # 添加attention机制
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_seq))
        context_vector = self.attention(hidden[-1], encoder_outputs)  # 计算context vector 和注意力权重
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)  # 连接embedding和context vector
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        fc_output = self.fc(output.squeeze(1))
        return fc_output, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        hidden_cell = self.encoder.init_hidden_cell(batch_size)   # 初试化
        hidden_cell = tuple(t.to(self.device) for t in hidden_cell)   # 转移设备
        encoder_output, (hidden, cell) = self.encoder(input_seq, hidden_cell[0], hidden_cell[1])
        input_seq = target_seq[:, 0]
        for t in range(1, target_len):
            output, (hidden, cell) = self.decoder(input_seq.unsqueeze(1), hidden, cell, encoder_output)
            outputs[:, t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_seq = target_seq[:, t] if teacher_force else top1
        return outputs
