import torch
import torch.nn as nn

# 定义Encoder类，继承自nn.Module
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, inputs, hidden):
        # inputs: (batch_size, seq_len, input_size)
        # hidden: (num_layers*num_directions, batch_size, hidden_size)
        outputs, hidden = self.lstm(inputs, hidden)
        # outputs: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers*num_directions, batch_size, hidden_size)
        return outputs, hidden

    def init_hidden(self, batch_size):
        # 初始化hidden状态
        return torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(self.num_layers, batch_size, self.hidden_size)


# 定义Decoder类，继承自nn.Module
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        # 定义输出层，将LSTM的输出转化为输出大小
        self.out = nn.Linear(hidden_size, output_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, hidden):
        # inputs: (batch_size, 1, input_size)
        # hidden: (num_layers*num_directions, batch_size, hidden_size)
        outputs, hidden = self.lstm(inputs, hidden)
        # outputs: (batch_size, 1, hidden_size)
        outputs = self.softmax(self.out(outputs[:, :, :]))
        # outputs: (batch_size, 1, output_size)
        return outputs, hidden

    def init_hidden(self, batch_size):
        # 初始化hidden状态
        return torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(self.num_layers, batch_size, self.hidden_size)


# 定义Seq2Seq类，继承自nn.Module
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, output_size, num_layers, dropout)

    def forward(self, inputs, targets, teacher_forcing_ratio=0.5):
        # inputs: (batch_size, seq_len, input_size)
        # targets: (batch_size, seq_len, output_size)
        batch_size = inputs.size(0)
        max_len = targets.size(1)
        # 初始化hidden状态
        encoder_hidden = self.encoder.init_hidden(batch_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden)
        # decoder_outputs: (batch_size, seq_len, output_size)
        decoder_outputs = torch.zeros(batch_size, max_len, self.decoder.output_size)
        decoder_hidden = encoder_hidden
        # 取出第一个时刻的decoder输入
        decoder_input = targets[:, 0, :].unsqueeze(1)
        for t in range(1, max_len):
            # decoder_output: (batch_size, 1, output_size)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output[:, :, :]
            # 根据teacher forcing ratio决定是使用当前时刻的预测结果，还是使用真实的输出序列
            use_teacher_forcing = True if torch.rand(1) < teacher_forcing_ratio else False
            if use_teacher_forcing:
                # 直接使用真实输出序列作为下一时刻的输入
                decoder_input = targets[:, t, :].unsqueeze(1)
            else:
                # 使用当前时刻的预测结果作为下一时刻的输入
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach().unsqueeze(1)
        return decoder_outputs
