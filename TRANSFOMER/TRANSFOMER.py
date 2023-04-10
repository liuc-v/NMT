import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Positional Encoding层，用于将序列的位置信息编码成向量
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个可训练的Positional Encoding矩阵pe
        pe = torch.zeros(max_len, d_model)
        # 创建一个与pe形状相同的位置矩阵position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建一个除数div_term，用于计算正弦和余弦值
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 根据位置矩阵和除数计算正弦和余弦值，并将它们赋值给PE矩阵
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将PE矩阵转置后，将其作为模型参数pe进行注册
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将输入x与pe相加，
        x = x + self.pe[:x.size(0), :]
        # 对结果进行dropout处理
        return self.dropout(x)


# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model, n_head, num_layers, dim_feedforward, device, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.device = device

        # 定义输入嵌入层
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(output_vocab_size, d_model)
        # 定义位置编码层
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 定义Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # 定义Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        # 定义输出全连接层
        self.fc = nn.Linear(d_model, output_vocab_size)

        # 初始化模型权重
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # 对嵌入层和全连接层的权重进行均匀分布初始化
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 对源语言序列进行嵌入和位置编码
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # 对源语言序列进行Transformer编码
        memory = self.transformer_encoder(src, src_mask)
        # 对目标语言序列进行嵌入和位置编码
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        # 对目标语言序列以及编码后的源语言序列进行Transformer解码
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)
        # 将解码结果通过全连接层输出
        output = self.fc(output)
        return output


# d_model: Transformer模型中embedding向量的维度，也是Positional Encoding矩阵中正弦余弦函数的维度，以及Transformer编码器和解码器中多头注意力机制中Q、K、V向量的维度。
# dropout: dropout概率，用于对Positional Encoding和解码器中的attention机制进行dropout处理。
# max_len: Positional Encoding矩阵的最大长度，用于确定矩阵的大小。
# input_vocab_size: 输入词汇表的大小。
# output_vocab_size: 输出词汇表的大小。
# n_head: 多头注意力机制中的头数。
# num_layers: Transformer编码器和解码器中的层数。
# dim_feedforward: Transformer中前向网络的隐藏层维度。
# 以下是变量的具体维度：
#
# pe: (max_len, d_model)
# position: (max_len, 1)
# div_term: (d_model / 2)
# x: (seq_len, batch_size, d_model)
# encoder_layer: 一个Transformer编码器层，无固定维度。
# transformer_encoder: 多层Transformer编码器，输入和输出的维度都是(seq_len, batch_size, d_model)。
# decoder_layer: 一个Transformer解码器层，无固定维度。
# transformer_decoder: 多层Transformer解码器，输入和输出的维度都是(seq_len, batch_size, d_model)。
# tgt: (tgt_seq_len, batch_size, d_model)
# memory: (src_seq_len, batch_size, d_model)，作为解码器的外部记忆使用。
# src_mask: (src_seq_len, src_seq_len)，用于在Transformer编码器中屏蔽padding符号。
# tgt_mask: (tgt_seq_len, tgt_seq_len)，用于在Transformer解码器中屏蔽未来符号以及padding符号。
# memory_mask: (tgt_seq_len, src_seq_len)，用于在解码器中屏蔽padding符号。
# output: (tgt_seq_len, batch_size, output_vocab_size)

# src_mask和tgt_mask都是用于在Transformer模型中屏蔽掉对应位置的padding符号和未来符号，具体作用如下：
#
# src_mask：在源语言序列编码过程中，由于不同句子的长度可能不同，需要将短句子的padding位置对应的编码屏蔽掉，避免它们对后续计算造成影响。
# tgt_mask：在目标语言序列解码过程中，为了避免模型在预测时使用未来符号，需要将当前时刻之后的符号全部屏蔽掉，避免解码器偷看未来信息。同时，同样需要将padding位置对应的编码屏蔽掉。
# 两者的维度都是二维的正方形矩阵，矩阵的大小由序列的长度决定，矩阵中对应位置的元素值为0或者-∞，其中0表示对应位置不需要屏蔽，-∞表示对应位置需要屏蔽。
