import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def generate_masks(src, tgt, src_padding, trg_padding, device):
    src_mask = (src != src_padding).unsqueeze(-2).to(device)  # batch_size x 1 x seq_len_src
    tgt_mask = (tgt != trg_padding).unsqueeze(-2).to(device)  # batch_size x 1 x seq_len_tgt
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(device)  # batch_size x seq_len_tgt x seq_len_tgt
    memory_mask = src_mask.unsqueeze(1)  # batch_size x 1 x seq_len_src
    return src_mask, tgt_mask, memory_mask


def subsequent_mask(size):
    # "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent) == 0    # 返回上三角矩阵，并转为bool型tensor


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
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_padding, trg_padding):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)


        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.fc(output)
        return output


def generate_masks2(src, tgt, src_padding, trg_padding, device):
    print((src != src_padding).shape)
    src_mask = (src != src_padding).unsqueeze(-2)
    print(src_mask.shape)
    if tgt is not None:
        tgt_mask = (tgt != trg_padding).unsqueeze(-2)
        size = tgt.size(-1)
        tgt_mask = tgt_mask & torch.tril(torch.ones((size, size), device=device)).bool()
        memory_mask = src_mask.unsqueeze(1)
    else:
        tgt_mask = None
        memory_mask = None
    return src_mask, tgt_mask, memory_mask

# src = torch.randint(0, 10, [8, 40])
# tgt = torch.randint(0, 10, [8, 40])
#
# mask = src_mask, tgt_mask, memory_mask = generate_masks(src, tgt, src_padding=0, trg_padding=0,device="cpu")
# print(mask[0].shape)
# print(mask[1].shape)
# print(mask[2].shape)
# print(mask)
