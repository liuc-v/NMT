import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_head, n_layers, dropout):
        super().__init__()

        # 定义输入和输出嵌入层
        self.embedding_src = nn.Embedding(input_dim, d_model)
        self.embedding_tgt = nn.Embedding(output_dim, d_model)

        # 定义Transformer编码器和解码器
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head, dropout), n_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_head, dropout), n_layers)

        # 定义输出层
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src_seq_len, N = src.shape
        tgt_seq_len, N = tgt.shape

        # 生成位置编码
        pos = torch.arange(0, src_seq_len).unsqueeze(1).repeat(1, N).to(device)

        # 输入嵌入
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)

        # 生成掩码
        src_padding_mask = self.generate_padding_mask(src, src)
        tgt_padding_mask = self.generate_padding_mask(tgt, tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(device)

        # Transformer编码器
        enc_output = self.encoder(src, src_key_padding_mask=src_padding_mask)

        # Transformer解码器
        dec_output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        # 输出层
        output = self.fc_out(dec_output)

        return output

    # 生成padding mask
    def generate_padding_mask(self, q, k):
        mask = (k.sum(dim=-1) == 0).unsqueeze(1).unsqueeze(2)
        return mask

    # 生成square subsequent mask
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        return mask

# 定义超参数
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TGT.vocab)
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.1

# 初始化模型和优化器
model = Transformer(INPUT_DIM, OUTPUT_DIM, D_MODEL, N_HEAD, N_LAYERS, DROPOUT)
optimizer = optim.Adam(model.parameters())

# 定义损失函数
criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

# 训练循环
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.trg

        optimizer.zero_grad()

        output = model(src, tgt[:-1])

        output_dim = output.shape[-1]

        output = output.reshape(-1, output_dim)
        tgt = tgt[1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 测试循环
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.trg

            output = model(src, tgt[:-1])

            output_dim = output.shape[-1]

            output = output.reshape(-1, output_dim)
            tgt = tgt[1:].reshape(-1)

            loss = criterion(output, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 训练模型
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
