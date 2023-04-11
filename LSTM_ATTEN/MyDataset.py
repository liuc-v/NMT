import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, en_data, zh_data, en_word2index, zh_word2index):
        self.en_data = en_data
        self.zh_data = zh_data
        self.en_word2index = en_word2index
        self.zh_word2index = zh_word2index

    def __getitem__(self, index):
        en_sentence = self.en_data[index]
        zh_sentence = self.zh_data[index]

        # 替换英文句子词汇表中没有的词
        for i, word in enumerate(en_sentence):
            if word not in self.en_word2index:
                en_sentence[i] = "UNK"
        # 替换中文句子词汇表中没有的词
        for i, word in enumerate(zh_sentence):
            if word not in self.zh_word2index:
                zh_sentence[i] = "UNK"

        en_index = [self.en_word2index[i] for i in en_sentence]
        zh_index = [self.zh_word2index[i] for i in zh_sentence]

        return en_index, zh_index

    def batch_data_process(self, batch_datas):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        en_index, zh_index = [], []
        en_len, zh_len = [], []

        for en, ch in batch_datas:
            en_index.append(en)
            zh_index.append(ch)
            en_len.append(len(en))
            zh_len.append(len(ch))

        max_en_len = max(en_len)
        max_zh_len = max(zh_len)

        for en in en_index:
            for i in range(max_en_len - len(en)):
                en.append(self.en_word2index["PAD"])

        for zh in zh_index:
            for i in range(max_zh_len - len(zh)):
                zh.append(self.zh_word2index["PAD"])

        en_index = torch.tensor(en_index, device=device)
        zh_index = torch.tensor(zh_index, device=device)

        return en_index, zh_index

    def __len__(self):
        assert len(self.en_data) == len(self.zh_data)
        return len(self.zh_data)
