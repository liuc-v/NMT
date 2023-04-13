import torch
from torch import nn
# CE = torch.nn.CrossEntropyLoss()
#
# print(CE(torch.tensor([.5, 0.2, 0.3]), torch.tensor([0.5, 0.2, 0.3])))


# batch_first = True
# input = torch.randn(10000, 500000, 10)
# lstm = nn.LSTM(10, 20, 2, batch_first=True)
# h0 = torch.randn(2, 10000, 20)
# c0 = torch.randn(2, 10000, 20)
# output, (hn, cn) = lstm(input, (h0, c0))
#
# # batch_first = False
# input = torch.randn(3, 5, 10)
# lstm = nn.LSTM(10, 20, 2)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = lstm(input, (h0, c0))


# 探究CE包不包含softmax
# loss_func = torch.nn.CrossEntropyLoss()
# pre = torch.tensor([1000000, 0, 0, 0], dtype=torch.float)
# tgt = torch.tensor([2, 0, 0, 0], dtype=torch.float)
# print("手动计算:")
# print("1.softmax")
# print(torch.softmax(pre, dim=-1))
# print("2.取对数")
# print(torch.log(torch.softmax(pre, dim=-1)))
# print("3.与真实值相乘")
# print(-torch.sum(torch.mul(torch.log(torch.softmax(pre, dim=-1)), torch.softmax(tgt, dim=-1)), dim=-1))
#
# print()
# print("调用损失函数:")
# print(loss_func(pre, tgt))

# a = torch.rand(64, 40, 1178)
# b = a[:, 1:].reshape(-1, 1178)
# print(b.shape)
#
# c = torch.randint(0, 100, [64, 40])
# d = c[:, 1:].reshape(-1)
# print(d.shape)
# criterion = nn.CrossEntropyLoss()
# print(criterion(b, d))

# a = torch.tensor([3.0, 3.0, 100.0, 2.0])
# b = torch.tensor([0.0, 0.0, 1.0, 0.0])
# c = torch.tensor(2)
#
# print(a.shape)
# criterion = nn.CrossEntropyLoss()
# print(criterion(a, c))
# print(criterion(a, b))

# from torchtext.datasets import IWSLT2016
# train_iter, valid_iter, test_iter = IWSLT2016()
#
#
# src_sentence, tgt_sentence = next(iter(train_iter))
# a = 9

# from datasets import load_dataset
# from torch.utils.data import random_split
#
# torch.manual_seed(6)
#
# dataset = load_dataset("news_commentary", "en-zh")
# total_count = len(dataset['train'])
#
# from langdetect import detect
#
# text = "你好，世界"
# lang = detect("रोगाणुरोधी प्रतिरोध के ख़िलाफ़ निष्पक्ष लड़ाई")  # 'zh-cn'
# print(lang)
# print(total_count)
# for i in range(total_count):
#     if detect(dataset["train"][i]["translation"]["en"]) != 'en':
#         print(dataset["train"][i]["translation"]["en"])
#
#
# print(total_count)
# train_count = int(0.7 * total_count)
# valid_count = int(0.2 * total_count)
# test_count = total_count - train_count - valid_count
# train_dataset, valid_dataset, test_dataset = random_split(dataset['train'], [train_count, valid_count, test_count],
#                                                           torch.Generator())
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# for batch_index, sentence in enumerate(train_loader):
#     print(batch_index)
#     print(sentence)
#     # break  # 只打印第一个批次
# from torchtext.data import Field
#
# DE = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)
# EN = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)


# from datasets import load_dataset
# train_data, valid_data, test_data = load_dataset('multi30k', 'de')

# import torchtext
# from torchtext.datasets import TranslationDataset
# from torchtext.data import Field, BucketIterator
#
# # 定义源语言和目标语言的 Field
# src = Field(tokenize='spacy', tokenizer_language='en', init_token='<sos>', eos_token='<eos>', lower=True)
# tgt = Field(tokenize='moses', tokenizer_language='de', init_token='<sos>', eos_token='<eos>', lower=True)
#
# # 加载训练集、验证集、测试集
# train_data, valid_data, test_data = torchtext.datasets.WMT14(
#     root='data',
#     exts=('.en', '.de'),
#     fields=(src, tgt))

# 创建一个mask，形状为(10, 10)，其中对角线以下的元素为-inf，表示因果关系
mask = torch.triu(torch.ones(10, 10)) * -float('inf')
print(mask)
mask = mask.masked_fill(mask == 0, float(0.0))
print(mask)
