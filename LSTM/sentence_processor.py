import json
from collections import Counter

import jieba
import nltk
import torch


def load_data(file_path, max_sentence_num=100000000):
    english_sentences = []
    chinese_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < max_sentence_num:
                sentence = json.loads(line)
                sentence['english'] = sentence['english'].replace('\"', '"')
                sentence['chinese'] = sentence['chinese'].replace('\"', '"')
                english_sentences.append(["BOS"] + nltk.word_tokenize(sentence['english'].lower()) + ["EOS"])
                chinese_sentences.append(["BOS"] + list(jieba.cut(sentence['chinese'], cut_all=False)) + ["EOS"])
            else:
                break
    return english_sentences, chinese_sentences


def create_dict(sentences, max_word_num=100000000):   #
    # 统计文本中每个词出现的频数，并用出现次数最多的max_words个词创建词典，
    # 且在词典中加入'UNK'表示词典中未出现的词，'PAD'表示后续句子中添加的padding（保证每个batch中的句子等长）
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1

    most_common_words = word_count.most_common(max_word_num)  # 最常见的max_words个词
    total_words = len(most_common_words) + 2  # 总词量（+2：词典中添加了“UNK”和“PAD”）

    word2index = {}
    index2word = ["a"] * total_words

    for index, w in enumerate(most_common_words):
        word2index[w[0]] = index
        index2word[index] = w[0]
    word2index["UNK"] = total_words - 2
    word2index["PAD"] = total_words - 1
    index2word[total_words-2] = "UNK"
    index2word[total_words-1] = "PAD"

    return word2index, index2word


def translate(sentence, en_word2index, zh_index2word, model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sentence = ["BOS"] + nltk.word_tokenize(sentence.lower()) + ["EOS"]

    for i, word in enumerate(sentence):   # 替换未知词
        if word not in en_word2index:
            sentence[i] = "UNK"

    en_index = torch.tensor([[en_word2index[i] for i in sentence]], device=device)
    result = []
    encoder_hidden = model.encoder(en_index)
    start_index = [i for i, x in enumerate(zh_index2word) if x == "BOS"]
    decoder_input = torch.tensor([start_index], device=device)

    decoder_hidden = encoder_hidden
    while True:
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        pre = model.classifier(decoder_output)

        w_index = int(torch.argmax(pre, dim=-1))
        word = zh_index2word[w_index]

        if word == "EOS" or len(result) > 200:
            break

        result.append(word)
        decoder_input = torch.tensor([[w_index]], device=device)

    print("译文: ", "".join(result))
