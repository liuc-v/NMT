import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from load import load_model, load_model_eve
from sentence_processor import create_dict, translate, translate_2,load_data2, get_scores
from MyDataset import MyDataset
from LSTM import Encoder, Decoder, Seq2Seq
from train import train, valid


if __name__ == "__main__":
    f = open('param.txt', encoding='utf-8')
    s = f.readlines()[1]
    f.close()

    # 读取参数
    [sentence_nums, word_nums, lr, encoder_embed, encoder_hidden,
     decoder_embed, decoder_hidden, step_epoch, batch_size] = list(map(float, s.split(" ")))
    [sentence_nums, word_nums, encoder_embed, encoder_hidden,
     decoder_embed, decoder_hidden, step_epoch, batch_size] = list(map(int, [sentence_nums, word_nums,
                                                                 encoder_embed, encoder_hidden,
                                                                 decoder_embed, decoder_hidden,
                                                                 step_epoch, batch_size]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取从文件中读取句子
    en_data, zh_data = load_data2("../cmn_train.txt", sentence_nums)
    valid_en_data, valid_zh_data = load_data2("../cmn_valid.txt", max_sentence_num=1000000)

    en_word2index, en_index2word = create_dict(en_data, word_nums)
    zh_word2index, zh_index2word = create_dict(zh_data, word_nums)

    #  语料库的总长度
    en_corpus_len = len(en_word2index)
    zh_corpus_len = len(zh_word2index)

    epoch = 1000    # 训练次数
    model_dir = "../MODEL/LSTM/"

    dataset = MyDataset(en_data, zh_data, en_word2index, zh_word2index)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=dataset.batch_data_process)
    valid_dataset = MyDataset(valid_en_data, valid_zh_data, en_word2index, zh_word2index)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, collate_fn=valid_dataset.batch_data_process)

    # 一些超参数
    hyperparameter = [sentence_nums, word_nums, lr, encoder_embed, encoder_hidden,
                      decoder_embed, decoder_hidden]
    hyperparameter = list(map(str, hyperparameter))
    hyperparameter = "_".join(hyperparameter)

    file_index = open(hyperparameter + "re_word2index.txt", 'a', encoding='utf-8')
    file_index.write(str(en_word2index) + "\n" + str(zh_word2index) + "\n")
    file_index.close()

    criterion = nn.CrossEntropyLoss()

    loss_file = open(hyperparameter + ".loss", "a+", encoding='utf-8')
    loss_file.seek(0)
    start_epoch = len(loss_file.readlines())
    loss_file.close()
    print(1)
    model_names = load_model_eve(hyperparameter, model_dir=model_dir)
    print(1)
    for model_name in model_names:
        model = torch.load(model_dir + model_name).to(device)
        print(valid(model, valid_dataloader, criterion,zh_corpus_len))
        print(model_name)
