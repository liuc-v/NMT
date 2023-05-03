import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from TRANSFOMER_bak import TransformerModel
from TRANSFOMER import Transformer
from load import load_model
from sentence_processor import create_dict, translate, get_scores, load_data2, translate_transfomer
from MyDataset import MyDataset
from train import train, valid

if __name__ == "__main__":
    f = open('param.txt', encoding='utf-8')
    s = f.readlines()[1]
    f.close()

    model_dir = "../MODEL/TRANSFOMER/"

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

    en_word2index, en_index2word = create_dict(en_data, word_nums)
    zh_word2index, zh_index2word = create_dict(zh_data, word_nums)

    #  语料库的总长度
    en_corpus_len = len(en_word2index)
    zh_corpus_len = len(zh_word2index)

    epoch = 10000    # 训练次数

    dataset = MyDataset(en_data, zh_data, en_word2index, zh_word2index)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=dataset.batch_data_process)

    # 一些超参数
    hyperparameter = [sentence_nums, word_nums, lr, encoder_embed, encoder_hidden,
                      decoder_embed, decoder_hidden]
    hyperparameter = list(map(str, hyperparameter))
    hyperparameter = "_".join(hyperparameter)

    file_index = open(hyperparameter + "re_word2index.txt", 'a', encoding='utf-8')
    file_index.write(str(en_word2index) + "\n" + str(zh_word2index) + "\n")
    file_index.close()

    print(hyperparameter)

    if load_model(hyperparameter, model_dir="../MODEL/TRANSFOMER/") is None:   # 之前没有存model
        print("未找到历史模型，重新构建TRansfomer模型")
        model = Transformer(en_corpus_len, zh_corpus_len, "cuda", 128, 128)
        model = model.to(device)
        # # 保存字典        # !!!!!
        # file_index = open(hyperparameter + "re_word2index.txt", 'w', encoding='utf-8')
        # file_index.write(str(en_word2index) + "\n" + str(zh_word2index))
        # file_index.close()
    else:   # 使用model继续
        model_name = load_model(hyperparameter, model_dir='../MODEL/TRANSFOMER/')
        print("加载历史模型:" + model_name)
        model = torch.load(model_name)
        model = model.to(device)
        # 从文件中读取单词编码表
     #   en_word2index, en_index2word, zh_word2index, zh_index2word =

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_file = open(hyperparameter + ".loss", "a+", encoding='utf-8')
    loss_file.seek(0)
    start_epoch = len(loss_file.readlines())
    loss_file.close()

    loss_temp = []
    best_loss = torch.tensor(np.Inf)
    patience = 10  # patience轮不降低则自动停止计算
    for e in range(epoch):
        train_loss = train(model, dataloader, criterion, opt, zh_corpus_len)
        # if valid_loss.item() < best_loss.item():
        #     best_loss = valid_loss
        #     patience = 10
        # else:
        #     patience -= 1
        #     if patience < 0:
        #         break
        loss_temp.append(str(e) + " " + str(train_loss.item()))
        print(str(e) + " " + str(train_loss.item()))

        if (e + 1) % step_epoch == 0:
            loss_file = open(hyperparameter + ".loss", "a+", encoding='utf-8')
            loss_file.write('\n'.join(loss_temp))
            loss_file.write('\n')
            loss_file.close()
            loss_temp = []
            torch.save(model, model_dir + 'model_' + hyperparameter + '_' + str(e + 1 + start_epoch) + '.pth')  # 保存模型参数


    while True:
        sentence = input()
        sentence = ["BOS"] + nltk.word_tokenize(sentence.lower()) + ["EOS"]
        print(translate_tra
        nsfomer(sentence, en_word2index, zh_index2word, model))
