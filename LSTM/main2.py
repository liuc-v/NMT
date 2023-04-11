import nltk
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
    model_dir = "../MODEL/LSTM 0"

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

    # if load_model(hyperparameter, model_dir=model_dir) is None:   # 之前没有存model
    #     print("未找到历史模型，重新构建LSTM模型")
    #     encoder = Encoder(en_corpus_len, encoder_embed, encoder_hidden, 2, 0.0).to(device)
    #     decoder = Decoder(zh_corpus_len, decoder_embed, decoder_hidden, 2, 0.0).to(device)
    #     model = Seq2Seq(encoder, decoder, device)
    #     model = model.to(device)
    # else:   # 使用model继续
    #     model_name = load_model(hyperparameter, model_dir=model_dir)
    #     print("加载历史模型:" + model_name)
    #     model = torch.load(model_name)
    #     model = model.to(device)

   # opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_file = open(hyperparameter + ".loss", "a+", encoding='utf-8')
    loss_file.seek(0)
    start_epoch = len(loss_file.readlines())
    loss_file.close()

    loss_temp = []
    models = load_model_eve(hyperparameter, model_dir=model_dir)
    print(models)
    for model_name in models:
        model = torch.load(model_dir + model_name)
        # train_loss = train(model, dataloader, criterion, opt, zh_corpus_len)
        # loss_temp.append(train_loss.item())
        valid_loss = valid(model, valid_dataloader, criterion, zh_corpus_len)

        print("valid_loss:" + str(valid_loss.item()))
