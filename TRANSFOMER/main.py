import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from load import load_model
from sentence_processor import create_dict, translate, load_data, get_scores
from MyDataset import MyDataset
from LSTM import Encoder, Decoder, Seq2Seq


if __name__ == "__main__":
    f = open('param.txt', encoding='utf-8')
    s = f.readlines()[1]
    f.close()
    print(s)

    # 读取参数
    [sentence_nums, word_nums, lr, encoder_embed, encoder_hidden,
     decoder_embed, decoder_hidden, step_epoch] = list(map(float, s.split(" ")))
    [sentence_nums, word_nums, encoder_embed, encoder_hidden,
     decoder_embed, decoder_hidden, step_epoch] = list(map(int, [sentence_nums, word_nums,
                                                                 encoder_embed, encoder_hidden,
                                                                 decoder_embed, decoder_hidden,
                                                                 step_epoch]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取从文件中读取句子
    en_data, zh_data = load_data("../translation2019zh_train_part.json", sentence_nums)

    en_word2index, en_index2word = create_dict(en_data, word_nums)
    zh_word2index, zh_index2word = create_dict(zh_data, word_nums)

    # 保存字典
    file_index = open("re_word2index.txt", 'w', encoding='utf-8')
    file_index.write(str(en_word2index) + "\n" + str(zh_word2index))
    file_index.close()

    #  语料库的总长度
    en_corpus_len = len(en_word2index)
    zh_corpus_len = len(zh_word2index)

    batch_size = 64   # 一次喂多少数据
    epoch = 10000    # 训练次数

    dataset = MyDataset(en_data, zh_data, en_word2index, zh_word2index)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_process)

    # 一些超参数
    hyperparameter = [sentence_nums, word_nums, lr, encoder_embed, encoder_hidden,
                      decoder_embed, decoder_hidden]
    hyperparameter = list(map(str, hyperparameter))

    hyperparameter = "_".join(hyperparameter)
    if load_model(hyperparameter) is None:   # 之前没有存model
        print(hyperparameter)
        encoder = Encoder(en_corpus_len, encoder_embed, encoder_hidden, 1, 0.0).to(device)
        decoder = Decoder(zh_corpus_len, decoder_embed, decoder_hidden, 1, 0.0).to(device)
        model = Seq2Seq(encoder, decoder, device)
        model = model.to(device)
    else:   # 使用model继续
        print(11111111111111111111111)
        model_name = load_model(hyperparameter)
        print(model_name)
        model = torch.load(model_name)
        model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_file = open(hyperparameter + ".loss", "a+", encoding='utf-8')
    loss_file.seek(0)
    start_epoch = len(loss_file.readlines())
    loss_file.close()

    loss_temp = []
    # for e in range(epoch):
    #     print(start_epoch + e + 1)
    #     for en_index, zh_index in dataloader:
    #         output = model(en_index, zh_index)
    #         loss = criterion(output[:, 1:].reshape(-1, zh_corpus_len), zh_index[:, 1:].reshape(-1))
    #         loss.backward()
    #         opt.step()
    #         opt.zero_grad()  # 将模型的参数梯度初始化为0
    #     print('epoch=' + str(start_epoch+e+1) + " " + f"loss:{loss:.8f}")
    #     loss_temp.append('epoch=' + str(start_epoch + e + 1) + " " + f"loss:{loss:.8f}")
    #     if (e + 1) % step_epoch == 0:
    #         loss_file = open(hyperparameter + ".loss", "a+", encoding='utf-8')
    #         loss_file.write('\n'.join(loss_temp))
    #         loss_file.close()
    #         loss_temp = []
    #         torch.save(model, 'model_' + hyperparameter + '_' + str(e + 1 + start_epoch) + '.pth')  # 保存模型参数
    #         print(get_scores(en_data, zh_data, en_word2index, zh_index2word, model))

    print(get_scores(en_data, zh_data, en_word2index, zh_index2word, model))