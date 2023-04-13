import torch


def train(model, dataloader, criterion, opt, zh_corpus_len):
    loss_file = open('loss_first', 'a+', encoding='utf-8')
    epoch_loss = 0.0
    total_batches = len(dataloader) / dataloader.batch_size
    calcu_batches = 0
    for en_index, zh_index in dataloader:
        output = model(en_index, zh_index)
        loss = criterion(output[:, 1:].reshape(-1, zh_corpus_len), zh_index[:, 1:].reshape(-1))
        loss.backward()
        epoch_loss += loss
        opt.step()
        opt.zero_grad()  # 将模型的参数梯度初始化为0
        calcu_batches += 1
        loss_file.write(str(loss.item()) + " " + str(calcu_batches) + " / " + str(total_batches) + "\n")
    loss_file.write("epoch_loss:" + str(epoch_loss) + "\n")
    return epoch_loss


def valid(model, valid_dataloader, criterion,  zh_corpus_len):
    epoch_loss = 0.0
    for en_index, zh_index in valid_dataloader:
        output = model(en_index, zh_index)
        with torch.no_grad():
            loss = criterion(output[:, 1:].reshape(-1, zh_corpus_len), zh_index[:, 1:].reshape(-1))
            epoch_loss += loss
    return epoch_loss
