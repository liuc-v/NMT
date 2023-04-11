def train(model, zh_corpus_len, optimizer, criterion, train_loader):
    loss_temp = []
    for en_index, zh_index in train_loader:
        output = model(en_index, zh_index)
        loss = criterion(output[:, 1:].reshape(-1, zh_corpus_len), zh_index[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # 将模型的参数梯度初始化为0

def valid(model, zh_corpus_len, optimizer, criterion, train_loader)
