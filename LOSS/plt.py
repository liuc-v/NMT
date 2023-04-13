import matplotlib.pyplot as plt

# 从文件中读取数据
with open("ATTEN NEW 1000000_20000_0.0001_128_256_128_256.loss", "r") as f:
    lines = f.readlines()

epochs = []
losses = []

# 解析每个epoch的loss值
for line in lines:
    try:
        epoch, loss = line.split()
        epochs.append(epoch)
        losses.append(float(loss))
    except:
        pass


# 绘制loss曲线
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LSTM")
plt.show()