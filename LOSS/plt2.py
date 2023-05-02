import matplotlib.pyplot as plt

# 从文件中读取数据
with open("ATTEN NEW 1000000_20000_0.0001_128_256_128_256.loss", "r") as f:
    lines1 = f.readlines()

epochs1 = []
losses1 = []


for i, line in enumerate(lines1):
    try:
        epoch, loss = line.split()
        epochs1.append(i)
        losses1.append(float(loss) / 43.0)
    except:
        pass

with open("atten test loss", "r") as f:
    lines2 = f.readlines()

epochs2 = []
losses2 = []

for i, line in enumerate(lines2):
    try:
        epoch, loss = line.split(':')
        print(epoch)
        epochs2.append(i * 20)
        losses2.append(float(loss) / 6.0 * 2)
    except:
        pass


# 绘制loss曲线
plt.plot(epochs1, losses1)
plt.plot(epochs2, losses2)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LSTM_ATTEN")
plt.show()