import matplotlib.pyplot as plt

# 从文件中读取数据
with open("100_20000_0.0001_150_200_150_200.loss", "r") as f:
    lines = f.readlines()

epochs = []
losses = []

# 解析每个epoch的loss值
for line in lines:
    try:
        epoch, loss = line.strip().split()
        epochs.append(int(epoch.split("=")[-1]))
        losses.append(float(loss.split(":")[-1]))
    except:
        pass


# 绘制loss曲线
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("RNN 1000")
plt.show()