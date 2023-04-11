import opencc
converter = opencc.OpenCC('t2s.json')

results = []
with open("cmn.txt", "r", encoding="utf-8") as f:
    # 逐行读取文件内容
    for line in f:
        # 去掉行尾的换行符
        line = line.strip()
        # 用tab分隔字符串，得到一个列表
        data = line.split("\t")
        # 检查列表长度是否为3，如果不是，跳过这一行
        if len(data) != 3:
            continue
        # 将列表中的三个字符串分别赋值给变量
        a, b, c = data
        b = converter.convert(b)  # 漢字
        results.append(a + '\t' + b + '\n')
        # 打印或处理这三个变量

import random

def split(full_list, shuffle=False, ratio=0.2):
    # 根据给定的比例分割一个列表
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


sublist_1, sublist_2 = split(results, shuffle=True, ratio=0.7) # 按照 0.2 的比例分割
print(len(sublist_1))
print(len(sublist_2))
sublist_3, sublist_4 = split(sublist_2, shuffle=True, ratio=0.66666) # 按照 0.375 的比例分割
print(len(sublist_3))
print(len(sublist_4))

with open("cmn_train.txt", 'w', encoding='utf-8') as f:
    for a in sublist_1:
        f.write(a)

with open("cmn_valid.txt", 'w', encoding='utf-8') as f:
    for a in sublist_3:
        f.write(a)

with open("cmn_test.txt", 'w', encoding='utf-8') as f:
    for a in sublist_4:
        f.write(a)
