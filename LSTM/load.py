import os
import re


def load_model(parameter, model_dir=""):
    pattern = re.compile(rf"model_{parameter}_(\d+)\.pth")
    max_suffix = None
    max_file = None
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            suffix = int(match.group(1))
            if max_suffix is None or suffix > max_suffix:
                max_suffix = suffix
                max_file = filename
    if max_file:
        return model_dir + max_file
    else:
        return None


def load_model_eve(parameter, model_dir=""):  # 返回使用model
    pattern = re.compile(rf"model_{parameter}_(\d+)\.pth")
    max_suffix = None
    max_file = None
    results = []
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            results.append(filename)

    return results



def load_dict(hyperparameter):
    with open(hyperparameter+"re_word2index.txt", 'r') as f:
        content = f.readlines()
    dict1 = {}
    dict2 = {}
    for line in content:
        words = line.split()
        for word in words:
            if word not in dict1:
                dict1[word] = len(dict1)
            if word not in dict2:
                dict2[word] = len(dict2)
    return dict1, dict2


