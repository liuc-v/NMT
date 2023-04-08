import os
import re


def load_model(parameter):
    pattern = re.compile(rf"model_{parameter}_(\d+)\.pth")
    max_suffix = None
    max_file = None
    for filename in os.listdir('./'):
        match = pattern.match(filename)
        if match:
            suffix = int(match.group(1))
            if max_suffix is None or suffix > max_suffix:
                max_suffix = suffix
                max_file = filename
    if max_file:
        return max_file
    else:
        return None


print(load_model("100_20000_0.0001_150_200_150_200"))

