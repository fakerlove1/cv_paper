import numpy as np
import math


def get_data(batch_size):
    length = 101
    data = np.arange(length)
    print(data)
    for i in range(math.ceil(length / batch_size)):
        if batch_size * (i + 1) > length:
            yield data[batch_size * i:length], i
        else:
            yield data[batch_size * i:batch_size * (i + 1)], i


for i, idx in get_data(10):
    print(i, idx)
