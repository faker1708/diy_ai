# %matplotlib inline
import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b

    avg = 0
    variance = 0.01

    y += torch.normal(avg,variance, y.shape)
    return X, y.reshape((-1, 1))



# lw = len(w)
num_examples = 3
lw = 4

X = torch.normal(0, 1, (num_examples, lw))

print(X)

