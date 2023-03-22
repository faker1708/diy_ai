import torch

# 正态分布

# 均值，标准差
means = 1
std = 1


print('输出二维张量，矩阵')

c = 2
d = 3
x = torch.normal(means, std, (c,d))
print(x)

print('也可以输出更高维的张量')

c = 1
d = 3
e = 4
x = torch.normal(means, std, (c,d,4))
print(x)


