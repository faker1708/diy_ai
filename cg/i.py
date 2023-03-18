

import torch

print("bmm 练习")


# a = torch.tensor([[]])

a = torch.normal(0,1,(3,4,5))

print(a)

b = torch.normal(0,1,(3,5,6))


c = torch.bmm(a,b)

print(c)
print(c.shape)# 应该是 3 4 6


