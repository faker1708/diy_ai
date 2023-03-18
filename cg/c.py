import torch

print('练习矩阵乘法')


m  = 2
n = 4
k = 7

a = torch.rand(m,n)
print(a)


b = torch.rand(n,k)
print(b)

# a = a.float()
a = a.cuda()
print(a.dtype)

a = a.half()
print(a.dtype)

a = a.int()
print(a.dtype)
