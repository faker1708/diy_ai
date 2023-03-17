import torch


# a = 5
# b = 3
# x = torch.empty(a,b)
# print(x)

# x = torch.rand(5, 3)
# print(x)


# x = torch.tensor([5.5, 3])
# print(x)


# means = 0
# std = 1
# x = torch.normal(means,std,out = None)
# print(x)



# torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))

# a = 1
# b =2
# c = 3
# d =4
# x = torch.normal(a, b, (c, d))
# print(x)


a = 1
b = 1
c = 1
d = 3
x = torch.normal(a, b, (c, d))
print(x)


