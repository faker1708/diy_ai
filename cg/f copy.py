import torch

n = 3
m = 4

# x = torch.normal(0,1,(n,1)).half().cuda()
# w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


x = torch.normal(0,1,(n,1)).half().cuda()
w = torch.normal(0,1,(m,n)).half().cuda()


x.requires_grad=True
w.requires_grad=True

y = torch.matmul(w,x)
ys = y.sum()


print(x.is_leaf)

ys.backward(retain_graph=True)
ys.backward(retain_graph=True)
ys.backward(retain_graph=True)


print(x.is_leaf)
print(x.grad)
print(w.grad)
