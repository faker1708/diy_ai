import torch


n = 3+1
m = 2+1

x = torch.normal(0,1,(n,1))
x = x.half()
# print(x)

# x[-1]=1
# print(x)


w = torch.normal(0,1,(m,n))
w = w.half()

# w [-1]=0
# w [-1,-1]=1




x.requires_grad = True
x = x.cuda()



w.requires_grad = True
w = w.cuda()


y = torch.mm(w,x)
y.sum().backward()



print(x)
print(w)

# print(x.grad)
# print(w.grad)