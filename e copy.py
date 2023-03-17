import torch
# x = torch.tensor(1.0, requires_grad=True)
# y = torch.tensor(2.0, requires_grad=True)
# z = x**2+y
# z.backward()
# print(z, x.grad, y.grad)

# # >>> tensor(3., grad_fn=<AddBackward0>) tensor(2.) tensor(1.)

n = 1+1
m = 1+1

x = torch.normal(0,1,(n,1),requires_grad=True)
x[-1]=1


w = torch.normal(0,1,(m,n),requires_grad=True)
w [-1]=0
w[-1,-1]=1


rl = torch.nn.ReLU(inplace=False)


y = torch.mm(w,x)
y = rl(y)
ys = y.sum()
# ys.backward()
print(y)
print(ys)
# print(x.grad)
# print(w.grad)


print('减')
# y = y[:-1]
# ys = y.sum()

# ys = ys-1
ys.backward(retain_graph=True)
ys.backward(retain_graph=True)
ys.backward(retain_graph=True)

print(y)
print(ys)
print(x.grad)
print(w.grad)
