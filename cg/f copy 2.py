import torch

n = 1+1
m = 1+1

# x = torch.normal(0,1,(n,1)).half().cuda()
# w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


true_w = torch.tensor([[3,4],[0,1]])
true_w = true_w.half().cuda()

print(true_w)


x = torch.normal(0,1,(n,1)).half().cuda()
w = torch.normal(0,1,(m,n)).half().cuda()

x[-1]=1


true_y = true_w @ x





w [-1]=0
w [-1,-1]=1





x.requires_grad=True
w.requires_grad=True





# y = torch.matmul(w,x)
y = torch.mm(w,x)


yy = y-true_y

loss = yy**2/2

print('y',y)

print('loss',loss)

ys = loss.sum()


# print(x.is_leaf)

ys.backward(retain_graph=True)


# print(x.is_leaf)
# print(x.grad)
# print(w.grad)

wg = w.grad
print(wg)