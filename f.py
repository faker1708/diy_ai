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


def loss_f(y,true_y):
    # print('输出一个标量')
    yy = y-true_y

    loss_tensor = yy**2/2

    # print('y',y)

    # print('loss',loss_tensor)

    loss = loss_tensor.sum()

    return loss

epoch = 3
lr = 0.01

for i in range(epoch):



    y = torch.mm(w,x)
    

    loss = loss_f(y,true_y)

    loss.backward(retain_graph=True)

    with torch.no_grad():
        w -= lr * w.grad 
        w.grad.zero_()

    print(w)