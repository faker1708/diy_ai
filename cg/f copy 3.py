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

epoch = 1
lr = 0.01

for i in range(epoch):
    # print('wl',w.is_leaf)

    print('\n\n')
    print('epoch',i)
    # print('wl',w.is_leaf)

    # print(w)
    y = torch.mm(w,x)
    

    loss = loss_f(y,true_y)

    loss.backward(retain_graph=True)


    print('loss',loss)

    wg = w.grad
    print(wg)

    upd = -lr * wg
    w = w+ upd
    
    # w.requires_grad=True
    w = w.detach()

    print('wl',w.is_leaf)
    
    print('wg',wg)

    

    ####
    y = torch.mm(w,x)
    

    loss = loss_f(y,true_y)

    loss.backward(retain_graph=True)
    print(w.grad)