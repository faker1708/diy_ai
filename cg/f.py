import torch

n = 1+1
m = 1+1

# x = torch.normal(0,1,(n,1)).half().cuda()
# w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


true_w = torch.tensor([[3,4],[0,1]]).half().cuda()
# true_w = torch.normal(0,1,(m,n)).half().cuda()
true_w [-1]=0
true_w [-1,-1]=1
true_w = true_w.half().cuda()








def loss_f(y,true_y):
    # print('输出一个标量')
    yy = y-true_y

    loss_tensor = yy**2/2

    # print('y',y)

    # print('loss',loss_tensor)

    loss = loss_tensor.sum()

    return loss




w = torch.normal(0,1,(m,n)).half().cuda()
w [-1]=0
w [-1,-1]=1
w.requires_grad=True


epoch = 3000
lr = 0.03


# x = torch.normal(0,1,(n,1)).half().cuda()

x = torch.tensor([[7],[1]]).half().cuda()
x[-1]=1
true_y = true_w @ x

for i in range(epoch):


    # print(true_y)


    # y = torch.mm(w,x)
    # rl(y)
    
    y = w @ x

    loss = loss_f(y,true_y)

    loss.backward(retain_graph=True)

    with torch.no_grad():
        w -= lr * w.grad 
        w.grad.zero_()


    ll = float(loss)
    # print(ll)

print(w)
print(true_w)
print(y)
print(true_y)




x = torch.tensor([[8],[1]]).half().cuda()
true_y = true_w @ x
y = w @ x

print(y)
print(true_y)