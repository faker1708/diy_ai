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













x = torch.normal(0,1,(n,1)).half().cuda()
x[-1]=1
# x.requires_grad=True



# rl = torch.nn.ReLU(inplace=False)




# rl = torch.nn.ReLU()



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


epoch = 30
lr = 0.01

for i in range(epoch):


    x = torch.normal(0,1,(n,1)).half().cuda()
    x[-1]=1
    # true_y =rl(true_w @ x)
    true_y = true_w @ x
    # print(true_y)


    # y = torch.mm(w,x)
    # rl(y)
    
    y = w @ w

    loss = loss_f(y,true_y)

    loss.backward(retain_graph=True)

    with torch.no_grad():
        w -= lr * w.grad 
        w.grad.zero_()

    # xx = 2
    if(loss<1e-2):
        print('ok',i,float(loss))
        break

# print(w-true_w)

# y = torch.mm(w,x)
# rl(y)

# y

# print(y)
# print(true_y)
# loss = loss_f(y,true_y)
# print(loss)


# print(w)
# print(true_w)
    ll = float(loss)
    # print(loss)
    print(ll)
print(w)