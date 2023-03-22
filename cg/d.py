

import torch

# print('这个结构不方便用backward函数，放弃')

print("尝试写个简单的神经网络")

# 构造一个参数矩阵


# 输入输出都是一维张量
# 输入维度是n
# 输出维度是m

n = 1+1
m = 1+1


x = torch.rand(n,1,requires_grad=True).cuda().half()
x = torch.tensor([[-7],[1]]).cuda().half()

x[-1]=1



true_w = torch.normal(0,1,(m,n)).cuda().half()
true_w = torch.tensor([[3,4],[0,1]]).cuda().half()
true_w [-1]=0
true_w[-1,-1]=1


# 生成噪声
noise = torch.normal(0,1,(m,1)).cuda().half()
noise[-1]  = 0




# 非线性函数relu
rl = torch.nn.ReLU(inplace=False)

# 线性变换
true_y_pure = torch.mm(true_w,x)
# print('true_y_pure',true_y_pure)

# 非线性变换
true_y_pure = rl(true_y_pure)
# print('true_y_pure',true_y_pure)


# 生成数据集
true_y = true_y_pure + noise 





print('x',x)

print('true_w',true_w)

print('noise',noise)


print('true_y',true_y)


print("\n\n")

print('\n\ntrain')
print("现在比较寒酸，数据集中只有一个数据，batch也只能取一个了")
print("\n\n")


def loss_f(y):
    # 返回一个标量
    diff = y-true_y
    l = diff*diff
    l = l/2
    out = l.sum()
    return out




# 训练项 w
w = torch.normal(0,1,(m,n),requires_grad=True).cuda().half()

# w = torch.rand(m,n,requires_grad=True).cuda().half()
w [-1]=0
w [-1,-1]=1

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

epoch = 3

for i in range(epoch):
    print('w',w)


    print("forward")
    # train_y_linear = torch.mm(w,x)
    train_y_linear = torch.matmul(w,x)


    train_y = rl (train_y_linear)
    # train_y.sum().backward(retain_graph=True)
    loss = squared_loss(true_y,train_y)
    loss.sum().backward(retain_graph=True)


    print(x.grad)
    print(w.grad)

    # print('train_y_linear\n',train_y_linear)
    # print('train_y\n',train_y)

    # print("backward")

    # # loss = loss_f(train_y)

    # diff = (train_y-true_y)
    # loss = diff*diff/2
    # # loss = loss.sum()

    # print(diff)
    # print(loss)
    # loss = train_y

    # loss.sum().backward()



    # print(loss)
    # print(w.grad)

    # print("\n\n")
    
    