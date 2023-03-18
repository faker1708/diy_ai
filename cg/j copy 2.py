import torch
import math

    # x = torch.normal(0,1,(n,1)).half().cuda()
    # w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


rl = torch.nn.ReLU(inplace=False)

def get_data_set(n,m):
    


    # true_w = torch.tensor([[3,4],[0,1]]).half().cuda()
    true_w = torch.normal(0,1,(m,n))


    true_b = torch.normal(0,1,(m,1))

    batch_width = 100
    batch_high = 10


    '''
        每条数据由输入x 与 输出 y 组成
        x 是1维张量，长度为 n
        y 是1维张量，长度为 m
        一共有 bw*bh 这么 多条数据

        (注意，x y 最后一个数字都是1，作用是占位)

    
    '''

    #   batch_width  相当于batch_size

    set_count = batch_width * batch_high
    # 人工生成1000个数据的数据集 我不知道怎样调用cuda核心来并行地计算矩阵（或者说神经网络），算了，就用串行的写法吧


    ds = list()
    for i in range(batch_high):
        for j in range(batch_width):
            data = list()
            x = torch.normal(0,1,(n,1))
            x[-1]=1
            # y = true_w @ x
            y = rl(true_w @ x + true_b)

            data.append(x)
            data.append(y)
            ds.append(data)
        
    print(len(ds))
    return ds


def loss_f(y,true_y):
    # print('输出一个标量')
    yy = y-true_y

    loss_tensor = yy**2/2

    # print('y',y)

    # print('loss',loss_tensor)

    # print(loss_tensor)
    # print(loss_tensor[:-1])

    # aa = loss_tensor[:-1]
    # loss = aa.sum()
    

    # loss = loss_tensor[:-1].sum()
    loss = loss_tensor.sum()

    return loss


def train():
    print('单层非线性')

    n = 12

    m = 3


    ds = get_data_set(n,m)
    print(ds[0])
    print(len(ds))


    w = torch.normal(0,1,(m,n)).half().cuda()
    w.requires_grad=True

    b = torch.normal(0,1,(m,1)).half().cuda()
    b.requires_grad=True

    
    epoch = 33
    lr = 0.01

    for j in range(epoch):

        for i in range(1000):
            data = ds[i]
            x = data[0].cuda().half()
            true_y = data[1].cuda().half()
           
            y =rl( w @ x + b)

            # y1 = rl(w1 @ y)

          
            loss = loss_f(y,true_y)

            loss.backward(retain_graph=True)

            with torch.no_grad():
                w -= lr * w.grad 
                w.grad.zero_()
                b -= lr * b.grad 
                b.grad.zero_()

        ll = float(loss)
        p = -math.log(ll)
        print(p)
    w1 =0
    return w,w1,ds
    

def fa():

    w,w1,ds = train()
    # ds = lw[2]


    data = ds[0]
    x = data[0].cuda().half()
    true_y = data[1].cuda().half()

    # w= lw[0]
    # w1= lw[1]

    y =rl( w @ x)

    print(true_y)
    print(y)


if __name__ == "__main__":
    fa()