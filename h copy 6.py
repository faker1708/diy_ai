import torch
import math

    # x = torch.normal(0,1,(n,1)).half().cuda()
    # w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


rl = torch.nn.ReLU(inplace=False)

def get_data_set(n,k,m):
    


    # true_w = torch.tensor([[3,4],[0,1]]).half().cuda()
    true_w = torch.normal(0,1,(k,n))
    true_w [-1]=0
    true_w [-1,-1]=1

    true_w1 = torch.normal(0,1,(m,k))
    true_w1 [-1]=0
    true_w1 [-1,-1]=1


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
            y = rl(true_w @ x)
            y1 = rl(true_w1 @ y)

            data.append(x)
            data.append(y1)
            ds.append(data)
        
    print(len(ds))
    return ds


def loss_f(y,true_y):
    # print('输出一个标量')
    yy = y-true_y

    loss_tensor = yy**2/2

    # print('y',y)

    # print('loss',loss_tensor)

    loss = loss_tensor.sum()

    return loss


def fa():
    print('双层非线性')

    n = 1+12

    k = 1+10
    m = 1+3

    k = n
    m =n


    ds = get_data_set(n,k,m)
    print(ds[0])
    print(len(ds))

    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


    w = torch.normal(0,1,(k,n)).half().cuda()
    w [-1]=0
    w [-1,-1]=1
    w.requires_grad=True


    
    w1 = torch.normal(0,1,(m,k)).half().cuda()
    w1 [-1]=0
    w1 [-1,-1]=1
    w1.requires_grad=True

    
    epoch = 30
    lr = 0.01

    for j in range(epoch):

        for i in range(1000):
            data = ds[i]
            x = data[0].cuda().half()
            true_y = data[1].cuda().half()
            # print(x)
            # print(true_y)


            
            # y =w @ x
            y =rl( w @ x)

            y1 = rl(w1 @ y)

            # print(y1)
            # break
            # print(y[-1])
            # print(y1[-1])
            # break
            if(y1[-1]!=1):
                # raise('wefoio')
                # print('错了',i)
                # print(y,y1)
                # print(x,w,w1)
                # exit(-1)

                pass
            loss = loss_f(y1,true_y)

            loss.backward(retain_graph=True)

            with torch.no_grad():
                w -= lr * w.grad 
                w.grad.zero_()


                w1 -= lr * w1.grad 
                w1.grad.zero_()

            # print(w)
            # print(w.data)
            
            
            # 曲线救国
            # w.data[-1]=0
            # w.data[-1,-1]=1
            # w1.data[-1]=0
            # w1.data[-1,-1]=1


        ll = float(loss)
        # print(ll)
        p = -math.log(ll)
        print(p)
            # print(math.ceil(p))
            # break
    
    print('检查')
    datac = ds[0]
    xc = datac[0].cuda().half()
    true_yc = datac[1].cuda().half()
    
    yc =rl( w @ x)

    y1c = rl(w1 @ yc)

    print(true_yc)
    print(y1c)

    

if __name__ == "__main__":
    fa()