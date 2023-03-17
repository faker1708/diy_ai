impor torch
import math

    # x = torch.normal(0,1,(n,1)).half().cuda()
    # w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


rl = torch.nn.ReLU(inplace=False)

def get_data_set(n,m):
    



    # true_w = torch.tensor([[3,4],[0,1]]).half().cuda()
    true_w = torch.normal(0,1,(m,n))
    true_w [-1]=0
    true_w [-1,-1]=1


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

    loss = loss_tensor.sum()

    return loss


def fa():
    n = 1+12
    m = 1+3
    ds = get_data_set(n,m)
    print(ds[0])
    print(len(ds))

    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


    w = torch.normal(0,1,(m,n)).half().cuda()
    w [-1]=0
    w [-1,-1]=1
    w.requires_grad=True

    
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

            loss = loss_f(y,true_y)

            loss.backward(retain_graph=True)

            with torch.no_grad():
                w -= lr * w.grad 
                w.grad.zero_()

        ll = float(loss)
        # print(ll)
        p = -math.log(ll)
        print(p)
            # print(math.ceil(p))
            # break


    

if __name__ == "__main__":
    fa()