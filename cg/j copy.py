
 # 重写 h.py




import torch


    # x = torch.normal(0,1,(n,1)).half().cuda()
    # w = torch.normal(0,1,(m,n),requires_grad=True).half().cuda()


rl = torch.nn.ReLU(inplace=False)

def get_data_set(n,m):
    

    batch_size = 100

    # true_w = torch.tensor([[3,4],[0,1]]).half().cuda()
    true_w = torch.normal(0,1,(m,n))
    true_w [-1]=0
    true_w [-1,-1]=1
    true_w = true_w




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
        for j in range(batch_high):
            data = list()
            x = torch.normal(0,1,(n,1))
            x[-1]=1
            y = rl(true_w @ x)
            data.append(x)
            data.append(y)
            ds.append(data)
        

    return ds


def fa():
    n = 1+14
    m = 1+4
    ds = get_data_set(n,m)
    print(ds[0])

    



if __name__ == "__main__":
    fa()