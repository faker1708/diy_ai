
import torch 


class dnn():

    # batch_size 越小，成绩越好。奇怪
    # 模型大时，batchsize 也要 大，不然loss一直是nan

    # 数据太多，但模型不够大，也就是说，数据集中会存在矛盾的冲突的数据。导致难以训练 。也有可能 啊。

    # 模型越大，loss 越难收敛到小的范围里。几十都是福报了。可能 还是那个问题，模型太小，但数据太大，有太多矛盾的数据 了。

    #   模型太大，数据 太少，也训练不出来。
    # 一般来说，loss小于1才叫成功吧。大于1可以说是看个乐呵。

    # 数据相对模型一旦少，成绩就不会好。数据 一旦多，由于我这个程序设计的问题，会产生矛盾数据，成绩也会差。无法收敛到很小 的范围内

    # bs 也不是越小越好。。。太玄学了。


    n = 2**10   # 输入维度
    m = 2**10   # 输出维度
    batch_size = 2**7
    batch_hight = 2**7


    rl = torch.nn.ReLU(inplace=False)
    
    
    lr = 0.03
    lr = 10
    train_count = 100* 2**6


    true_w = torch.normal(0,1,(m,n)).half().cuda()
    true_b = torch.normal(0,1,(m,batch_size)).half().cuda()



    def update(self,w,b):
        with torch.no_grad():
            w -= self.lr * w.grad 
            w.grad.zero_()


            b -= self.lr * b.grad 
            b.grad.zero_()


    def loss_f(self,y,true_y,batch):

        diff_y = y-true_y
        # print(diff_y)

        pp = diff_y**2
        ps = pp/2

        la = ps.sum()
        loss = la/batch
        return loss
    


    def dlf(self):
        true_w = self.true_w
        true_b = self.true_b
        batch_hight = self.batch_hight
        batch_size = self.batch_size

        n = self.n

        data_list = list()
        for i in range(batch_hight):
            x = torch.normal(0,1,(n,batch_size)).half().cuda()


            y = self.rl(true_w @ x + true_b)
            


            data = dict()
            data['x']=x
            data['y']=y
            
            data_list.append(data)
        return data_list

    def fa(self):

        batch_size = self.batch_size
        
        true_w = self.true_w
        true_b = self.true_b



        # x = torch.normal(0,1,(n,batch_size)).half().cuda()

        
        w = torch.normal(0,1,(self.m,self.n)).half().cuda()
        w.requires_grad=True
        b = torch.normal(0,1,(self.m,batch_size)).half().cuda()
        b.requires_grad=True






        data_list = self.dlf()  # data_list是个三维,高度是 batch_hight
        # data 是二维，维度分别是 m batch_size 


        train_cost = 0

        for epoch in range(self.train_count):

            for i in range(self.batch_hight):
                data = data_list[i]
                x = data['x']
                true_y = data['y']

                # true_y = self.rl(true_w @ x +true_b)
                
                # wx = w @ x

                # wxb = wx+b
                y = self.rl(w @ x+ b)
                
                loss = self.loss_f(y,true_y,batch_size)
                # if(loss>100):
                #     loss = 100

                    

                loss.backward()
                # self.update(w,b)
                with torch.no_grad():
                    w -= self.lr * w.grad / batch_size
                    w.grad.zero_()


                    b -= self.lr * b.grad / batch_size
                    b.grad.zero_()


            if(epoch%100 == 0):
                print(float(loss),end='\n')


                # print(self.lr)
                if(loss>10):
                    # print('太大了')
                    self.lr=20
                else:
                    self.lr = float(loss)/4
                
                if(loss<0.01):
                    print('不需要再训练了')
                    print('练习时长',epoch)
                    break


        print('test')


        x = torch.normal(0,1,(self.n,self.batch_size)).half().cuda()
        y = self.rl(w @ x+b)
        true_y = self.rl(true_w @ x + true_b)
        loss = self.loss_f(y,true_y,self.batch_size)

        loss = self.loss_f(b,true_b,self.batch_size)
        print('成绩',float(loss))


        if(loss<1):
            print('通过')
        else:
            print('不通过')

        # y = y.transpose(0,1)
        # print(y[0])

        # true_y = true_y.transpose(0,1)
        # print(true_y[0])



# noise = torch.normal(0,0.1,(3,3))
# print(noise)

# dnn.fa()
a = dnn()
a.fa()

