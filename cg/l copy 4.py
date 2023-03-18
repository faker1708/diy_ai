
import torch 


class dnn():




    n = 20   # 输入维度
    m = 14   # 输出维度
    batch_size = 2**5
    batch_hight = 8


    rl = torch.nn.ReLU(inplace=False)
    
    
    lr = 0.03
    lr = 1
    train_count = 100* 22


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

                loss.backward()
                # self.update(w,b)
                with torch.no_grad():
                    w -= self.lr * w.grad / batch_size
                    w.grad.zero_()


                    b -= self.lr * b.grad / batch_size
                    b.grad.zero_()


            if(epoch%100 == 0):
                print(float(loss),end='\n')

        print('test')


        x = torch.normal(0,1,(self.n,self.batch_size)).half().cuda()
        y = self.rl(w @ x+b)
        true_y = self.rl(true_w @ x + true_b)
        loss = self.loss_f(y,true_y,self.batch_size)

        loss = self.loss_f(b,true_b,self.batch_size)
        print(float(loss))


        y = y.transpose(0,1)
        print(y[0])

        true_y = true_y.transpose(0,1)
        print(true_y[0])



# noise = torch.normal(0,0.1,(3,3))
# print(noise)

# dnn.fa()
a = dnn()
a.fa()

