
import torch 


class dnn():




    n = 1   # 输入维度
    m = 1   # 输出维度
    batch_size = 10
    batch_hight = 8


    rl = torch.nn.ReLU(inplace=False)
    
    
    lr = 0.03
    train_count = 20



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
    


    def lxf(self):
        out = list()
        for i in range(10):
            x = torch.normal(0,1,(self.n,self.batch_size)).half().cuda()
            out.append(x)
        return out

    def fa(self):

       

        true_w = torch.normal(0,1,(self.m,self.n)).half().cuda()
        true_b = torch.normal(0,1,(self.m,self.batch_size)).half().cuda()


        # x = torch.normal(0,1,(n,batch_size)).half().cuda()

        
        w = torch.normal(0,1,(self.m,self.n)).half().cuda()
        w.requires_grad=True
        b = torch.normal(0,1,(self.m,self.batch_size)).half().cuda()
        b.requires_grad=True


        lx = self.lxf()


        for epoch in range(self.train_count):

            for i in range(self.batch_hight):
                x = lx[i]
                true_y = self.rl(true_w @ x +true_b)
                
                # wx = w @ x

                # wxb = wx+b

                y = self.rl(w@x+b)
                
                loss = self.loss_f(y,true_y,self.batch_size)

                loss.backward()
                # self.update(w,b)
                with torch.no_grad():
                    w -= self.lr * w.grad / self.batch_size
                    w.grad.zero_()


                    b -= self.lr * b.grad / self.batch_size
                    b.grad.zero_()



                print(loss)

        print('test')

        x = torch.normal(0,1,(self.n,self.batch_size)).half().cuda()
        y = self.rl(w@x+b)
        loss = self.loss_f(y,true_y,self.batch_size)
        print(loss)


        print(y)
        print(true_y)



# noise = torch.normal(0,0.1,(3,3))
# print(noise)

# dnn.fa()
a = dnn()
a.fa()

