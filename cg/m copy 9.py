
import torch 
import matplotlib.pyplot as plt



class dnn():
    # 开始写双层

    n = 2**4   # 输入维度
    m = 2**3   # 输出维度
    l = 2**2
    k = 2**1
    # print(k)

    batch_size = 2**4
    batch_hight = 2**2


    rl = torch.nn.ReLU(inplace=False)
    
    
    lr = 0.03
    # lr = 1

    print_period = 2**8
    train_count = print_period * 2**4


    true_w = torch.normal(0,1,(m,n)).half().cuda()
    true_b = torch.normal(0,1,(m,batch_size)).half().cuda()

    true_w1 = torch.normal(0,1,(l,m)).half().cuda()
    true_b1 = torch.normal(0,1,(l,batch_size)).half().cuda()

    
    true_w2 = torch.normal(0,1,(k,l)).half().cuda()
    true_b2 = torch.normal(0,1,(k,batch_size)).half().cuda()




    def loss_f(self,y,true_y,batch):

        diff_y = y-true_y
        # print(diff_y)

        torch.clamp(diff_y,0,255)

        pp = diff_y**2
        # pp = diff_y



        ps = pp/2

        la = ps.sum()
        loss = la/batch
        return loss
    


    def dlf(self):
        # 人工生成数据集

        true_w = self.true_w
        true_b = self.true_b

        true_w1 = self.true_w1
        true_b1 = self.true_b1
        
        true_w2 = self.true_w2
        true_b2 = self.true_b2
        
        
        batch_hight = self.batch_hight
        batch_size = self.batch_size

        n = self.n

        data_list = list()
        for i in range(batch_hight):
            x = torch.normal(0,1,(n,batch_size)).half().cuda()


            x1 = self.rl(true_w @ x + true_b)
            x2 = self.rl(true_w1 @ x1 + true_b1)
            y = self.rl(true_w2 @ x2 + true_b2)


            data = dict()
            data['x']=x
            data['y']=y
            
            data_list.append(data)
        return data_list

    def fa(self):

        batch_size = self.batch_size
        
        true_w = self.true_w
        true_b = self.true_b

        true_w1 = self.true_w1
        true_b1 = self.true_b1

        
        true_w2 = self.true_w2
        true_b2 = self.true_b2


        # x = torch.normal(0,1,(n,batch_size)).half().cuda()

        
        w = torch.normal(0,1,(self.m,self.n)).half().cuda()
        w.requires_grad=True
        b = torch.normal(0,1,(self.m,batch_size)).half().cuda()
        b.requires_grad=True

        w1 = torch.normal(0,1,(self.l,self.m)).half().cuda()
        w1.requires_grad=True
        b1 = torch.normal(0,1,(self.l,batch_size)).half().cuda()
        b1.requires_grad=True
        
        w2 = torch.normal(0,1,(self.k,self.l)).half().cuda()
        w2.requires_grad=True
        b2 = torch.normal(0,1,(self.k,batch_size)).half().cuda()
        b2.requires_grad=True





        data_list = self.dlf()  # data_list是个三维,高度是 batch_hight
        # data 是二维，维度分别是 m batch_size 


        # train_cost = 0
        

        ###########################################
        
        # 训练前测试

        loss_sum = 0
        sample_count = 2**5
        for ss in range(sample_count):


            x = torch.normal(0,1,(self.n,self.batch_size)).half().cuda()
            x1 = self.rl(w @ x+b)
            x2 = self.rl(w1 @ x1+b1)
            y = self.rl(w2 @ x2 + b2)


            
            true_x1 = self.rl(true_w @ x + true_b)
            true_x2 = self.rl(true_w1 @ true_x1 + true_b1)
            true_y = self.rl(true_w2 @ true_x2 + true_b2)
            
            loss = self.loss_f(y,true_y,self.batch_size)
            loss = float(loss)



            loss_sum += loss

        loss_before = loss_sum/sample_count
        print('训练前损失',loss_before)


        ###########################################

        plt.ion()
        
        print('\a')
        for epoch in range(self.train_count):

            for i in range(self.batch_hight):
                data = data_list[i]
                x = data['x']
                true_y = data['y']

               
                x1 = self.rl(w @ x+ b)
                x2 = self.rl(w1 @ x1+ b1)
                y = self.rl(w2 @ x2 + b2)
                
                loss = self.loss_f(y,true_y,batch_size)
            

                loss.backward()

                with torch.no_grad():
                    w -= self.lr * w.grad / batch_size
                    w.grad.zero_()
                    b -= self.lr * b.grad / batch_size
                    b.grad.zero_()

                    w1 -= self.lr * w1.grad / batch_size
                    w1.grad.zero_()
                    b1 -= self.lr * b1.grad / batch_size
                    b1.grad.zero_()




                    w2 -= self.lr * w2.grad / batch_size
                    w2.grad.zero_()
                    b2 -= self.lr * b2.grad / batch_size
                    b1.grad.zero_()


            if(epoch%(self.print_period)== 0):
                # print(loss)
                # print(float(loss),end='\n')
                print(float(loss),epoch)


                # print(self.lr)
                if(loss>10):
                    # print('太大了')
                    self.lr=2
                elif(loss>1):
                    self.lr = 1 #0.1
                else:
                    self.lr = 0.03



                
                if(loss<0.01):
                    print('不需要再训练了')
                    print('练习时长',epoch)
                    break


                cl = loss.cpu()
                cl = float(cl)
                print(cl)
                
                plt.grid(True)
                x_index = [epoch]
                y_index = [cl]
                plt.scatter(x_index, y_index, marker="o")
                
                plt.pause(0.2)

        print('\a')
        # print('test')

        # 训练后测试

        loss_sum = 0
        for ss in range(sample_count):


            x = torch.normal(0,1,(self.n,self.batch_size)).half().cuda()


            x1 = self.rl(w @ x+b)
            x2 = self.rl(w1 @ x1+b1)
            y = self.rl(w2 @ x2 + b2)


            true_x1 = self.rl(true_w @ x + true_b)
            true_x2 = self.rl(true_w1 @ true_x1 + true_b1)
            true_y = self.rl(true_w2 @ true_x2 + true_b2)
            
            loss = self.loss_f(y,true_y,self.batch_size)
            loss = float(loss)


            loss_sum += loss

        loss = loss_sum/sample_count
        print('训练前损失',loss_before)

        print('训练后损失',loss)
        # diff_loss = abs(loss-loss_before)
        # ratio_loss = diff_loss/loss
        # print(ratio_loss)
        plt.pause(0)



# noise = torch.normal(0,0.1,(3,3))
# print(noise)

# dnn.fa()
a = dnn()
a.fa()

