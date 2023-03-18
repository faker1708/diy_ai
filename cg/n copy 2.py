
import torch 
import matplotlib.pyplot as plt



class dnn():
    # 开始写双层

    # print(k)


    batch_size = 2**6
    batch_hight = 2**2

    super_param = [4,3,2,2,1]
    depth = len(super_param)
    
    rl = torch.nn.ReLU(inplace=False)   # 定义relu
    
    
    lr = 0.03
    # lr = 1

    print_period = 2**8
    train_count = print_period * 2**6

    def test_a(self,x,true_y,param):
        y = self.forward(x,param)
        loss = self.loss_f(y,true_y,self.batch_size)
        return loss




    def test(self,true_param,param):
        n = self.super_param[0]
        
        test_count = 32
        for i in range(test_count):
            x = torch.normal(0,1,(n,self.batch_size)).half().cuda()
            true_y = self.forward(x,true_param)
            self.test_a(x,true_y,param)
            


    def build_nn(self):
        super_param=self.super_param
        depth = len(super_param)

        param = dict()

        w_list = list()
        b_list = list()
        for i,ele in enumerate(super_param):
            if(i<=depth -2):
                n = super_param[i]
                m = super_param[i+1]
                w = torch.normal(0,1,(m,n)).half().cuda()
                b = torch.normal(0,1,(m,self.batch_size)).half().cuda()
                
                # 默认记录计算图
                w.requires_grad=True
                b.requires_grad=True


                w_list.append(w)
                b_list.append(b)
                    


        param['w_list'] = w_list
        param['b_list'] = b_list
        param['depth'] = depth

        return param

    def forward(self,x,param):
        # y = 0

        w_list= param['w_list']
        b_list= param['b_list']


        depth = param['depth']
        for i in range(depth-1):    # 如果 是4层，则只循环3次，分别 是012
            w = w_list[i]
            b = b_list[i]

            
            x = self.rl(w @ x + b)

        return x


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
    


    def dlf(self,param):
        # 人工生成数据集


        # 解码对象数据
        # param = self.build_nn()

        batch_hight = self.batch_hight
        batch_size = self.batch_size

        n = self.super_param[0]
        



        data_list = list()
        for i in range(batch_hight):
            x = torch.normal(0,1,(n,batch_size)).half().cuda()

            y = self.forward(x,param)


            data = dict()
            data['x']=x
            data['y']=y
            
            data_list.append(data)
        return data_list

    def update(self,param):
        
        w_list= param['w_list']
        b_list= param['b_list']

        batch_size = self.batch_size
        lr = self.lr

        with torch.no_grad():
            
            depth = param['depth']
            for i in range(depth-1): 
                w = w_list[i]
                b = b_list[i]

                w -= lr * w.grad / batch_size
                w.grad.zero_()


                b -= lr * b.grad / batch_size
                b.grad.zero_()


    def fa(self):

        batch_size = self.batch_size
        

        true_param = self.build_nn()
        data_list = self.dlf(true_param)  

        
        train_param = self.build_nn()


        plt.ion()
        
        print('\a')
        for epoch in range(self.train_count):

            for i in range(self.batch_hight):
                data = data_list[i]
                x = data['x']
                true_y = data['y']

                loss = self.test_a(x,true_y,train_param)
                
            

                loss.backward(retain_graph=True)

                self.update(train_param)


            if(epoch%(self.print_period)== 0):
                # print(loss)
                # print(float(loss),end='\n')
                print(float(loss),epoch)


                # print(self.lr)
                if(loss>10):
                    print('损失太大了')
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
        self.test(true_param,train_param)
        plt.pause(0)



# noise = torch.normal(0,0.1,(3,3))
# print(noise)

# dnn.fa()
a = dnn()
a.fa()

