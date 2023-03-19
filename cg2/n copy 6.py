









































import torch 
import matplotlib.pyplot as plt



class dnn():
    
    # super_param = [6,2,2,2,2,2,2,2]
    super_param = [6,6,6]
    super_param = [4,4,4,2]
    super_param = [4,4,4,2]
    super_param = [4,4,4]

    #　数据量
    batch_size = 2**10
    batch_hight = 2**2

    # 训练量
    print_period = 2**9
    train_count = print_period * 2**10



    depth = len(super_param)
    lr = 0.03
    rl = torch.nn.ReLU(inplace=False)   # 定义relu

    def test_a(self,x,true_y,param):
        # 不止被 test调用注意。
        y = self.forward(x,param)
        loss = self.loss_f(y,true_y,self.batch_size)
        return loss




    def test(self,true_param,param):
        kn = self.super_param[0]
        n = 2**kn
        
        test_count = 2**10
        fls = 0
        fll = list() #　float_loss_list


        valid_count = test_count

        invaild = 0
        for i in range(test_count):
            x = torch.normal(0,1,(n,self.batch_size)).half().cuda()
            true_y = self.forward(x,true_param)
            loss = self.test_a(x,true_y,param)
            fl = float(loss)
            fll.append(fl)

            if(fl>2**10):
                invaild = 1
                fl = 0
                valid_count -=1
            
            fls += fl
        # print('')

        # print('vc',valid_count)

        if(valid_count==0):
            flv = float('inf')
        else:
            flv = fls /valid_count
        # if(flv>2**10):

        valid_ratio = valid_count/test_count
        

        if(invaild ==1 ):
            # print('训练失败,测试成绩如下')
            # print(fll)
            # print('测试中有无效的测试点，有效率',valid_ratio)# 剩下的点的损失大到离谱
            # print('平均测试损失',flv)
            pass
        else:
            # print('平均测试损失',flv)
            pass

        return flv,valid_ratio


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

                n = 2**n
                m = 2**m

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
        n = 2**n



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

        # batch_size = self.batch_size
        

        plt.ion()
        plt.figure(1)

        
        try_index = 0
        while(1):
            print("try_index",try_index)
            try_index+=1


            find_it = 0
            patience = 2**4

            

            # 重新随一个目标网络
            true_param = self.build_nn()
            data_list = self.dlf(true_param)  


            # 重新随一个初始训练网络
            train_param = self.build_nn()

            # print('训练前测试')
            # loss_before = self.test(true_param,train_param)

            
            # print('\a')
            for epoch in range(self.train_count):

                for i in range(self.batch_hight):
                    data = data_list[i]
                    x = data['x']
                    true_y = data['y']

                    loss = self.test_a(x,true_y,train_param)
                    
                

                    loss.backward(retain_graph=True)

                    self.update(train_param)


                # if(epoch%(self.print_period * 2**2)== 0):
                    
                #     test_loss = self.test(true_param,train_param)


                if(epoch%(self.print_period)== 0):
                    pp = epoch//(self.print_period)
                    # 动态调整学习率
                    if(loss>10):
                        self.lr=2
                    elif(loss>1):
                        self.lr = 1 #0.1
                    else:
                        self.lr = 0.03


                    
                    if(loss<0.01):
                        print('不需要再训练了')
                        print('练习时长',epoch)
                        break

                    # fl = float(loss)
                    # if fl == float('inf'):
                    #     print('无效')
                    # else:
                    #     print('训练集损失',float(loss),pp,'lr = ',self.lr)


                    test_loss,valid_ratio = self.test(true_param,train_param)
                    print(test_loss,valid_ratio)                
                    fl = float(loss)
                    print(fl)

                    if(valid_ratio>2**-7):
                        if(find_it == 0):
                            # 如果找到了就滴一声
                            print('\a')
                            find_it = 1
                        pass
                    else:
                        patience -=1
                        print('patience',patience)
                    
                    if(patience<=0):
                        break
                                
                        
                    
                    if(pp>2**2):
                        # if(loss<100):



                            cl = loss.cpu()
                            cl = float(cl)
                            # print(cl)
                            
                            # plt.grid(True)
                            x_index = [epoch]



                            # 训练损失
                            # y_index = [cl]
                            # plt.scatter(x_index, y_index)

                            if(valid_ratio>2**-7):
                            # if(test_loss<2**10):

                                # 测试集 损失
                                y_index = [test_loss]
                                plt.scatter(x_index, y_index)
                                # plt.plot(x_index, y_index,color = 'deeppink',linewidth = 2,linestyle = '-')

                                print('测试集损失',test_loss)

                                # y_index = [valid_ratio]
                                # plt.scatter(x_index, y_index)
                            


                            plt.pause(0.01)


                            # plt.scatter(x_index, y_index, marker="o")
                            
            # 训练循环结束


        # self.test(true_param,train_param)

        # print('训练前损失',loss_before)

        # 验证 test 函数的自洽性
        # self.test(true_param,true_param)

        
        print('\a')
        plt.pause(0)

a = dnn()
a.fa()

