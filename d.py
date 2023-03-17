

import torch

print("尝试写个简单的神经网络")

# 构造一个参数矩阵


# 输入输出都是一维张量
# 输入维度是n
# 输出维度是m

n = 1+1
m = 1+1


x = torch.rand(n,1).cuda().half()
x = torch.tensor([[7],[1]]).cuda().half()

x[-1]=1



true_w = torch.normal(0,1,(m,n)).cuda().half()
true_w = torch.tensor([[3,4],[0,1]]).cuda().half()
true_w [-1]=0
true_w[-1,-1]=1


# 生成噪声
noise = torch.normal(0,1,(m,1)).cuda().half()
noise[-1]  = 0




# 非线性函数relu
rl = torch.nn.ReLU(inplace=True)

# 线性变换
true_y_pure = torch.mm(true_w,x)
# print('true_y_pure',true_y_pure)

# 非线性变换
rl(true_y_pure)
# print('true_y_pure',true_y_pure)


# 生成数据集
true_y = true_y_pure + noise 





print('x',x)

print('true_w',true_w)

print('noise',noise)


print('true_y',true_y)


print("\n\n")

print('\n\ntrain')
print("现在比较寒酸，数据集中只有一个数据，batch也只能取一个了")
print("\n\n")



# 训练项 w

w = torch.rand(m,n).cuda().half()
w [-1]=0
w [-1,-1]=1


epoch = 3

for i in range(epoch):
    print('w',w)


    print("forward")
    train_y = torch.mm(w,x)
    rl (train_y)
    print(train_y)


    print("\n\n")

    