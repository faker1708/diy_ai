

import torch

print("尝试写个简单的神经网络")

# 构造一个参数矩阵


# 输入输出都是一维张量
# 输入维度是n
# 输出维度是m

n = 1+1
m = 1+1


x = torch.rand(n,1).cuda().half()
x[-1]=1
print('x',x)

# x = torch.tensor([[1],[2]])

# true_w = torch.normal()

# print('x',x)


true_w = torch.normal(0,1,(m,n)).cuda().half()
true_w [-1]=0
true_w[-1,-1]=1
# true_w = torch.tensor([[4,5],[6,7],[8,9]])

print('true_w',true_w)

# true_y = x @ w
true_y_pure = torch.mm(true_w,x)



noise = torch.normal(0,1,(m,1)).cuda().half()
noise[-1]  = 0
print(noise)
true_y = true_y_pure + noise 

print('true_y',true_y)


# 训练项 w

w = torch.rand(m,n).cuda().half()


w [-1]=0
w [-1,-1]=1


print('w',w)

# true_y = true_y.half()




print(true_y.dtype)
