'''
动态折线图演示示例
'''
 
import numpy as np
import matplotlib.pyplot as plt
 
plt.ion()
plt.figure(1)




t_list = []
result_list = []
t = 0
 
while True:
    if t >= 10 * np.pi:
        plt.clf()
        t = 0
        t_list.clear()
        result_list.clear()
    else:
        t += np.pi / 4
        t_list.append(t)
        result_list.append(np.sin(t))

        x = t_list
        y = result_list
        plt.plot(x, y,c='pink',ls='-', marker='o', mec='b',mfc='w')  ## 保存历史数据
        #plt.plot(t, np.sin(t), 'o')
        plt.pause(0.1)