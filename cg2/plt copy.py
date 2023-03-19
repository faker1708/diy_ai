

import matplotlib.pyplot as plt


plt.ion()
plt.grid(True)
for i in range(10):
    x = [i]
    y = [x]
    # plt.scatter(x, y)
    plt.plot(x,y)
    # plt.plot(x, y,color = 'deeppink',linewidth = 2,linestyle = '-')
    plt.pause(0.01)
    plt.show()
plt.pause(10)
