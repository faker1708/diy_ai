import numpy as np
import matplotlib.pyplot as plt



for i in range(10):
    x = i
    y= 2*x
    plt.plot(x, y)

    plt.grid()

    plt.show()