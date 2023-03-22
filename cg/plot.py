# _*_ coding: utf-8 _*_

"""
python_visual_animation.py by xianhu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# 解决中文乱码问题
matplotlib.rcParams["axes.unicode_minus"] = False


def simple_plot():
    """
    simple plot
    """
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式
    plt.ion()

    # 循环
    for index in range(100):
        # 清除原有图像
        plt.cla()

        # 设定标题等
        plt.title("dynamic figure")
        plt.grid(True)

        # 生成测试数据
        x = np.linspace(-np.pi + 0.1*index, np.pi+0.1*index, 256, endpoint=True)
        y_cos, y_sin = np.cos(x), np.sin(x)

        # 设置X轴
        plt.xlabel("X")
        plt.xlim(-4 + 0.1*index, 4 + 0.1*index)
        plt.xticks(np.linspace(-4 + 0.1*index, 4+0.1*index, 9, endpoint=True))

        # 设置Y轴
        plt.ylabel("Y")
        plt.ylim(-1.0, 1.0)
        plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 画两条曲线
        plt.plot(x, y_cos, "b--", linewidth=2.0, label="cos示例")
        plt.plot(x, y_sin, "g-", linewidth=2.0, label="sin示例")

        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper left", shadow=True)

        # 暂停
        plt.pause(1)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return
# simple_plot()




def scatter_plot():
    """
    scatter plot
    """
    # 打开交互模式
    plt.ion()


    loss = 10
    # 循环
    for index in range(50):
        # 清除原有图像
        # plt.cla()
        loss = loss/2
        # 设定标题等
        plt.title("dynamic random point")
        plt.grid(True)

        # 生成测试数据

        xl = list()
        point_count = 1
        x_index = [index]
        y_index = [loss]
        print(x_index)
        # exit()


        # 设置相关参数
        color_list = np.random.random(point_count)
        scale_list = np.random.random(point_count) * 100

        # 画散点图
        plt.scatter(x_index, y_index, s=scale_list, c=color_list, marker="o")

        # 暂停
        plt.pause(0.2)

    # 关闭交互模式
    plt.ioff()

    # 显示图形
    plt.show()
    return
scatter_plot()