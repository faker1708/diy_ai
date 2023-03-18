
pytorch中默认浮点类型位32位

如何修改成8bit呢？？

a = a.half()
这样可以使用半精度


对不起，放弃了，找不到相关教程。
我也不知道我的显卡是否支持。

我讲实话，这个功能非常关键。

要不然，我们直接用char 自己来写个吧。
char 不就是8bit吗？？



现在有个损失越界的问题。
首先，我们是不能随意写函数的，因为需要torch来管理计算图，让它自动求导。

而有时权重 矩阵太离谱，会导致猜出来 的输出的损失过大，超过了float的范围。




https://github.com/xianhu/LearnPython/blob/master/python_visual_animation.py