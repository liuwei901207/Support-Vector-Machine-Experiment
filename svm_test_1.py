# 实验一： 支持向量 SVM，toy 数据的训练可视化

from svmutil import *
import numpy
from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')

toy_train = []

# 读取样例文件
labels, train = svm_read_problem(r'dataset/toy_train_data.txt')

# 使用样例数据进行训练（修改不同参数，观察实验结果）
'''
可选参数说明：
    -t 设置核函数，0为线性核
    -c 表示松弛变量的惩罚因子
'''
model = svm_train(labels, train, '-t 0 -c 1')

# 能够将训练产生的model保留成为文件
svm_save_model('mySaveModel.model', model)

# 获取源数据
for inx, val in enumerate(labels):
    [CX, CY] = train[inx].values()
    toy_train.append([CX, CY])

# 求超平面法向量 y = wx + b
w = numpy.matmul(numpy.array(toy_train)[numpy.array(model.get_sv_indices()) - 1].T, model.get_sv_coef())
b = -model.rho.contents.value

# print(w)
# print(b)

# 绘制支持向量
for i in model.get_sv_indices():
    plt.scatter(toy_train[i - 1][0], toy_train[i - 1][1], color='red', s=80)

# 绘制数据点
toy_train = numpy.array(toy_train).T
plt.scatter(toy_train[0], toy_train[1], c=labels)

# 绘制超平面
plt.plot([-5, 10], [-(-5 * w[0] + b) / w[1], -(10 * w[0] + b) / w[1]])

# 设置图表属性
plt.title("TEST 1: Support Vector Machine Experiment By Liuwei")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()
