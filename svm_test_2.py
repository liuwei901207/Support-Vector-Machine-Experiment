# 实验一： 支持向量 SVM，给定数据的分类并计算分类性能

# START

from svmutil import *

# 读取样例文件
labels_3a3, train = svm_read_problem(r'dataset/3a3.txt')  # 训练集
labels_3a3_test, test = svm_read_problem(r'dataset/3a3_test.txt')  # 测试集

model = svm_train(labels_3a3, train,
                  '-t 2 -c 512 -g 3.0517578125e-05')  # 使用 grid.py 通过自动 grid 获得的参数 RBF核 精度 => 84.2695%
# model = svm_train(labels_3a3, train, '-t 0 -c 0.15')  # 自己尝试修改的参数 线性核 精度 => 84.518%

p_label, p_acc, p_val = svm_predict(labels_3a3_test, test, model)

# 评价性能

TP = 0  # 真正例数目
FP = 0  # 假正例数目
TN = 0  # 真反例数目
FN = 0  # 假反例数目

for i in range(0, len(labels_3a3_test)):
    if p_label[i] == labels_3a3_test[i]:  # 分类正确的情况下
        if labels_3a3_test[i] > 0:  # 正例
            TP = TP + 1
        else:  # 反例
            TN = TN + 1
    else:  # 分类错误的情况下
        if labels_3a3_test[i] > 0:  # 正例
            FP = FP + 1
        else:  # 反例
            FN = FN + 1

A = (TP + TN) / (TP + TN + FP + FN)
P = TP / (TP + FP)
R = TP / (TP + FN)

print('精度：%.2f' % A)
print('查准率：%.2f' % P)
print('查全率：%.2f' % R)

# END
