import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from sklearn.naive_bayes import BernoulliNB


train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

# 贝叶斯分类器
# 展平
train.data = train.data.float().view(-1, 28 * 28) / 255
test.data = test.data.float().view(-1, 28 * 28) / 255
# 二值化
train.data[train.data >= 0.5] = 1
train.data[train.data < 0.5] = 0
test.data[test.data >= 0.5] = 1
test.data[test.data < 0.5] = 0

# 删除同一列的所有方差为0的列
cols = train.data.std(dim=0) != 0
train.data = train.data[:, cols]
test.data = test.data[:, cols]

# 训练
clf = BernoulliNB()
clf.fit(train.data, train.targets)

# 预测
pred = clf.predict(test.data)
accuracy = (pred == test.targets.numpy()).mean()

print('Accuracy:', accuracy)

# 