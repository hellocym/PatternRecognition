import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


def similarity(img1:torch.Tensor, img2:torch.Tensor, method='euclidean')->float:
    # 计算两张图片的相似度
    if method == 'euclidean':
        return torch.sqrt(torch.sum((img1 - img2) ** 2))
    elif method == 'cosine':
        return torch.dot(img1.view(-1), img2.view(-1)) / (torch.norm(img1) * torch.norm(img2))
    else:
        raise ValueError('method参数只能是euclidean或cosine')


train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())


# 在MNIST数据集中选择0-9的数字模板
numbers = []
for i in range(10):
    idx = np.random.choice(np.where(train.targets == i)[0])
    img = train.data[idx]
    img = img.float() / 255
    img = img.view(-1)
    numbers.append(img)

correct_num = 0
# 遍历测试集中的每一张图片
for img, label in test:
    img = img.float() / 255
    img = img.view(-1)
    # 计算测试图片与0-9数字模板的相似度
    similarities = []
    for number in numbers:
        similarities.append(similarity(img, number, method='cosine'))
    # 找到最相似的数字模板
    pred = np.argmax(similarities)
    if pred == label:
        correct_num += 1
    # print(similarities)
    # input(f'预测值：{pred}，真实值：{label}')
print(f'正确率：{correct_num / len(test)}')