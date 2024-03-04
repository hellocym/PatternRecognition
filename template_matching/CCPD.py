import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, Compose, CenterCrop, Resize, ToTensor, Float2Int
import numpy as np
import os
from PIL import Image


class CCPD(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.folders = [folder for folder in os.listdir(self.path) if folder.startswith("ccpd")]
        self.file_list = open(os.path.join(self.path, "splits", "train.txt" if train else "test.txt")).read().split()
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        pil_img = Image.open(os.path.join(self.path, file_name))
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, file_name


def get_ccpd(dataset_path, batch_size, num_workers):
    if num_workers is None:
        num_workers = 8
    ccpd_transform = Compose([
        CenterCrop([700, 700]),
        Resize([256, 256]),
        ToTensor(),
        Float2Int()
    ])
    train_set = CCPD(path=os.path.join(dataset_path, "CCPD2019"), train=True, transform=ccpd_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = CCPD(path=os.path.join(dataset_path, "CCPD2019"), train=False, transform=ccpd_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


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