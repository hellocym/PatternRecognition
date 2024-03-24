import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
import numpy as np
import os
from PIL import Image
import cv2


class CCPD(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.file_list = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        pil_img = Image.open(os.path.join(self.path, file_name))
        if self.transform:
            pil_img = self.transform(pil_img)
        annotations = self.parse_annotations(file_name)
        digit = annotations['license_plate_number'][-1]
        warped = self.cropNwarp(pil_img, np.array(annotations['vertices_locations'], dtype="float32"))
        # 灰度
        warped = warped.convert('L')
        return warped, digit

    def cropNwarp(self, img: Image, src_coords)->Image:
        # 绿牌尺寸为480mm*140mm
        # 给定四个顶点坐标，透视变换到固定尺寸
        dst_width, dst_height = 480, 140

        # print(f'src_coords:{src_coords}')
        dst_coords = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype='float32')
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_coords, dst_coords)
        
        # 应用透视变换
        warped = cv2.warpPerspective(np.array(img), M, (dst_width, dst_height))
        
        warped = warped[:, :55]

        warped = Image.fromarray(warped)

        # 旋转180度
        warped = warped.rotate(180)
        
        return warped

    def parse_annotations(self, file_name):
        parts = file_name.split('-')
        annotations = {
            'tilt_degrees': [float(parts[1].split('_')[0]), float(parts[1].split('_')[1])],
            'bounding_box_coords': [tuple(int(y) for y in x.split('&')) for x in parts[2].split('_')],
            'vertices_locations': [tuple(int(y) for y in x.split('&')) for x in parts[3].split('_')],
            'license_plate_number': self.decode_license_plate_number(parts[4]),
            'brightness': int(parts[5]),
            'blurriness': int(parts[6].split('.')[0])
        }
        annotations['area'] = (annotations['bounding_box_coords'][1][0] - annotations['bounding_box_coords'][0][0]) * (annotations['bounding_box_coords'][1][1] - annotations['bounding_box_coords'][0][1])
        return annotations

    def decode_license_plate_number(self, code):
        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        
        codes = code.split('_')
        # print(f'codes:{codes}')
        province = provinces[int(codes[0])]
        alphabet = alphabets[int(codes[1])]
        ad = ''.join([ads[int(c)] for c in codes[2:]])

        return province + alphabet + ad


def similarity(img1:torch.Tensor, img2:torch.Tensor, method='euclidean')->float:
    # 计算两张图片的相似度
    if method == 'euclidean':
        return torch.sqrt(torch.sum((img1 - img2) ** 2))
    elif method == 'cosine':
        return torch.dot(img1.view(-1), img2.view(-1)) / (torch.norm(img1) * torch.norm(img2))
    else:
        raise ValueError('method参数只能是euclidean或cosine')


test = CCPD(path=os.path.join("data", "ccpd_green"), train=False)
test.file_list = test.file_list[-1000:]

test2 = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
# 在MNIST数据集中随机选择1000张图片 使用subset
test2 = torch.utils.data.Subset(test2, np.random.choice(len(test2), 1000, replace=False))

# 0-9的数字模板
template_path = os.path.join("data", "NPtemplate")
template_list = os.listdir(template_path)
template_list.sort()
numbers = [cv2.imread(os.path.join(template_path, x), 0) for x in template_list]
numbers = [cv2.resize(x, (28, 28)) for x in numbers]
numbers = [torch.tensor(x).float().view(-1) / 255 for x in numbers]

train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
# 在MNIST数据集中选择0-9的数字模板
for i in range(10):
    idx = np.random.choice(np.where(train.targets == i)[0])
    img = train.data[idx]
    img = img.float() / 255
    img = img.view(-1)
    numbers.append(img)


correct_num = 0
# 遍历CCPD
for img, label in test:
    # 28x28
    img = img.resize((28, 28))
    img = np.array(img)
    # print(img.shape)
    img = torch.tensor(img).float() / 255
    img = img.view(-1)
    # 计算测试图片与0-9数字模板的相似度
    similarities = []
    for number in numbers:
        similarities.append(similarity(img, number, method='cosine'))
    # 找到最相似的数字模板
    pred = np.argmax(similarities)
    # print(pred, label)
    if str(pred%10) == label:
        correct_num += 1
    # print(similarities)
    # input(f'预测值：{pred%10}，真实值：{label}')
# print(correct_num)
# print(f'正确率：{correct_num / len(test)}')

# 遍历MNIST
for img, label in test2:
    img = img.float() / 255
    img = img.view(-1)
    # 计算测试图片与0-9数字模板的相似度
    similarities = []
    for number in numbers:
        similarities.append(similarity(img, number, method='cosine'))
    # 找到最相似的数字模板
    pred = np.argmax(similarities)
    if pred%10 == label:
        correct_num += 1
    # print(similarities)
    # input(f'预测值：{pred}，真实值：{label}')
print(f'正确率：{correct_num / (len(test) + len(test2))}')