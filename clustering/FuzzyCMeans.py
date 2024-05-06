import numpy as np
import skfuzzy as fuzz
from scipy.io import loadmat
from matplotlib import pyplot as plt


test_images = loadmat('data/test_images.mat')['test_images']
test_labels = loadmat('data/test_labels.mat')['test_labels1']
train_images = loadmat('data/train_images.mat')['train_images']
train_labels = loadmat('data/train_labels.mat')['train_labels1']

train_num = 300
data_train = train_images[:, :, :train_num].reshape(28*28, train_num).T

c = 10  # 聚类数
m = 1.5  # 模糊度
error = 1e-2  # 误差
max_iter = 50  # 最大迭代次数


cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_train.T, 10, m, error=error, maxiter=max_iter, init=None)

distances = np.zeros((train_num, 10))
for j in range(10):
    for i in range(train_num):
        distances[i, j] = np.linalg.norm(data_train[i, :] - cntr[j, :])
        
result = np.argmin(distances, axis=1)

# 展示效果
show_result = [[] for _ in range(10)]
for num in range(train_num):
    show_result[result[num]].append(train_images[:, :, num])

max_len = max(len(cluster) for cluster in show_result)

image = np.zeros((10 * 28, max_len * 28))
for j in range(10):
    len_cluster = len(show_result[j])
    for k in range(len_cluster):
        image[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = show_result[j][k]

plt.figure(figsize=(15, 10))
plt.imshow(image, cmap='gray')
plt.title('Clustered Images')
plt.axis('off')
plt.show()