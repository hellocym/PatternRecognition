import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat
from matplotlib import pyplot as plt


test_images = loadmat('data/test_images.mat')['test_images']
test_labels = loadmat('data/test_labels.mat')['test_labels1']
train_images = loadmat('data/train_images.mat')['train_images']
train_labels = loadmat('data/train_labels.mat')['train_labels1']

train_num = 300
data_train = train_images[:, :, :train_num].reshape(28*28, train_num).T

kmeans = KMeans(n_clusters=10, init='random', random_state=42)

result = kmeans.fit_predict(data_train)

# 展示效果
show_result = [[] for _ in range(10)]
for num in range(train_num):
    show_result[result[num]].append(train_images[:, :, num])

max_len = max(len(cluster) for cluster in show_result)

image = np.zeros((10 * 28, max_len * 28))  # Assume each image is 28x28
for j in range(10):
    len_cluster = len(show_result[j])
    for k in range(len_cluster):
        image[j * 28:(j + 1) * 28, k * 28:(k + 1) * 28] = show_result[j][k]

plt.figure(figsize=(15, 10))
plt.imshow(image, cmap='gray')
plt.title('Clustered Images')
plt.axis('off')
plt.show()