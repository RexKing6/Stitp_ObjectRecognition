# coding=utf-8

import numpy as np
from skimage import feature as ft
from sklearn.svm import SVC
import os
import pickle
import cv2

classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '货车']


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray

# 从序列化存储文件中读取所有数据
filePath = './cifar-10-batches-py'

TrainData = []
for childDir in os.listdir(filePath):
    if 'data_batch_' in childDir:
        f = os.path.join(filePath, childDir)
        data = unpickle(f)
        if 'a' not in dir():
            a = data['data']
            b = data['labels']
            c = data['filenames']
        else:
            a = np.vstack((a, data['data']))
            b.extend(data['labels'])
            c.extend(data['filenames'])


train = np.reshape(a, (50000, 3, 32 * 32))
labels = np.reshape(b, (50000, 1))
fileNames = np.reshape(c, (50000, 1))
TrainData = zip(train, labels, fileNames)

# 提取训练样本并训练，存储训练出来的模型数据
train_x = []
for data in train[:10000]:
    image = np.reshape(data.T, (32, 32, 3))
    gray = rgb2gray(image) / 255.0
    x = ft.hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(3, 3), block_norm='L2-Hys')
    train_x.append(x)

train_y = labels[:10000]
train_y.ravel()
train_y.tolist()
Y = []
for i in train_y:
    Y.append(i[0])

clf = SVC(decision_function_shape='ovc')
clf.fit(train_x, Y)
pickle.dump(clf, open("model", "wb"))

'''
# 从序列化存储文件中读取模型数据
clf = cPickle.load(open("model", "rb"))

# 提取测试数据，可以测试或者存储


def testall():
    test_x = []
    for i in range(20001, 20200):
        data = train[i]
        image = np.reshape(data.T, (32, 32, 3))
        cv2.imwrite('./images/' + fileNames[i][0], image)
        gray = rgb2gray(image) / 255.0
        x = ft.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(3, 3), block_norm='L2-Hys')
        test_x.append(x)

    test_y = labels[20001:20100]
    test_y.ravel()
    test_y.tolist()
    Y2 = []
    for i in test_y:
        Y2.append(i[0])

    # 测试并计算正确率
    Y3 = []
    Y3 = clf.predict(test_x)

    total = 0
    for i in range(len(Y2)):
        if Y2[i] == Y3[i]:
            total = total + 1
    print 1.0 * total / (len(Y2) + 1)


# 读取单个测试样本并输出预测类
def testone():
    path = 'images/speedboat_s_000048.png'
    image = cv2.imread(path)
    gray = rgb2gray(image) / 255.0
    x = ft.hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(3, 3), block_norm='L2-Hys')
    i = clf.predict([x])
    return classes[i[0]]
'''