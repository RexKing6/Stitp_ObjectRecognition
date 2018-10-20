# HOG+SVM on CIFAR-10

## Dataset

* [CIFAP-10](http://www.cs.toronto.edu/~kriz/cifar.html)

## Requirements

* python
* scikit-learn
* scikit-image
* cPickle
* wxPython
* opencv

## Usage

1. From https://drive.google.com/drive/folders/1ycDcidhsywSPBmgZuKSP-NdtrdIATVyr,

* Download the model to put it in `./`
* Download the batches to put them in `./cifar-10-batches-py/`
* Download the images to put them in `./images/`

2. Model training and saving:

```python
python train_save_test.py
```

3. Demo presentation with wxPython:

```
python wx_show_test.py
```

## Chinese Introduction

* 通过sklearn和skimage提取hog特征，并且用svm进行物体识别。
* 用wxpython做了一个简单的界面，选择一张图片进行预测，图片来自CIFAR-10，总共10个类：'飞机','汽车','鸟','猫','鹿','狗','蛙','马','船','货车'。
* 因为为了节省测试的时间，然后就事先把训练出来的svm模型变量通过cPickle这个包序列化存储在本地上，测试时直接读取模型变量model就好。