# coding=utf-8

import numpy as np
from skimage import feature as ft
from sklearn.svm import SVC
import wx
import os
import sys
import pickle
import cv2

wildcard = u"PNG 文件 (*.png)|*.png|"\
           u"JPEG 文件 (*.jpg)|*.jpg|"\
           u"JPEG 文件 (*.jpeg)|*.jpeg|"\
           "All files (*.*)|*.*"

classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '货车']


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


class FileDialog(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, -1, "", (500, 220), (192, 70),
                          wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        b1 = wx.Button(self, -1, u"浏览", (0, 0))
        self.Bind(wx.EVT_BUTTON, self.OnButton1, b1)

        b2 = wx.Button(self, -1, u"开始测试", (87, 0))
        self.Bind(wx.EVT_BUTTON, self.Svm, b2)

    def OnButton1(self, event):
        dlg = wx.FileDialog(self, message=u"选择文件",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=wildcard,
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            self.path = dlg.GetPaths()[0]
            dlg.Destroy()

    def Svm(self, event):
        print('读取模型中……')
        clf = pickle.load(open("./model", "rb"))
        self.path.replace('\\', '/')
        image = cv2.imread(self.path)
        gray = rgb2gray(image) / 255.0
        x = ft.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(3, 3), block_norm='L2-Hys')
        print('识别中……')
        i = clf.predict([x])
        print('识别结果:')
        print(classes[i[0]]).decode('utf-8')
        self.Destroy()

if __name__ == '__main__':
    app = wx.App(redirect=True)
    frame = FileDialog()
    frame.Show()
    app.MainLoop()
