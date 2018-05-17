import scipy
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

class OpenCLR:
    
    def __init__(self):
        self.model = []
        self.lib = ctypes.cdll.LoadLibrary("./ctest.so")
        self.learn = self.lib.stageForLR
        self.learn.restype = None
        self.learn.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int,ctypes.c_int,  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                        
        self.predict = self.lib.stageForPredict
        self.predict.restype = None
        self.predict.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                               ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int,ctypes.c_int,  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
    
    
    

    def LR_learn(self, x, y, model_space):
        self.learn(x, y, 12288, 209, model_space)
        self.model = list(model_space)

    
    def LR_predict(self, x, y, outdata, ysize):
        self.predict(x, np.ascontiguousarray(self.model, np.float32), 12288, ysize, outdata)
        
#self.model = list(outdata)

    





import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import numpy as np


def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes






train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

x = train_set_x_flatten
y = train_set_y

xt = test_set_x_flatten
yt = test_set_y
#indata = np.ones((5,6))
outdata = np.ones(12288 + 1)
dat = np.ascontiguousarray(outdata, np.float32)
Test = OpenCLR()
Test.LR_learn(np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y, np.float32), dat)
#print(dat)
#print(Test.model)
ysize = 209
testdat = np.ones(ysize)
dat = np.ascontiguousarray(testdat, np.float32)
Test.LR_predict(np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y, np.float32), dat, ysize)


ysize = 50
testdatt = np.ones(ysize)
dat2 = np.ascontiguousarray(testdatt, np.float32)
Test.LR_predict(np.ascontiguousarray(xt, np.float32), np.ascontiguousarray(yt, np.float32), dat2, ysize)

yt = yt[0]

y = y[0]
right = 0
for i in range(len(y)):
    if y[i] == dat[i]:
        right = right+ 1

percent = right/len(y)
print("Accuracy on training data " + str(percent*100) + "%")

right = 0
for i in range(len(yt)):
    if yt[i] == dat2[i]:
        right = right+ 1
        
        
 
percent = right/len(yt)

print("Accuracy on unseen data " + str(percent*100) + "%")





