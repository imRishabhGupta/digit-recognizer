# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 00:24:19 2017

@author: lenovo
"""

from sklearn import svm
import numpy as np
import pandas as pd

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("train.csv")
y = np.array(train.pop('label'))
x = np.array(train)/255.
#train = dataset.iloc[:,1:].values
test = pd.read_csv("test.csv")
x_ = np.array(test)/255.

clf=svm.SVC(decision_function_shape='ovo')

clf.fit(x,y)
print clf.classes_
#print clf.n_layers_
pred=clf.predict(x_)
print pred
np.savetxt('submission_rand_forest_svm.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
