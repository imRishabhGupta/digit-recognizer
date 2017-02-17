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

clf=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=True)

clf.fit(x,y)
print clf.classes_
#print clf.n_layers_
pred=clf.predict(x_)
print pred
np.savetxt('submission_rand_forest_svm.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
