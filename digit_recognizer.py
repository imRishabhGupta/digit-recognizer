# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:02:16 2017

@author: lenovo
"""

#from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("train.csv")
y = np.array(train.pop('label'))
x = np.array(train)/255.
#train = dataset.iloc[:,1:].values
test = pd.read_csv("test.csv")
x_ = np.array(test)/255.
print x
print x_
'''
#gives array of matrix representation
target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
print train[2000]

plt.imshow(train[1729][0], cmap=cm.binary)

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=250)
rf.fit(train, target)
pred = rf.predict(test)
'''
clf = MLPClassifier(solver='sgd', activation='relu',
                    hidden_layer_sizes=(100,30),
                    learning_rate_init=0.001, learning_rate='adaptive', alpha=0.1,
                    momentum=0.9, nesterovs_momentum=True,
                    tol=1e-4, max_iter=200,
                    shuffle=True, batch_size=300,
                    early_stopping = False, validation_fraction = 0.15,
                    verbose=True)
clf.fit(x,y) 
print clf.classes_
print clf.n_layers_
pred=clf.predict(x_)
print pred
np.savetxt('submission_rand_forest_mlp.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
