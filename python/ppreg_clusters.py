import ppreg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold

data = np.loadtxt("../data/comp2-codify-c1.csv",delimiter=",",skiprows=1)

columns=["CuT","CuS","Fe","As"]

n = len(data)

x = data[:,:6]
y = data[:,7]

sc = StandardScaler()
X = x.copy() #sc.fit_transform(x)

for d in xrange(5,10):
    kf = KFold(n_splits=5,shuffle=True)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        #print X_train.shape,X_test.shape
        #print y_train.shape,y_test.shape
        #print len(y_train) + len(y_test),n
        model = linear_model.LinearRegression()
        #model = ppreg.ProjectionPursuitRegression(d,d)
        model.fit(X_train,y_train)
        score = model.score(X_test,y_test)
        prediction = model.predict(X_test)

        error = y_test-prediction

        print d,score,np.mean(error),np.var(error),np.var(y_test)
        
        
