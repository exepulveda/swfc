import ppreg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_csv("../data/comp2-codify.csv")

clusters = [1,2,3,4]
#columns=["x","y","z","CuT","CuS","Fe","As","C1","C2","C3","C4"]
#columns=["CuT","CuS","Fe","As","C1","C2","C3","C4"]
columns=["x","y","z","CuT","CuS","Fe","As"]

n = len(df)

x = df.as_matrix(columns=columns)
y = df['Rec30']

sc = StandardScaler()
#X = sc.fit_transform(x)
X = x.copy()
#X[:,-4:] = x[:,-4:]
#X = x
for i in range(7):
    plt.scatter(y,X[:,i])
    plt.show()
quit()


best=None
for d in xrange(2,11):
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        model = ppreg.ProjectionPursuitRegression(d,d)
        model.fit(X_train,y_train)
        score = model.score(X_test,y_test)
        prediction = model.predict(X_test)

        error = y_test-prediction

        print d,score,np.mean(error),np.var(error),np.var(y_test)


#plt.scatter(y,prediction)
#plt.show()
    
