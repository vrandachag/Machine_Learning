import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55,60,65,70,75,80]
y=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65,70,75,80,85,90]
X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

regr = LinearRegression()

regr.fit(X_train, y_train)

print(regr.score(X_test,y_test))

y_pred = regr.predict(X_test)

plt.scatter(X_test, y_test, color = 'b')

plt.plot(X_test, y_pred,color = 'k')

plt.show()
