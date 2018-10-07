import sys
import numpy as np
from sklearn import linear_model
import sklearn.metrics as smi
import sklearn.metrics as sm

x = []
y = []
file_name = 'data/data_single.txt'
with open(file_name, 'r') as fp:
    for line in fp.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)
num_training = int(0.8*len(x))
num_test = len(x) - num_training
x_train = np.array(x[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

x_test = np.array(x[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

lr = linear_model.LinearRegression()

lr.fit(x_train, y_train)
y_test_prob = lr.predict(x_test)

print('mean absolute err:',round(sm.mean_absolute_error(y_test, y_test_prob),2))
print('mean squared err:',round(sm.mean_squared_error(y_test, y_test_prob),2))
print('median absolute err:',round(sm.median_absolute_error(y_test, y_test_prob),2))
print('variance:',round(sm.explained_variance_score(y_test, y_test_prob),2))
print('r2:',round(sm.r2_score(y_test, y_test_prob),2))





