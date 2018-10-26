import numpy as np
from framework import NN
from helpers import sigmoid
from sklearn.preprocessing import StandardScaler

data = []
target =[]

filename = "data/diabetes.csv" # Insert csv file name here
handle = open(filename, 'r')
for line in handle:
    row = list(map(float, line.split(sep=',')))
    instance = row[:-1]
    prediction = row[-1]
    data.append(instance)
    target.append(prediction)

data = np.array(data)
scaler = StandardScaler()
data = scaler.fit_transform(data)
np.random.shuffle(data)
data = data.T
(n, m) = data.shape
target = np.array(target).reshape((1, m))

X = data[:, :500]
Y = target[:, :500]

X_test = data[:, 500:]
Y_test = target[:, 500:]

h = 8 # Number of neurons in hidden layer

model = NN(X, Y, [n, h, 1])
model.train(learning_rate=0.0005, num_iterations=100)

y_pred = model.predict(X)
print ('Accuracy:', ((Y == y_pred).sum()/Y.size*100), '%')

y_pred_test = model.predict(X_test)
print ('Accuracy:', ((y_pred_test == Y_test).sum()/Y_test.size*100), '%')
