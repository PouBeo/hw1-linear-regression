#GD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

# 10 samples:
x = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700], dtype = 'float64') # square
y = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255], dtype = 'float64')
x = x/np.max(x)
y = x/np.max(y)
alpha = 10**(-3)
xbar = np.concatenate(np.ones(x.shape),x)
xbar = xbar.T
#theta = np.zeros((2,1))
theta00 = 0
theta01 = 0
epochs = 100000
m = 10 # len(X)
for i in range (epochs):
    #temp00 = np.sum(theta00 + theta01 * x.T - y)/m #theta0 - Alpha * (np.sum(error)) / m
    #temp01 = np.sum((theta01 + theta01 * x.T - y))*x.T / m
    #theta00 = theta00 - alpha * temp00
    #theta01 = theta01 - alpha * temp01
    a = (theta.T @ xbar - y) @ xbar ).T / m ### phias @xab... ddang cos looix
### nhiem vu:  dua cac pt trong for thanh dang matrix va chi 1 dong theta = ... (matran 2 bien)
### dung x_bar, them hang 1, scaling truoc khi them hang 1... tham khao ml co ban
    J = np.sum((theta00+theta01*x.T-y) ** 2) / (2 * m)

print('D0=',theta00)
print('D1=',theta01)
print('J=',J)

plt.scatter(x, y, color='orange', marker='*', label='Scatter')
plt.xlabel('square (feet)')
plt.ylabel('House price ($1000s)')
plt.plot(x, theta00 + theta01 * x, color='blue', label='Linear Regression')
plt.legend()  # Hiển thị chú thích##
plt.show()