import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

# 10 samples:
square = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]) # square
X_bar = np.concatenate((np.ones(square.shape),square))  # dua ve ma tran nhu ham stack
X_bar = X_bar.T
price = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])           # house's price
### luu y: X trong slide la vecto cot --> sua ct code
reg = LR.fit(square, price.T)
print(reg)
THETA = np.linalg.pinv(X_bar.T@X_bar)@X_bar.T@price
print(THETA)
reg = LR.fit(square.T, price)

# Các thông số
alpha = 0.001  # learning rate
N = 1000     # số lần lặp
M = len(square)

def h_theta (B, A, Xh):       # hàm h
    h = B + A * Xh
    return h

def Jtheta (D0, D1, m, Xj, Yj):    # hàm tính J
    h = h_theta(D0,D1,Xj)
    error = h - Yj
    J = np.sum(error ** 2) / (2 * m)
    #J = ((np.sum(h-Yj))**2)/(2 * m)
    return J

# Thực hiện Gradient Descent
def gradientdesent (X, Y, Alpha, m, n):
    J00 = 0
    J01 = 0
    theta1 = 0
    theta0 = 0
    J_store = [] # danh sach luu
    for i in range(0,n):
        h = h_theta(theta0, theta1, X)
        J00 = Jtheta(theta0, theta1, m, X, Y)
        J_store.append(J00)
        error = h - Y
        temp0 = theta0 - Alpha * (np.sum(error)) / m
        temp1 = theta1 - Alpha * (np.sum(error) * X) / m
        theta0 = temp0
        theta1 = temp1
        J01 = Jtheta(theta0, theta1, m, X, Y)
        #if np.round(J00,20) == np.round(J01, 20):
        #    print ('End: n= %d ;  J= %.6f'%(i,J01))

        #    break
        J00 = J01
    return theta0, theta1, J01, h, J_store

# main
theta0_opt, theta1_opt, J_opt, h_opt, J_list = gradientdesent (square, price, alpha, M, N)
print("Giá trị của theta0 và theta1 là:", theta0_opt, theta1_opt)
print("num of loop", len(J_list))


# Vẽ đồ thị
plt.scatter(square, price, color='orange', marker='*', label='Scatter')
plt.xlabel('square (feet)')
plt.ylabel('House price ($1000s)')
x1 = 1000
y1 = theta0_opt + theta1_opt * x1
x2 = 3000
y2 = theta0_opt + theta1_opt * x1
#plt.plot((x1,x2),(y1,y2))
#plt.plot(range(N), J_list, color='red', label='Cost Function J')
plt.plot(square, theta0_opt + theta1_opt * square, color='blue', label='Linear Regression')
plt.legend()  # Hiển thị chú thích##
plt.show()