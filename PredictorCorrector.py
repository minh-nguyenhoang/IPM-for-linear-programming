from turtle import forward
import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from matplotlib.animation import FuncAnimation
from sympy.plotting import plot
import matplotlib.pyplot as plt
import time
import copy


class PredictorCorrector:
    def __init__(self, c, A, b, x0, s0,y0=0, eps=1e-6, phi=0.99995, alpha=0.75):
        self.c = c
        self.A = A
        self.b = b
        self.x = x0
        self.y=y0
        self.s = s0
        self.m = self.A.shape[0]
        self.n = self.x.shape[0]
        if self.y == 0:
            self.y=np.linalg.pinv(self.A.T)@(self.c-self.s)
        self.eps = eps
        self.phi=phi
        self.alpha = alpha
        self.mu = self.alpha*self.x*self.s/self.n
        self.iter=0
        self.xList=[self.x]
        self.yList=[self.y]
        self.sList=[self.s]
        self.dualGapList=[np.sum(self.x*self.s)]
        

    def calculate(self):
        oneVec=np.ones(self.n)
        tempVec1 = self.b-self.A@self.x
        tempVec2 = self.c - self.A.T@self.y -self.s
        tempVec3 = self.mu * oneVec - self.x*self.s
        self.resVec = np.concatenate((tempVec1, tempVec2,tempVec3)) 
        
        tempMat1 = np.concatenate((self.A, np.zeros((self.m, self.m)), np.zeros((self.m, self.n))), axis=1)
        tempMat2 = np.concatenate((np.zeros((self.n, self.n)), self.A.T , np.identity(self.n)), axis=1)
        tempMat3 = np.concatenate((np.diag(self.s), np.zeros((self.n, self.m)) , np.diag(self.x)), axis=1)
        self.Hess = np.concatenate((tempMat1, tempMat2,tempMat3), axis=0) 


    def forward(self):
        self.calculate()
        res = np.linalg.solve(self.Hess, self.resVec)
        # Chia kết quả thành vector hướng Newton deltaX và biến đối ngẫu y
        self.deltaX = np.asarray(res[0:self.n])
        self.deltaY = np.asarray(res[self.n:self.n+self.m])
        self.deltaS = np.asarray(res[self.n+self.m:])

    def update(self):
        self.forward()
        tempCoefPri=[-self.x[i]/self.deltaX[i] for i in range(self.n)] ## Tính toán hệ số -x_i/ deltaX_i
        tempCoefPri2=np.asarray([x for x in tempCoefPri if x >0],dtype=np.float32) ## Chọn các hệ số lớn hơn mà có deltaX <0 
        tempCoefPri2=tempCoefPri2*self.phi ## Tránh hệ số x_i chạm đến biên
        coefPri = np.amin(np.append(tempCoefPri2,1)) ## Chọn hệ số dương lớn nhất có thể mà không khiến x bị âm
        self.x += coefPri*self.deltaX 
        tempX = copy.deepcopy(self.x)
        self.xList.append(tempX)

        tempCoefDual=[-self.s[i]/self.deltaS[i] for i in range(self.n)] ## Tính toán hệ số -s_i/ deltaS_i
        tempCoefDual2=np.asarray([s for s in tempCoefDual if s >0],dtype=np.float32) ## Chọn các hệ số lớn hơn mà có deltaS <0 
        tempCoefDual2=tempCoefDual2*self.phi ## Tránh hệ số s_i chạm đến biên
        coefDual = np.amin(np.append(tempCoefDual2,1)) ## Chọn hệ số dương lớn nhất có thể mà không khiến s bị âm
        self.y += coefDual*self.deltaY
        tempY = copy.deepcopy(self.y)
        self.yList.append(tempY)

        self.s += coefDual*self.deltaS   # Tính biến bù đối ngẫu
        tempS = copy.deepcopy(self.s)
        self.sList.append(tempS)
        self.mu = self.alpha*self.x*self.s/self.n
        self.iter += 1
        self.dualGapList.append(np.sum(self.x*self.s))

    def solve(self):
        self.update()
        print(f"Duality gap in iteration {self.iter} is {self.dualGapList[self.iter]}")
        while np.sum(np.abs(self.x*self.s))>self.eps: # Điều kiện dừng tính khoảng cách đối ngẫu
            self.update() 
            print(f"Duality gap in iteration {self.iter} is {self.dualGapList[self.iter]}")               
        print(f"The solution to the original problem is {self.x} in {self.iter} iterations")


    def plotConvergence(self):
        plt.step(np.arange(self.iter+1),self.dualGapList,where='post')
        plt.show()