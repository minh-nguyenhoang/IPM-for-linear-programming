import numpy as np
import matplotlib.pyplot as plt
import time
import copy


class Primal:
    def __init__(self, c, A, b, x0, eps=1e-6, mu=2, alpha=0.75, progress="no"):
        self.c = c
        self.A = A
        self.b = b
        self.x = x0
        self.eps = eps
        self.mu = mu
        self.alpha = alpha
        self.m = self.A.shape[0]
        self.n = self.x.shape[0]
        self.iter=0
        self.xList=[self.x]
        self.sList=[]
        self.dualGapList=[self.mu]
        self.progress=progress
        

    def calculate(self):
        tempVec1 = self.mu/self.x - self.c
        tempVec2 = self.b-self.A@self.x
        self.resVec = np.concatenate((tempVec1, tempVec2)) # [mu*X^-1*e - c 
                                                           #     b - Ax]
        
        revX2 = self.mu*np.diag(1/self.x**2)
        tempMat1 = np.concatenate((revX2, -self.A.T), axis=1)
        tempMat2 = np.concatenate((self.A, np.zeros((self.m, self.m))), axis=1)
        self.Hess = np.concatenate((tempMat1, tempMat2), axis=0) #[[mu*X^-2 , -A^T],
                                                                 # [A       ,  0  ]]
    def forward(self):
        self.calculate()
        res = np.linalg.solve(self.Hess, self.resVec)
        # Chia kết quả thành vector hướng Newton deltaX và biến đối ngẫu y
        self.deltaX = np.asarray(res[0:self.n])
        self.y = np.asarray(res[self.n:])

    def update(self):
        self.forward()
        tempCoef=[-self.x[i]/self.deltaX[i] for i in range(self.n)] ## Tính toán hệ số -x_i/ deltaX_i
        tempCoef2=np.asarray([x for x in tempCoef if x >0],dtype=np.float32) ## Chọn các hệ số lớn hơn mà có deltaX <0 
        #print(tempCoef2)
        coef = np.amin(np.append(tempCoef2,1)) ## Chọn hệ số dương lớn nhất có thể mà không khiến x bị âm
        self.x += 0.2*self.deltaX
        tempX = copy.deepcopy(self.x)
        self.xList.append(tempX)
        self.s = self.c - self.A.T@self.y   # Tính biến bù đối ngẫu
        tempS = copy.deepcopy(self.s)
        self.sList.append(tempS)
        self.mu = self.alpha * self.mu 
        self.iter += 1
        self.dualGapList.append(np.sum(self.x*self.s))

    def solve(self):
        self.update()
        if self.progress in ["Y" , "y" ,"yes" ,"Yes"]:
            print(f"Duality gap in iteration {self.iter} is {self.dualGapList[self.iter]}")
        while np.sum(np.abs(self.x*self.s))>self.eps: # Điều kiện dừng tính khoảng cách đối ngẫu
            self.update() 
            if self.progress in ["Y" , "y" ,"yes" ,"Yes"]:
                print(f"Duality gap in iteration {self.iter} is {self.dualGapList[self.iter]}")               
        print(f"The solution to the original problem is {self.x} in {self.iter} iterations")


    def plotConvergence(self):
        plt.step(np.arange(self.iter+1),self.dualGapList,where='post')
        plt.show()


    def checkPrimalFeasible(self,eps=1e-3):
        A_new = np.concatenate([self.A,np.identity(self.A.shape[0])],axis=1)
        c_new=np.concatenate([np.zeros(A_new.shape[1]-A_new.shape[0]),np.ones(A_new.shape[0])])
        x0=np.ones(A_new.shape[1])
        p1 = Primal(c_new, A_new, self.b, x0,eps=eps)
        p1.solve()
        slack=np.round(p1.x[-A_new.shape[0]:],-np.log10(eps)-1)
        if np.sum(slack) == 0:
            return True
        else:
            return False



def checkPrimalFeasible(A,b,eps=1e-6):  ## Nếu bài toán có tồn tại tập chấp nhận được, thì hàm mục tiêu min 1^Ts vđk Ax+s=b có kq = 0
        A = np.concatenate([A,np.identity(A.shape[0])],axis=1)
        c=np.concatenate([np.zeros(A.shape[1]-A.shape[0]),np.ones(A.shape[0])])
        x0=np.ones(A.shape[1])
        p1 = Primal(c, A, b, x0,eps=eps,progress="no")
        p1.solve()
        p1.plotConvergence()
        slack=p1.x[-A.shape[0]:]
        if np.sum(slack)< A.shape[0]*eps:
            return True
        else:
            print("The primal problem is infeasible!")
            return False


def checkDualFeasible(A,c,eps=1e-6):  ## Bổ đề Farkas:
    b=np.zeros(A.shape[0])
    x0=np.ones(A.shape[1])
    p1 = Primal(-c, A, b, x0,eps=eps)  #min c^Tx --> max -c^Tx
    p1.solve()
    p1.plotConvergence()
    #print(np.sum(c*p1.x))
    if np.sum(c*p1.x) >= 0:
        return True
    else:
        print("The dual problem is infeasible!")
        return False
    


c = np.asarray([3, -3, 1, -1],dtype=np.float32)
x0 = np.asarray([1, 1, 1, 1],dtype=np.float32)
A = np.asarray([[-1,1,2,1],[1,1,-1,-1],[3,2,-6,3]])
b = np.asarray([2, 6, 9],dtype=np.float32)

#checkPrimalFeasible(A,b)
print(checkPrimalFeasible(A,b))



