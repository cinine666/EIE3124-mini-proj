# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:35:24 2024

@author: Administrator
"""


import pandas as pd
import matplotlib.pyplot as plt

names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset = pd.read_csv('boston.csv', skiprows=1,names=names)
dataset = dataset.iloc[1:]

n = dataset.shape[0]

xi = dataset['LSTAT'].sum()
zi = dataset['RM'].sum()
yi = dataset['MEDV'].sum()

xi_square = (dataset['LSTAT']*dataset['LSTAT']).sum()
zi_square = (dataset['RM']*dataset['RM']).sum()
yi_square = (dataset['MEDV']*dataset['MEDV']).sum()

yixi = (dataset['MEDV']*dataset['LSTAT']).sum()
yizi = (dataset['MEDV']*dataset['RM']).sum()

xizi = (dataset['LSTAT']*dataset['RM']).sum()

w2 = ( ( n*yi*xi_square)-(n*yizi*xi*xi)-(yi*zi*xi_square)+(yi*zi*xi*xi)-(n*n*yixi*xizi)+(n*yixi*xi*zi)+(n*xizi*yi*xi)-(xi*xi*yi*zi) ) / ( (zi*xi*xizi) + (n*zi*xi*xizi) - (n*xizi*xizi) - (xi*zi*xi*zi) + (n*zi_square*xi_square) - (n*zi_square*xi*xi) - (zi*zi*xi_square) + (zi*zi*xi*xi))

w1 = ( (n*yixi) - (yi*xi) - (w2*xizi) + (w2*zi*xi)) / ( (xi_square) - (xi*xi))

w0 = (1/n) * (yi - w1*xi - w2*zi)

def regression (x,z):
    return w0 + w1 * x + w2*z

predict = []

for i in range(n) :
    temp_x = dataset['LSTAT'][i+1]
    temp_z = dataset['RM'][i+1]
    predict.append(regression(temp_x,temp_z))
    
plt.plot(predict,'bo',label = 'predict')
plt.plot(dataset['MEDV'],'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
