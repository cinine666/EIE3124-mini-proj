# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset = pd.read_csv('boston.csv', skiprows=1,names=names)
dataset = dataset.iloc[1:]

def k_fold (dataset,k):
    
    results = []
    models = []
    subsets = np.array_split(dataset, k)
    
    for subset in subsets:
        testing_set = subset
        training_set = dataset.drop(testing_set.index)
        model = training(training_set)
        predict = []
        n = training_set.shape[0]
        for index, row in testing_set.iterrows():
            temp_x,temp_z,temp_y = row['LSTAT'],row['RM'],row['MEDV']
            predict.append(regression(temp_x,temp_z,model))
        r_square = r2_score(testing_set['MEDV'],predict)
        results.append(r_square)
        models.append(model)
    print(results)
    
    best_result = max(results)
    index_of_best_model = results.index(best_result)
    
    return models[index_of_best_model]
        
    
def training(dataset):
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
    
    return (w2,w1,w0)

def regression (x,z,model):
    w2,w1,w0 = model 
    return w0 + w1 * x + w2*z

X = dataset[['LSTAT', 'RM']].values
y = dataset['MEDV'].values
# divide training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
X_train_df = pd.DataFrame(X_train,columns=['LSTAT', 'RM'])
Y_train_df = pd.DataFrame(y_train,columns=['MEDV'])
training_data = pd.concat([X_train_df,Y_train_df],axis = 1)

X_test_df = pd.DataFrame(X_test,columns=['LSTAT', 'RM'])
Y_test_df = pd.DataFrame(y_test,columns=['MEDV'])
testing_data = pd.concat([X_test_df,Y_test_df],axis = 1)

model_k = k_fold(dataset,5)

model_k
predict = []
    
n = testing_data.shape[0]
for i in range(n) :
    temp_x = testing_data['LSTAT'][i]
    temp_z = testing_data['RM'][i]
    predict.append(regression(temp_x,temp_z,model_k))
 
plt.plot(predict,'bo',label = 'predict')
plt.plot(testing_data['MEDV'],'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
plt.show()
    
r_square = r2_score(testing_data['MEDV'],predict)
print('k-fold :',r_square)


model = training(training_data)

predict = []

n = testing_data.shape[0]
for i in range(n) :
    temp_x = testing_data['LSTAT'][i]
    temp_z = testing_data['RM'][i]
    predict.append(regression(temp_x,temp_z,model))

plt.plot(predict,'bo',label = 'predict')
plt.plot(testing_data['MEDV'],'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()



r_square = r2_score(testing_data['MEDV'],predict)
print(r_square)
