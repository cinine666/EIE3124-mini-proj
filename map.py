import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

#read features from dataset
from sklearn.model_selection import train_test_split
names = ['LSTAT','RM','MEDV']
dataset = pd.read_csv('boston.csv', usecols=names)


# division labels
X = dataset[['LSTAT', 'RM']].values
y = dataset['MEDV'].values
# divide training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
#linear regression
model = LinearRegression()
# fit the linear regression model
model.fit(X, y)
print(f"{model.coef_}{model.intercept_}")

#conduct MAP
class LinearRegressionMAP:
    def __init__(self):
        self.weights = None

    def train(self, train_x, train_y, lambd):
        bias = np.ones((train_x.shape[0], 1))
        X = np.concatenate((train_x, bias), axis=1)
        self.weights = np.linalg.inv(X.T.dot(X) + lambd * np.eye(X.shape[1])).dot(X.T).dot(train_y)

    def predict(self, test_x):
        bias = np.ones((test_x.shape[0], 1))
        X = np.concatenate((test_x, bias), axis=1)
        return X.dot(self.weights)
    def get_weights(self):
        return self.weights
map_model = LinearRegressionMAP()
lambd = 0.1
map_model.train(X_train, y_train, lambd)
w=map_model.get_weights()
print(w)
# Create a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data
ax.scatter(dataset['RM'], dataset['LSTAT'], dataset['MEDV'], color="blue", label='Actual MEDV')

# We need to create a mesh of x values (RM and LSTAT) to plot the surface.
rm = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), len(X))
lstat = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), len(X))
rm, lstat = np.meshgrid(rm, lstat)

# Now we use our model to predict the values across the grid
z = map_model.predict(np.c_[lstat.ravel(), rm.ravel()])
z = z.reshape(rm.shape)

# Plot the surface for predicted data
ax.plot_surface(rm, lstat, z, color="red", alpha=0.7, label="Predicted MEDV")

# Set labels and title
ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('MEDV')
ax.set_title('Actual vs Predicted MEDV')

# Show plot
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


y_pred1 = map_model.predict(X_test)
y_pred2 = map_model.predict(X_test)
# calculate MSE
mse1 = mean_squared_error(y_test, y_pred1)
mse2 = mean_squared_error(y_test, y_pred2)
print(mse1,mse2)

# calculate RMSE
rmse1 = mean_squared_error(y_test, y_pred1, squared=False)
rmse2 = mean_squared_error(y_test, y_pred2, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse2}")

# calculateMAE
mae = mean_absolute_error(y_test, y_pred1)
print(f"Mean Absolute Error (MAE): {mae}")

# R² to check model alienation capability
r1 = r2_score(y_test, y_pred1)
r2 = r2_score(y_test, y_pred2)
print(r1,r2)
plt.plot(y_pred1,'bo',label = 'predict')
plt.plot(y_test,'ro', label = 'actual value')
plt.ylabel('MEDV')
plt.legend()
plt.show()
