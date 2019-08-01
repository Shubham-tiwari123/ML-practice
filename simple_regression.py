import numpy as np
import pandas as pd
import sklearn as sk

# importing dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# fitting single linear regression to the training set
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit(X_train,y_train)

# predicting the test set
y_pred = regresser.predict(X_test)

# visualize the training set
"""import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regresser.predict(X_train), color='blue')
plt.title('Sal vs exp(training set)')
plt.xlabel('exp')
plt.ylabel('sal')
plt.show()"""

# visualize the test set
import matplotlib.pyplot as plt2
plt2.scatter(X_test,y_test,color='red')
plt2.plot(X_train, regresser.predict(X_train), color='blue')
plt2.title('Sal vs exp(test set)')
plt2.xlabel('exp')
plt2.ylabel('sal')
plt2.show()