import numpy as np
import pandas as pd
import sklearn as sk

# importing dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dumy variable trap
X = X[:,1:]

# splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit(X_train,y_train)

# predicting the test case
y_pred = regresser.predict(X_test)

# optimissing the model
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
