# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate input and output variables.

2. Split the data into training and testing sets.

3. Train the linear regression model using the training data.

4. Predict the output for the test data and evaluate the results.
  

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: BUSHRA FATHIMA I
RegisterNumber: 212225040051 
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Startup.csv")

X = data['R&D Spend'].values
y = data['Profit'].values

X = (X - X.mean()) / X.std()

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = m * X + b
    
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

y_pred = m * X + b

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
```

## Output:
<img width="938" height="633" alt="Screenshot 2026-04-22 094719" src="https://github.com/user-attachments/assets/7639bb18-9830-4ff2-ac16-4cc0fea618af" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
