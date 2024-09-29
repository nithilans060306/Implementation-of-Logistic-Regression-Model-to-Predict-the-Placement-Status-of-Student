# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start
#### Step 2: Import Libraries
#### Step 3: Load the Dataset
#### Step 4: Split the Data
#### Step 5: Instantiate the Model
#### Step 6: Train the Model
#### Step 7: Make Predictions
#### Step 8: Evaluate the Model
#### Step 9: Stop


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nithilan S
RegisterNumber: 212223240108
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
df = pd.read_csv('Placement_Data.csv')
X = df.drop(columns=['sl_no','salary','status'])
Y = df['status']
regression = LogisticRegression(max_iter = 1000, tol = 1e-3)
minmax = MinMaxScaler()
col_minmax = ['ssc_p','hsc_p','degree_p','etest_p','mba_p']
for col in col_minmax:
    X[col] = minmax.fit_transform(X[[col]])
le = LabelEncoder()
col_encode = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']
for col in col_encode:
    X[col] = le.fit_transform(X[col])
Y = le.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
regression.fit(x_train, y_train)
y_pred = regression.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Output:

![image](https://github.com/user-attachments/assets/b954ba1a-fe20-48d7-b0b0-073d1794070a)

![image](https://github.com/user-attachments/assets/dc54745f-b41e-4f0c-abb4-3f9d3a65b1d9)

![image](https://github.com/user-attachments/assets/77ed32f2-05c0-47b6-a73f-612ba74d5973)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
