# Developing a Neural Network Regression Model
## NAME:LOSHINI.G
## DEPARTMENT:IT
## REFERENCE NUMBER:212223220051
## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. A neural network regression model uses interconnected layers of artificial neurons to learn the mapping between input features and a continuous target variable. It leverages activation functions like ReLU to capture non-linear relationships beyond simple linear trends. Training involves minimizing the loss function (e.g., Mean Squared Error) through an optimizer (e.g., Gradient Descent). Regularization techniques like L1/L2 and dropout prevent overfitting. This approach offers flexibility and high accuracy for complex regression problems.

## Neural Network Model

![Screenshot 2024-03-05 194922](https://github.com/Loshini2301/basic-nn-model/assets/150007305/22271eaf-ec37-4e3e-a768-ff4aa31c2de9)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:LOSHINI.G
### Register Number:212223220051
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('exp no 1').sheet1
data=worksheet.get_all_values()
print(data)

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

import pandas as pd


from sklearn.model_selection import train_test_split

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()


Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

ai_brain =Sequential([
    Dense(units = 5,activation = 'relu',input_shape = [1]),
    Dense(units = 3,activation = 'relu'),
    Dense(units = 4,activation = 'relu'),
    Dense(units = 1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')

ai_brain.fit(X_train1,y_train,epochs=20)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()


X_test1 = Scaler.transform(X_test)


ai_brain.evaluate(X_test1,y_test)


X_n1 = [[30]]


X_n1_1 = Scaler.transform(X_n1)


ai_brain.predict(X_n1_1)


```
## Dataset Information

![Screenshot 2024-03-05 195757](https://github.com/Loshini2301/basic-nn-model/assets/150007305/8b94cb02-187e-4afd-be4d-2ff7c91eaca5)

## OUTPUT
### Training Loss Vs Iteration Plot

![image](https://github.com/Loshini2301/basic-nn-model/assets/150007305/87e975c9-62c2-4f6b-8172-b7b621bcd56d)


### Test Data Root Mean Squared Error

![Screenshot 2024-03-05 200359](https://github.com/Loshini2301/basic-nn-model/assets/150007305/da6c1941-9d78-408a-8a04-f02a0f57f574)


### New Sample Data Prediction

![Screenshot 2024-03-05 200359](https://github.com/Loshini2301/basic-nn-model/assets/150007305/062c5e1e-b41e-42ad-bc33-f1becb3e4075)


## RESULT

A neural network regression model for the given dataset has been developed successfully
