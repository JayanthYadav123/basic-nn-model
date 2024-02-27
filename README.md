# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
```
Input layer: 1 neuron.
First hidden layer: 3 neurons and ReLU activation.
Second hidden layer: 2 neurons.
Output layer: 1 neuron.
```
## Neural Network Model

<img width="631" alt="image" src="https://github.com/JayanthYadav123/basic-nn-model/assets/94836154/1aa6bfc5-74f7-4cad-885e-9503f0eed245">

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
### Name: G Jayanth.
### Register Number: 212221230030.
```python


from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp01data').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

X = df[['Input']].values
y = df[['Output']].values



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model_AI = Sequential([
    Dense(units = 3, activation = 'relu', input_shape=[1]),
    Dense(units = 2),
    Dense(units = 1)
])

model_AI.compile(optimizer= 'rmsprop', loss="mse")
model_AI.fit(X_train1,y_train,epochs=5000)
model_AI.summary()
loss_df = pd.DataFrame(model_AI.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
model_AI.evaluate(X_test1,y_test)
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
model_AI.predict(X_n1_1)

```
## Dataset Information

<img width="194" alt="image" src="https://github.com/JayanthYadav123/basic-nn-model/assets/94836154/ce107a23-0ed2-4622-a3f9-784445e068d6">

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/JayanthYadav123/basic-nn-model/assets/94836154/7fe84eb8-0655-407e-b847-86665aed9cc5)

### Test Data Root Mean Squared Error

<img width="421" alt="image" src="https://github.com/JayanthYadav123/basic-nn-model/assets/94836154/1e297b0d-5dae-4eb8-ae0c-b5c6a466ac5a">

### New Sample Data Prediction

<img width="314" alt="image" src="https://github.com/JayanthYadav123/basic-nn-model/assets/94836154/3fa7f639-646e-4804-8d73-7ceb29cd4f5d">

## RESULT

Thus, The Process of developing a neural network regression model for the created dataset is successfully executed.

