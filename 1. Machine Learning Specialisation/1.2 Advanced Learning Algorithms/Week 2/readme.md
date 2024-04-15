# Important Notes Week 2

## Neural Network Training

### Training a Network in TensorFlow

1. **Create The Model**

```Python
import tensorflow as tf
#importing the Sequential model type from the Keras library
#Sequential models are a standard way to build linear stacks of neural network layers
from tensorflow.keras import Sequential
#This imports the Dense layer type from Keras
#A Dense layer is a fully-connected layer, meaning all neurons in one layer are connected to all neurons in the next layer
from tensorflow.keras.layers import Dense

#defining a Sequential model in Keras
model = ([
    #creates the first Dense layer with 25, 15 and 1 neurons
    Dense(units = 25, activation = 'sigmoid'),
    Dense(units = 15, activation = 'sigmoid'),
    Dense(units = 1, activation = 'sigmoid'),
    ])
```

2. Loss And Cost Functions

```Python
#This line specifically imports the BinaryCrossentropy loss function from the tensorflow.keras.losses module.
#Binary Crossentropy is particularly suited for binary classification problems
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss = BinaryCrossentropy)# This line compiles our previously defined Keras model
"""
Here we initiate the training process of our neural network.
X: Your input data (features).
Y: Your target labels (the true values we want to predict).
epochs=100: The model will train for 100 epochs.
"""
model.fit(X,Y, epochs=100)
```
