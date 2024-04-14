# Important Notes Week 2

## Neural Network Training

### Training a Network in TensorFlow

```Python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = ([
    Dense(units = 25, activation = 'sigmoid'),
    Dense(units = 15, activation = 'sigmoid'),
    Dense(units = 1, activation = 'sigmoid'),
    ])
```

```Python
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss = BinaryCrossentropy)
model.fit(X,Y, epochs=100)
```
