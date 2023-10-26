# Hyperparameter Tuning and Cross-Validation with TensorFlow: Fine-Tuning Neural Networks for CIFAR-100

## Introduction

In this tutorial, we will walk through the process of hyperparameter tuning and cross-validation for training a neural network model on the CIFAR-100 dataset using TensorFlow. We will also discuss the use of dropout for regularization and early stopping to prevent overfitting.

## Step 1: Importing Libraries

We'll start by importing the necessary libraries.

```python
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
```

## Step 2: Loading and Preprocessing Data

### Loading the CIFAR-100 dataset

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
```

### Preprocessing the Data

```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

## Step 3: Splitting Data

We will split the data into training, validation, and test sets.

```python
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
```

## Step 4: Creating the Neural Network Model

In this step, we will define a function `create_model` to build the neural network model. The architecture of the model includes dropout layers for regularization and L2 regularization to control overfitting. Let's name and explain each layer of the model:

```python
def create_model(reg_strength, learning_rate):
    model = tf.keras.Sequential()
```

- We start by creating a sequential model, which allows us to build the neural network by stacking layers one after the other.

### Layer 1: Input Layer

```python
    model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
```

- The input layer is a flatten layer that reshapes the input data into a 1D vector. It takes images of size 32x32 pixels with 3 color channels (RGB).

### Layer 2: Fully Connected Layer (Dense Layer)

```python
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
```

- This is a fully connected dense layer with 512 units. 
- The activation function used is Rectified Linear Unit (ReLU), which introduces non-linearity to the model.
- `kernel_initializer='he_normal'` initializes the weights of the layer using the He normal initialization method, which helps with training deep networks.
- `kernel_regularizer=tf.keras.regularizers.l2(reg_strength)` applies L2 regularization with a strength defined by the `reg_strength` hyperparameter. L2 regularization helps prevent overfitting by adding a penalty term to the loss function based on the magnitude of weights.

### Layer 3: Dropout Layer

```python
    model.add(tf.keras.layers.Dropout(0.3))
```

- The dropout layer introduces dropout regularization with a dropout rate of 0.3. Dropout randomly sets a fraction of input units to 0 during training, which helps prevent overfitting by reducing co-dependency between neurons.

### Layer 4: Fully Connected Layer

```python
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
```

- Another fully connected dense layer with 256 units and ReLU activation.
- It shares the same properties as the previous dense layer, including He normal weight initialization and L2 regularization.

### Layer 5: Fully Connected Layer

```python
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
```

- A third fully connected dense layer with 128 units and ReLU activation.
- Similar to the previous layers, it uses He normal initialization and L2 regularization.

### Layer 6: Output Layer

```python
    model.add(tf.keras.layers.Dense(20))
```

- The final layer is the output layer with 20 units, corresponding to the 20 superclasses in the CIFAR-100 dataset. 
- This layer doesn't use an activation function because it's a part of a classification problem, and the raw scores (logits) are used for calculating the loss.

### Compiling the Model

```python
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model
```

- We configure the model for training by specifying the optimizer, loss function, and evaluation metrics.
- The optimizer is Stochastic Gradient Descent (SGD) with a learning rate defined by the `learning_rate` hyperparameter and a momentum of 0.9.
- The loss function is sparse categorical cross-entropy, which is suitable for multi-class classification tasks.
- We also track the accuracy as an evaluation metric.
- The function returns the compiled model.

This neural network architecture, with dropout and L2 regularization, is designed to handle the CIFAR-100 classification task, and it aims to balance model complexity and overfitting while achieving good classification performance.

## Step 5: Hyperparameter Tuning with Cross-Validation

In this step, we will perform hyperparameter tuning by systematically exploring different combinations of learning rates and L2 regularization strengths. The primary goal is to find the optimal set of hyperparameters that results in the highest validation accuracy while avoiding overfitting. To achieve this, we'll use early stopping as a mechanism to halt training if the model starts to overfit the data.

Here's how the process works:

### Defining Hyperparameter Grid

```python
learning_rates = [0.01, 0.001]
reg_strengths = [0.001, 0.01]
```

- We define two lists: `learning_rates` and `reg_strengths`. These lists contain different values for the learning rate and L2 regularization strength that we want to explore. Hyperparameter tuning involves trying various combinations of these hyperparameters to find the best ones for our model.

### Initializing Best Model and Best Validation Accuracy

```python
best_model = None
best_val_acc = 0.0
```

- We initialize two variables, `best_model` and `best_val_acc`, to keep track of the model with the highest validation accuracy encountered during the hyperparameter tuning process. These will be updated as we explore different hyperparameter combinations.

### Nested Loop for Grid Search

```python
for learning_rate in learning_rates:
    for reg_strength in reg_strengths:
        model = create_model(reg_strength, learning_rate)
```

- We use nested loops to iterate through all possible combinations of learning rates and regularization strengths. For each combination, we create a new neural network model using the `create_model` function with the current values of the hyperparameters.

### Early Stopping Setup

The details of early stopping setup are typically configured when creating the model (inside the `create_model` function). Early stopping is a crucial technique to prevent overfitting. It monitors a specified validation metric (in this case, "val_accuracy") and stops training when the monitored metric ceases to improve or worsens for a certain number of epochs.

### Model Training

Within the nested loops, the model is trained using the training data. Training includes optimizing the model's weights and biases to minimize the loss function.

```python
history = model.fit(x_train, y_train, epochs=100, batch_size=256, 
                    validation_data=(x_val, y_val), callbacks=[early_stopping])
```

- `x_train` and `y_train` are the training data and labels.
- `epochs` and `batch_size` are set for training.
- `validation_data` is used for monitoring the model's performance on the validation set.
- `callbacks` include early stopping to halt training if necessary.

### Tracking Validation Accuracy

```python
val_acc = max(history.history['val_accuracy'])
```

- After training, we extract the maximum validation accuracy achieved during the training process.

### Updating Best Model and Best Validation Accuracy

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_model = model
```

- We compare the current validation accuracy (`val_acc`) with the best validation accuracy (`best_val_acc`) seen so far. If the current model performs better, we update `best_val_acc` and `best_model` to store the details of the best model configuration found during the hyperparameter tuning process.

This process continues until all combinations of learning rates and regularization strengths have been explored. The final result will be the model configuration that yielded the highest validation accuracy, and this model will be used for evaluation on the test dataset in the next step.

## Step 6: Evaluate the Best Model

We will evaluate the best model found during hyperparameter tuning on the test data and print the test accuracy.

```python
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
```

## Conclusion

In this notebook, we've learned how to fine-tune a neural network for the CIFAR-100 dataset using hyperparameter tuning and cross-validation techniques. We've also explored the use of dropout and early stopping for better model performance and reduced overfitting.
