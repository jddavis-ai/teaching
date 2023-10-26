## Hyperparameter Tuning and Cross-Validation with TensorFlow: Fine-Tuning Neural Networks for CIFAR-100

This code is an implementation of a neural network using TensorFlow and scikit-learn for hyperparameter tuning with cross-validation. Let's break down the code and explain each part, including variables and techniques used:

1. **Importing Libraries**: The code starts by importing the necessary libraries, including TensorFlow, NumPy, and scikit-learn. These libraries are used for deep learning, data manipulation, and cross-validation.

2. **Loading Data**:
   ```python
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
   ```
   - `x_train` and `y_train` represent the training images and their corresponding labels.
   - `x_test` and `y_test` are the test images and labels.
   - The CIFAR-100 dataset is used with a "coarse" label mode, which provides 20 superclasses of object categories.

3. **Data Preprocessing**:
   ```python
   x_train = x_train.astype('float32') / 255.0
   x_test = x_test.astype('float32') / 255.0
   ```
   - The pixel values in the images are scaled to the range [0, 1] by dividing by 255.0. This preprocessing step is common to ensure all data falls within the same scale.

4. **Splitting Data**:
   ```python
   x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
   ```
   - The training data is split into training and validation sets. The validation set will be used for hyperparameter tuning.

5. **Creating the Model**:
   ```python
   def create_model(reg_strength, learning_rate):
       # Model architecture
   ```
   - A function `create_model` is defined to create a neural network model. The model has several layers, including dropout and L2 regularization.
   - It returns a compiled TensorFlow model with a specific configuration of regularization strength and learning rate.

6. **Hyperparameter Tuning with Cross-Validation**:
   ```python
   learning_rates = [0.01, 0.001]
   reg_strengths = [0.001, 0.01]
   best_model = None
   best_val_acc = 0.0
   ```
   - Two lists, `learning_rates` and `reg_strengths`, are defined to specify the hyperparameters to be tuned.
   - `best_model` and `best_val_acc` are used to keep track of the model with the highest validation accuracy.

7. **Nested Loops for Hyperparameter Grid Search**:
   ```python
   for learning_rate in learning_rates:
       for reg_strength in reg_strengths:
           model = create_model(reg_strength, learning_rate)
   ```
   - Nested loops iterate through different combinations of hyperparameters.
   - For each combination, a new model is created using the `create_model` function.

8. **Early Stopping**:
   ```python
   early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
   ```
   - Early stopping is set up as a callback to prevent overfitting. It monitors the validation accuracy and stops training if it doesn't improve for a specified number of epochs (10 in this case).

9. **Model Training with Cross-Validation**:
   ```python
   history = model.fit(x_train, y_train, epochs=100, batch_size=256, 
                       validation_data=(x_val, y_val), callbacks=[early_stopping])
   ```
   - The model is trained using the training data with a specified number of epochs and batch size.
   - Validation data is used for monitoring and early stopping.

10. **Tracking the Best Model**:
    ```python
    val_acc = max(history.history['val_accuracy'])
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
    ```
    - The validation accuracy from the training history is compared with the best validation accuracy seen so far, and the best model is updated if the current model performs better.

11. **Evaluating the Best Model**:
    ```python
    test_loss, test_acc = best_model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc * 100:.2f}%')
    ```
    - The code evaluates the best model on the test dataset and prints the test accuracy.

In summary, this code demonstrates how to perform hyperparameter tuning using cross-validation and early stopping to train a neural network model on the CIFAR-100 dataset. The grid search is conducted over learning rates and L2 regularization strengths, and the best model based on validation accuracy is chosen for evaluation on the test set.
