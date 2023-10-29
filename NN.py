import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from data_reader import read_training_data

# 1. Read Training Data
df_train, df_test = read_training_data(True)

# Prepare data
Y = df_train["Label"]
X = df_train.drop('Label', axis=1)

# Number of epochs and learning rate
learning_rate = 0.005
epochs = 100

# Create the NN
def create_nn_model():
    model = Sequential()
    model.add(Dense(10, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Run cross validation with 5 folds
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
accuracies = []

for train_idx, val_idx in kfold.split(X, Y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    model = create_nn_model()
    model.fit(x=X_train, y=Y_train, epochs=epochs, verbose=0)  # Set verbose=0 to avoid verbose output
    loss, accuracy = model.evaluate(X_val, Y_val, verbose=0)
    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
variance = np.var(accuracies)

# Print the average accuracy and variance
print(f"Neural Network - Average Accuracy: {average_accuracy:.03f}, Variance: {variance:.03f}")

# Test the final model on all the data
final_model = create_nn_model()
final_model.fit(x=X, y=Y, epochs=epochs, verbose=0)  # Train on the entire dataset

# Save the predictions on the music dataste
pred_probs = final_model.predict(df_test)
preds = (pred_probs >= 0.5).astype(int).flatten()

with open('result/prediction_NN.csv', 'w') as f:
    f.write(','.join(map(str, preds)))
