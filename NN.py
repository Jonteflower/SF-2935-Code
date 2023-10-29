import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)

from data_reader import read_training_data

# 1. Read Training Data
df_train, df_test = read_training_data(False)

# Split the dataset into training and validation set
df_train, test_data = train_test_split(df_train, test_size=0.2)

# 2. Preprocess the Data
Y_Train = df_train["Label"]
Y_Test = test_data["Label"]

# These columns provided a better result
columns_to_drop = ["loudness","mode", 'Label']

X_Train = df_train.drop(columns=columns_to_drop)
X_Train = (X_Train - X_Train.mean()) / X_Train.std()

X_Test = test_data.drop(columns=columns_to_drop)
#Use X_Train's mean and std for scaling X_Test
X_Test = (X_Test - X_Train.mean()) / X_Train.std()

#Set the number of epochs and leartning rate
learning_rate = 0.005
epochs = 100

# 3. Define the Neural Network Model
def create_nn_model():
    model = Sequential()
    model.add(Dense(10, input_dim=len(X_Train.columns), activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model


model = create_nn_model()

# Train the model
model.fit(x=X_Train, y=Y_Train, epochs=epochs)

# Evaluate the model for accuracy directly
loss, model_accuracy = model.evaluate(X_Test, Y_Test)

# Get predictions
predictions = model.predict(X_Test)

# Convert predictions and true labels to binary 0 or 1 based on threshold
predicted_classes = [1 if pred >= 0.5 else 0 for pred in predictions]
true_classes = Y_Test.tolist()

# Calculate manual accuracy 
correct_predictions = sum([1 for predicted, true in zip(predicted_classes, true_classes) if predicted == true])
manual_accuracy = correct_predictions / len(true_classes)

# Print out accuracy and standard deviation in the desired format
print(f"Neural Network - Accuracy: {model_accuracy:.02f}, Stdev: {predictions.std():.02f}")
