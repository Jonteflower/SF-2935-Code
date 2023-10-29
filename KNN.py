import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from data_reader import read_training_data
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV  # <-- Import GridSearchCV

# Preprocess the data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)

def optimal_k_using_gridsearch(x, y):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 51)}
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)
    knn_gscv.fit(x, y)
    
    return knn_gscv.best_params_['n_neighbors']


# Plot KNN performance
def plot_knn_performance(x, y):
    k_values, accuracies, rmses = list(range(1, 51)), [], []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x, y, cv=5)
        
        accuracies.append(scores.mean())
        rmses.append(np.sqrt(((1 - scores) ** 2).mean()))

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, accuracies, color='tab:blue')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, rmses, color='tab:red')
    plt.xlabel('K Value')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.show()

# Read the training data
df_train, df_test = read_training_data(False)

# Preprocess data
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

x_train = df_train.drop('Label', axis=1)
y_train = df_train['Label']

# Plot performance
plot_knn_performance(x_train, y_train)

# Grid search
#optimal_k = optimal_k_using_gridsearch(x_train, y_train)

# Train and evaluate model
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
scores = cross_val_score(knn, x_train, y_train, cv=5)
print(f'Accuracy: {scores.mean():.02f}, Stdev: {scores.std():.02f}')

# Predict on df_test and save the predictions
preds = knn.predict(df_test).astype(int)
with open('result/prediction_KNN.csv', 'w') as f:
    f.write(','.join(map(str, preds)))
