import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from data_reader import read_training_data

df_train, df_test = read_training_data(False)
df_train_X = df_train.iloc[:, 0:11]
df_train_y = df_train.iloc[:, 11]

# Function for LDA
def lda_classification():
    lda = LinearDiscriminantAnalysis()
    lda.fit(df_train_X, df_train_y)

    # Cross_validation
    scores = cross_val_score(lda, df_train_X, df_train_y, cv=5)
    print('LDA - Accuracy: %.02f, Stdev: %.02f' % (scores.mean(), scores.std()))
    
    y_pred = lda.predict(df_test)
    np.savetxt('result/prediction_LDA.csv', y_pred, delimiter=',', fmt='%d')
    
    return lda

# Function for QDA
def qda_classification():
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(df_train_X, df_train_y)

    # Cross_validation
    scores = cross_val_score(qda, df_train_X, df_train_y, cv=5)
    print('QDA - Accuracy: %.02f, Stdev: %.02f' % (scores.mean(), scores.std()))

    y_pred = qda.predict(df_test)
    np.savetxt('result/prediction_QDA.csv', y_pred, delimiter=',', fmt='%d')
    
    return qda

# Plotting function
def plot_models(lda, qda):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    
    # LDA Plot
    X_lda = lda.transform(df_train_X)
    axs[0].scatter(X_lda[df_train_y == 0], [0]*sum(df_train_y == 0), color='r', label='Class 0', marker='o')
    axs[0].scatter(X_lda[df_train_y == 1], [0]*sum(df_train_y == 1), color='b', label='Class 1', marker='x')
    axs[0].set_title("LDA Transformation")
    axs[0].legend()
    
    # QDA Plot - Showing predicted probabilities
    y_prob = qda.predict_proba(df_train_X)
    axs[1].scatter(y_prob[df_train_y == 0][:, 1], [0]*sum(df_train_y == 0), color='r', label='Class 0', marker='o')
    axs[1].scatter(y_prob[df_train_y == 1][:, 1], [0]*sum(df_train_y == 1), color='b', label='Class 1', marker='x')
    axs[1].set_title("QDA Predicted Probabilities for Class 1")
    axs[1].legend()

    plt.show()

# Call the functions
lda_model = lda_classification()
qda_model = qda_classification()
#plot_models(lda_model, qda_model)
