from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix

def main():
    x, y = load_data('features.csv')

    x_pca = pca(x)
    x_pca = np.array(x_pca)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape = (54,)))
    #100 hidden nodes
    model.add(tf.keras.layers.Dense(units = 12, activation ='sigmoid'))

    model.add(tf.keras.layers.Dense(units = 2, activation = 'sigmoid'))

    #10 output nodes
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

    #Using adam over SGD, becasue it gives better results
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics =['accuracy'])
    
    model.fit(x_train, y_train, epochs = 50)

    prediction = model.predict(x_test)

    evaluator(prediction, y_test)



def load_data(file_name):
    with open(file_name, 'r') as infile:
        data = np.genfromtxt(file_name, delimiter=',')
        
    x = data[:,:-1]
    y = data[:,-1]

    return x,y

def pca(data):
    x = data
    # calculates the mean
    x -=tf.reduce_mean(data, axis=0)
    
    #eigen_values, eigen_vectors = tf.linalg.eigh(tf.tensordot(tf.transpose(x), x, axes=1))
    #x_pca = tf.tensordot(tf.transpose(eigen_vectors), tf.transpose(x), axes=1)

    return x

def evaluator(y_pred, y_test):
    #confusion martix
    y_pred = np.round(y_pred,0)
    print(confusion_matrix(y_test, y_pred))
    #prediction results
    print(classification_report(y_test, y_pred))

    
if __name__ == "__main__":
    main()
