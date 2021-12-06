from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

def main():

    x, y = load_data('features.csv')

    #Normalize data
    x_n = normalize(x)
    x_n = np.array(x_n)
    x_train, x_test, y_train, y_test = train_test_split(x_n, y, test_size=0.2, random_state=0)

    model = tf.keras.Sequential()

    #54-12-2-1 FFN
    model.add(tf.keras.layers.Input(shape = (54,)))
    model.add(tf.keras.layers.Dense(units = 12, activation ='tanh'))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'tanh'))
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

    #Implements BP+M and variable learning rate
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics =['accuracy'])
    
    #Train
    history = model.fit(x_train, y_train, epochs = 50)

    #Evaluate on test data
    prediction = model.predict(x_test)
    evaluator(prediction, y_test)



def load_data(file_name):
    """
    Loads data in from file
    """
    with open(file_name, 'r') as infile:
        data = np.genfromtxt(file_name, delimiter=',')
    x = data[:,:-1]
    y = data[:,-1]

    return x,y

def normalize(data):
    """
    Perform zero-mean standardaization on data
    """
    x = data
    # calculates the mean
    x -=tf.reduce_mean(data, axis=0)

    return x

def evaluator(y_pred, y_test):
    """
    Displays confusion matrix and performance metrics for network
    """
    #confusion martix
    y_pred = np.round(y_pred,0)
    print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    #prediction results
    print(classification_report(y_test, y_pred))

    
if __name__ == "__main__":
    main()
