from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def main():

    x, y = load_data('features.csv')

    #Normalize data
    x_n = normalize(x)
    x_n = np.array(x_n)
    #PCA
    pca = PCA(n_components=22)
    pca.fit(x_n)
    x_pca = pca.transform(x_n)
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=0)

    model = tf.keras.Sequential()

    #54-12-2-1 FFN
    model.add(tf.keras.layers.Input(shape = (22,)))
    model.add(tf.keras.layers.Dense(units = 12, activation ='tanh'))
    model.add(tf.keras.layers.Dense(units = 2, activation = 'tanh'))
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

    #Implements BP+M and variable learning rate
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics =['accuracy','AUC'])
    
    #Train
    history = model.fit(x_train, y_train, epochs = 250)

    #Evaluate on test data
    prediction = model.predict(x_test)

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    plot_loss_accuracy(loss, accuracy)
    evaluator(prediction, y_test)
    plot_roc_auc_curve(prediction, y_test)



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

def plot_loss_accuracy(loss,accuracy):
    """
    Generate loss andd accuracy graph after training
    """
    fig,ax = plt.subplots()
    ax.plot(range(len(accuracy)), accuracy, 'b', label = 'Training Accuracy')
    ax.plot(range(len(loss)), loss, 'r', label ='Training Loss')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def evaluator(y_pred, y_test):
    """
    Displays confusion matrix and performance metrics for network
    """
    #confusion martix
    y_pred = np.round(y_pred,0)
    print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    #prediction results
    print(classification_report(y_test, y_pred))


def plot_roc_auc_curve(y_probs, y_test):
    """
    Generates ROC curve and computes AUC
    """
    fpr, tpr, threshold = roc_curve(y_test,  y_probs)
    auc = roc_auc_score(y_test, y_probs)
    plt.plot(fpr,tpr,label="Classifier, auc="+str(auc))
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
if __name__ == "__main__":
    main()
