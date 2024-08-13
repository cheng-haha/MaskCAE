import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset.HAR_dataset import HARDataset

def read_data(file):
    data = pd.read_csv(file)
    
    # suffle data
    data = sklearn.utils.shuffle(data)
    
    X_data = data.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y_data = data.ActivityName
    
    return np.array(X_data), np.array(y_data)

def train_model(train_x, train_y, model_name='NaiveBayes', validation=None):
    """
    Possible model names: ['NaiveBayes', 'SVM', 'MLP', 'AdaBoost', 'BAG', 'RandomForest']
    default = 'NB'
    
    validation: (val_x, val_y) tupple for validation accuracy score.
    
    return: trained model
    """
    model = None
    if model_name == 'SVM':
        model = svm.SVC(gamma='scale', probability=True)
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=200, alpha=0.0001,
                     solver='sgd', verbose=10, tol=0.000000001)
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier(n_estimators=50)
    elif model_name == 'BAG':
        model = BaggingClassifier(n_jobs=2, n_estimators=50)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=200, max_depth=10)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    elif model_name == 'NaiveBayes':
        model = GaussianNB()
    else:
        raise RuntimeError(f"{model_name} not supported")
    model.fit(train_x, train_y)
    
    if validation is not None:
        y_hat = model.predict(validation[0])
        acc = metrics.accuracy_score(validation[1], y_hat)
        print(f"Validation Accuracy in '{model_name}' = {acc}")
        cm = metrics.confusion_matrix(validation[1], y_hat)
        print(cm)
        f1 = f1_score(validation[1], y_hat, average='weighted')
        print(f"F1 Score in '{model_name}' = {f1}")
               
    return model

def visualize(data, label):
    bs, nc, ws = data.shape

    accel = data[:, :nc // 2, :].reshape(bs, -1)
    gyro = data[:, nc//2:, :].reshape(bs, -1)
    
    tsne = TSNE(n_components=2)
    tsne_feat = tsne.fit_transform(np.concatenate((accel, gyro)))
    df = pd.DataFrame()
    df["y"] = label
    df["accel_comp-1"] = tsne_feat[:len(label), 0]
    df["accel_comp-2"] = tsne_feat[:len(label), 1]
    df["gyro_comp-1"] = tsne_feat[len(label):, 0]
    df["gyro_comp-2"] = tsne_feat[len(label):, 1]

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))
    sns.scatterplot(x="accel_comp-1", y="accel_comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 6),
                    data=df, ax=ax[0]).set(title="Accel feature T-SNE projection")
    sns.scatterplot(x="gyro_comp-1", y="gyro_comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 6),
                    data=df, ax=ax[1]).set(title="Gyro feature T-SNE projection")
    plt.savefig("tsne_visualize/UCI_HAR_raw.png")

if __name__ == "__main__":
    train = HARDataset('UCI_HAR', split='train', window_width=128)
    test = HARDataset('UCI_HAR', split='test', window_width=128)
    print(len(train))
    print(len(test))
#    train_model(train.data.reshape(train.data.shape[0], -1), train.label, model_name='RandomForest', validation=(test.data.reshape(test.data.shape[0], -1), test.label))
    visualize(test.data, test.label)