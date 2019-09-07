from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from adaline import AdalineGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data(object):

    def init_data(self):
        self.df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        self.y = self.df.iloc[0:100, 4].values
        self.y = np.where(self.y == 'Iris-setosa', -1, 1)
        self.X = self.df.iloc[0:100, [0, 2]].values

    def standard_scaled(self):
        # x' = (Xj - mean j)/standard deviatiion j
        self.X_std = np.copy(self.X)
        self.X_std[:, 0] = (self.X[:, 0] - self.X[:, 0].mean()) / self.X[:, 0].std()
        self.X_std[:, 1] = (self.X[:, 1] - self.X[:, 1].mean()) / self.X[:, 1].std()

    def standard_split(self):
        # Splitting test and train data
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1, stratify=self.y)
        sc = StandardScaler()
        sc.fit(X_train)
        self.X_train_std = sc.fit_transform(X_train)
        self.X_test_std = sc.fit_transform(X_test)

    def view_initial_data(self):
        plt.scatter(self.X[:50, 0], self.X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(self.X[50:100, 0], self.X[50:100, 1], color='blue', marker='x', label='versicolor')
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        return plt

    # decision boundary
    def plot_decision_regions(self, X, y, classifier, resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, c1 in enumerate(np.unique(y)):
            plt.scatter(x=X[y == c1, 0],
                        y=X[y == c1, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=c1,
                        edgecolors='black')
        return plt
def Test():
    # get data
    ml_data = Data()
    ml_data.init_data()
    ml_data.view_initial_data()

    # test perceptron
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(ml_data.X, ml_data.y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    plot = ml_data.plot_decision_regions(ml_data.X, ml_data.y, classifier=ppn)
    plot.xlabel('sepal length')
    plot.ylabel('petal length')
    plot.legend(loc='upper left')
    plot.show()

    # Adaline implementation
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = AdalineGD(n_iter=10, eta=0.1).fit(ml_data.X, ml_data.y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('log(Sum Squared error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(ml_data.X, ml_data.y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Sum Squared error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

    # Adaline with feature scaling
    ml_data.standard_scaled()

    ada = AdalineGD(n_iter=10, eta=0.01)
    ada.fit(ml_data.X_std, ml_data.y)
    plot = ml_data.plot_decision_regions(ml_data.X_std, ml_data.y, classifier=ada)
    plot.title('Adaline - Gradinet Descent')
    plot.xlabel('sepal length')
    plot.ylabel('petal length')
    plot.legend(loc='upper left')
    plot.tight_layout()
    plot.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum Squared error')
    plt.show()

    # Adaline with stochastic gradient
    ada3 = AdalineGD(n_iter=15, eta=0.01, random_state=1)
    ada3.fit(ml_data.X_std, ml_data.y)

    plot = ml_data.plot_decision_regions(ml_data.X_std, ml_data.y, classifier=ada3)
    plot.title('Adaline - Stochastic Gradient Descent')
    plot.xlabel('sepal length')
    plot.ylabel('petal length')
    plot.legend(loc='upper left')
    plot.show()

    plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average cost')
    plt.show()