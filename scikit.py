from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

#Splitting test and train data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
x_train_std = sc.fit_transform(X_train)
x_test_std = sc.fit_transform(X_test)

# train perceptron
ppn = Perceptron( n_iter_no_change=60, eta0=0.1, random_state=1)
ppn.fit(x_train_std, y_train)
y_pred = ppn.predict(x_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy score: %.2f' % ppn.score(x_test_std, y_test))