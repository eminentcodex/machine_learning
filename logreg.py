# Logistic regression using scikit learn
from sklearn.linear_model import LogisticRegression
from main import Data

# initialize data
data = Data()
data.init_data()
data.standard_split()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(data.X_train_std, data.y_train)
plt = data.plot_decision_regions(data.X_train_std, data.y_train, lr)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()