import pandas as pd
from os.path import join
from sklearn.tree import DecisionTreeClassifier

data_dir = '../datasets/plagiarism'

train_df = pd.read_csv(join(data_dir, 'train.csv'), header=None)
test_df = pd.read_csv(join(data_dir, 'test.csv'), header=None)

y_train = train_df.loc[:,0].values
X_train = train_df.loc[:,1:].values

y_test = test_df.loc[:,0].values
X_test = test_df.loc[:,1:].values

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

print(clf.feature_importances_)
print(clf.score(X_test, y_test))

test_y_preds = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true=y_test, y_pred=test_y_preds)

print(accuracy)


## print out the array of predicted and true labels, if you want
print('\nPredicted class labels: ')
print(test_y_preds)
print('\nTrue class labels: ')
print(y_test)