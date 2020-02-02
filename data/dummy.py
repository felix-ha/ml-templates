import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

N = 100

x_0 = np.random.rand(int(N / 2))
y_0 = np.repeat(0, int(N / 2))

x_1 = np.random.rand(int(N / 2)) + 2
y_1 = np.repeat(1, int(N / 2))


d = {'X': np.concatenate((x_0, x_1)), 'y': np.concatenate((y_0, y_1))}

df = pd.DataFrame(data=d)


clf = LogisticRegression(random_state=0).fit(df['X'].values.reshape(-1, 1) , df['y'].values)

print(clf.predict(np.array([[1]])))
print(clf.predict_proba(np.array([[1]])))
print(clf.coef_)
print(clf.intercept_)