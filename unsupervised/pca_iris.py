from sklearn.datasets import load_iris

#loading data
iris_dataset = load_iris()
X = iris_dataset['data']
y = iris_dataset['target']

#pre processing: z = (x - u) / s, u = mean, s = sd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)


print("PCA components: ", pca.components_)
print("Variance : ", pca.explained_variance_ratio_)

#plotting with pandas

import pandas as pd
df = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Target': y})

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.pairplot(x_vars=["PCA1"], y_vars=["PCA2"], data=df, hue="Target")
plt.show()
