import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(3,2))


def grouped_barplot():
    data = {'Attribute': ['X1', 'X2', 'X3', 'X1', 'X2', 'X3'],
            'Missing': [0.56, 0.45, 0.46, 0.60, 0.35, 0.48],
            'Dataset': ['A', 'A', 'A', 'B', 'B', 'B']}
    df = pd.DataFrame(data)

    ax = sns.barplot(x="Attribute", y="Missing", hue='Dataset', ci=None, data=df)
    ax.set_title("Grouped barplot")
    plt.show()

grouped_barplot()