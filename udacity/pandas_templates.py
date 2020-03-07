import pandas as pd
import numpy as np


def convert_string_to_category_to_dummies():
    data = {'Attribute': ['X1', 'X2', 'X3', 'X1', 'X2', 'X3'],
            'Missing': [0.56, 0.45, 0.46, 0.60, 0.35, 0.48]}
    df = pd.DataFrame(data)
    print("df prior:")
    print(df)

    #strings are python object, convert them to category:
    df['Attribute'] = df['Attribute'].astype('category')
    df = pd.get_dummies(df)
    print("df after:")
    print(df)

from sklearn.impute import SimpleImputer
def impute_mean():
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = [[1, 2], [np.nan, 3], [7, 6]]
    imp.fit(X_train)
    X_test = [[np.nan, 2], [6, np.nan], [7, 6]]

    print("X_train")
    print(X_train)
    print("imputed X_test")
    print(imp.transform(X_test))


from sklearn.impute import KNNImputer
def impute_knn():
    imp = KNNImputer(n_neighbors=2, weights="uniform")
    X_train = [[1, 2], [np.nan, 3], [7, 6]]
    imp.fit(X_train)
    X_test = [[np.nan, 2], [6, np.nan], [7, 6]]

    print("X_train")
    print(X_train)
    print("imputed X_test")
    print(imp.transform(X_test))


# WARNING: biased imputing!!!!! dot not use
def impute_integers_categories():
    data = {'Attribute': [1,1,1,2,2, np.nan],
            'Missing': [0.56, 0.45, 0.46, np.nan, 0.35, 0.48]}
    df = pd.DataFrame(data)

    print(df)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df)
    imp.transform(df)

    df = pd.DataFrame(imp.transform(df), columns=df.columns)
    df['Attribute'] = df['Attribute'].round().astype(int)


    print(df)


# imputing categorical and numeric data separate, merging them back together and create dummies
def impute_by_dtype():
    data = {'Attribute': [1,1,1,2,2, np.nan],
            'Missing': [0.56, 0.45, 0.46, np.nan, 0.35, 0.48]}
    df = pd.DataFrame(data)
    df['Attribute'] = df['Attribute'].astype('category')


    print(df)
    print(df.dtypes)
    print("")

    # selecting only category columns and apply SimpleImputer for categories
    df_category = df.select_dtypes(include='category')
    imp = SimpleImputer(strategy="most_frequent")
    imp.fit(df_category)
    df_category = pd.DataFrame(imp.transform(df_category), columns=df_category.columns)
    df_category = df_category.astype('int')
    df_category = df_category.astype('category')


    # selecting only numeric columns and apply SimpleImputer for categories
    df_float = df.select_dtypes(include='float64')
    imp = KNNImputer(n_neighbors=2, weights="uniform")
    imp.fit(df_float)
    df_float = pd.DataFrame(imp.transform(df_float), columns=df_float.columns)



    #mergin whole df back together
    df_imputed = pd.concat([df_category, df_float], axis=1)

    #creating dummies
    df_final = pd.get_dummies(df_imputed)

    print(df_final)





