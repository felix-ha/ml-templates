import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
from data_handler import get_gdp, convert_time_series_to_relative, get_oecd_data

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor 


def get_forecast_linear(df):
    X = df.iloc[:,:-1].values
    y = df['Y'].values
    
    model = LinearRegression()
    model.fit(X,y)
       
    y_pred = model.predict(X)
    
    result = pd.Series(data=y_pred, index = df.index)
    return result


def get_forecast_gbr(df):
    X = df.iloc[:,:-1].values
    y = df['Y'].values
    
    model = GradientBoostingRegressor(n_estimators = 150, max_depth = 3, min_samples_split=2, learning_rate = 0.025)
    model.fit(X,y)

       
    y_pred = model.predict(X)
    
    result = pd.Series(data=y_pred, index = df.index)
    return result


   

    



start = 1980
end = 2018

location =  'DEU'
country = 'Germany'



df = get_oecd_data(location, start, end)
df = convert_time_series_to_relative(df)


Y = get_gdp(country)

Y = Y[Y.index >= start+1]
Y = Y[Y.index <= end]

df = pd.concat([df, Y], axis=1)



y_pred_linear = get_forecast_linear(df)
y_pred_gbr = get_forecast_gbr(df)




fig, ax = plt.subplots()



ax.plot(df['Y'], label='real')
ax.plot(y_pred_linear, label='linear')
ax.plot(y_pred_gbr, label='gbr')


# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
#fig, ax = plt.subplots()  # Create a figure and an axes.
#ax.plot(x, x, label='linear')  # Plot some data on the axes.
#ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
#ax.plot(x, x**3, label='cubic')  # ... and some more.
#ax.set_xlabel('x label')  # Add an x-label to the axes.
#ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("GDP growth - real vs. forecast")  # Add a title to the axes.
ax.legend()  # Add a legend.