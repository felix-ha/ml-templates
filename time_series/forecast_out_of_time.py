import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
import numpy as np
import pandas as pd
from data_handler import get_gdp, convert_time_series_to_relative, get_oecd_data, get_predictions_weo, get_imf_data, get_imf_woe_data

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error


  
def forecast(model, df_training, df_prediction):
    X_train = df_training.iloc[:,:-1].values
    y_train = df_training['Y'].values
    
    X_predict = df_prediction.iloc[:,:-1].values
    y_predict = df_prediction['Y'].values
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_predict)

    mse = mean_squared_error(y_predict, y_pred)
    
    
    result = pd.Series(data=y_pred, index = df_prediction.index)
    return result, mse
    
    
 
start = 1980
end = 2018

location =  'DEU'
country = 'Germany'



df_oecd = get_oecd_data(location, start, end)

# oecd data only until 2018 availabel at the moment
df_weo = get_imf_data(country, start, remove_na=True)
df_weo = df_weo[df_weo.index < 2019]


df = pd.concat([df_oecd, df_weo], axis=1)


df = convert_time_series_to_relative(df)

# shift index to be according predicting method
df.index = df.index - 1

Y = get_gdp(country)

Y = Y[Y.index >= start]
Y = Y[Y.index <= end-1]

df = pd.concat([df, Y], axis=1)



start_forecast = 2010
t_train = df.index[df.index <= start_forecast]
t_forecast = df.index[df.index >= start_forecast]


df_training = df[df.index <= start_forecast]
df_prediction = df[df.index >= start_forecast]
y_predict = df_prediction['Y'].values


model = LinearRegression()
y_forecast_linear, mse_linear = forecast(model, df_training,df_prediction)

model = MLPRegressor(hidden_layer_sizes = (4,), max_iter=500)
y_forecast_mlp, mse_mlp = forecast(model, df_training,df_prediction)

model = GradientBoostingRegressor(n_estimators = 75, max_depth = 3, min_samples_split=2, learning_rate = 0.10)
y_forecast_gbr, mse_gbr = forecast(model, df_training,df_prediction)


y_forecast_woe = get_predictions_weo(start_forecast, end - 1)
mse_woe = mean_squared_error(y_predict, y_forecast_woe)



fig, ax = plt.subplots()

ax.plot(df['Y'], label='real')
ax.plot(t_forecast,y_forecast_woe, label='weo - mse: ' + str(round(mse_woe,2)))
#ax.plot(t_forecast,y_forecast_mlp, label='mlp - mse: ' + str(round(mse_mlp,2)))
ax.plot(t_forecast,y_forecast_gbr,    label='gbr    - mse: ' + str(round(mse_gbr,2)))

ax.axvline(x=start_forecast, ymin=0, ymax=1, color='black',linestyle='--', alpha=0.5)
ax.set_xlabel('year') 
ax.set_ylabel('change in %') 
ax.set_title("GDP growth - real vs. forecast")
ax.legend() 
ax.xaxis.set_ticks(np.arange(start, end, 2))
fig.autofmt_xdate()
plt.grid()


plt.savefig('forecast_out_of_time.png', dpi = 350)
plt.close()



