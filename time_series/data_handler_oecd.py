import os
import numpy as np
import pandas as pd

from data_handler import get_gdp, convert_time_series_to_relative, get_oecd_data



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



X = df.iloc[:,:-1].values
y = df['Y'].values



