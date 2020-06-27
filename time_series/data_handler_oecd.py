import os
import numpy as np
import pandas as pd

from data_handler import get_gdp


def convert_time_series_to_relative(df):
    # Assings each t the Values of X_t / X_(t-1)
    # X_t will be dropped
    
    df_new = df.iloc[1:, :].copy()
    
    for variable in df.columns:
        df_new[variable] = df[variable].iloc[:-1].values / df[variable].iloc[1:].values
        
    return df_new
 
    

def get_oecd_data(location, start, end): 


    
    result = pd.DataFrame()
    
    path = r'C:\Users\hauer\Dropbox\CFDS\Project\data\OECD'
    
    for file_name in os.listdir(path):
    
        
        file = os.path.join(path, file_name)
        
        df_orig = pd.read_csv(file)
        unique_subjects = df_orig['SUBJECT'].unique()
    
        
        for subject in unique_subjects:
            
            
            df = df_orig.copy()
            df = df[df['LOCATION'] == location] 
            df = df[df['SUBJECT'] == subject]
            
            # if there is only one unique subject, the name is TOT
            if(len(unique_subjects) == 1):
                subject = file_name[:-4]
            
            
            df = df.rename({df.columns[6]: subject}, axis='columns')
            
            df = df.set_index('TIME')
            
            result = pd.concat([result, df[subject]], axis=1)
        
        
    result = result[result.index >= start]
    result = result[result.index <= end]
    result = result.dropna(axis=1)
    
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



X = df.iloc[:,:-1].values
y = df['Y'].values



