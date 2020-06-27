import pandas as pd
import numpy as np
import os


def generate_subsequences(x, sequence_length):
    """
    Given a time series x, returns a prepared data set for a
    sequence-to-sequence model. 
    
    x = (x_0, x_1, ..., x_N) -->
     
    x_batch = (x_i+1, x_i+2, ..., x_sequence_length)
    y_batch = (x_i+2, x_i+3, ..., x_sequence_length+1)
    
    returns a tuple X, y with dimensions 
    [len(x) - sequence_length, sequence_length]
    
    """

    X, y = [], []
    
    for i in range(len(x)):
        x_start = i
        x_end = i+sequence_length     
       
        y_start = x_start+1
        y_end = x_end+1
        
        if y_end > len(x):
            break
        
        x_batch = x[x_start:x_end] 
        y_batch = x[y_start:y_end]
        
        X.append(x_batch)
        y.append(y_batch)
    
    return np.stack(X), np.stack(y)


def get_imf_data(country, year, remove_na=True):
    
    ##filtering weo data, to get dataframe with data from 1980 to 2019 

    path = r"C:\Users\hauer\Dropbox\CFDS\Project\data\IMF"
    
    
    
    result = pd.DataFrame()
    
    for file in os.listdir(path):
        
        variable = file
        
        if file[0:3] == 'WEO':
            continue
        
        print("  ")
        df =  pd.read_excel(os.path.join(path, file), skiprows=6)
        
        
        del df['Unnamed: 0']
        del df['Scale']
        try:
            del df['Base Year']
        except:
            pass
        
        try:
            del df['2020']
        except:
            pass
        
        df = df.replace('...', np.nan)
        df = df.replace('-', np.nan)
        
            # filter quaterly and monthly data
        for col in df.columns:
            if col[0] == '2' and ('Q' in col or 'M' in col):
                del df[col] 
        
        available_countries = df['Country'].unique()
        
        if country in available_countries:
            df = df[df['Country'] == country]
            
            
            df_curr = df.transpose()
            df_curr = df_curr.iloc[1:, :]
            df_curr = df_curr.rename({df_curr.columns[0]: variable}, axis='columns')
            
            df_curr = df_curr.loc[df_curr.index >= '1980']
            
            result = pd.concat([result, df_curr], axis=1)
            
    if remove_na:
        result = result.dropna(axis=1) 
        
    result.index = result.index.astype(dtype='int64') 
    result = result.astype('float')
    
    return result     


def get_imf_woe_data(country, remove_na=True):
    file = r"C:\Users\hauer\Dropbox\CFDS\Project\data\IMF_WEO\WEOApr2020all.csv"
    df = pd.read_csv(file)
    
    df = df[df['Country'] == country]
    
    result = pd.DataFrame()
    
    available_variables = df['Subject Descriptor'].unique()
    
    for variable in available_variables:
        df_curr = df[df['Subject Descriptor'] == variable]
        df_curr = df_curr.iloc[:, 9:49]
        df_curr = df_curr.transpose()
        df_curr = df_curr.rename({df_curr.columns[0]: variable}, axis='columns')
        result = pd.concat([result, df_curr], axis=1)
        
        
    if remove_na:
        result = result.dropna(axis=1) 
        
    return result


def get_gdp(country):
    df = get_imf_woe_data(country, remove_na=False)
    df.index = df.index.astype(dtype='int64')   
    df['Y'] = df['Gross domestic product, constant prices']  
    df['Y'] = df['Y'].str.replace(',', '').astype('float')
    return df['Y']


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


def get_predictions_weo(start_forecast, end_forecast):
    #  WEO Fall Year predicts GDP Growth in Year + 1
    # S = Spring
    # F = Fall
    
    path = r'C:\Users\hauer\Dropbox\CFDS\Project\data\IMF_WEO\WEOhistorical.xlsx'
    df =  pd.read_excel(path,sheet_name='ngdp_rpch')
    
    df = df[df['country'] == 'Germany']
    
    
    for col in df.columns:
        if 'S' in col:
            del df[col] 
            
    del df['WEO_Country_Code']     
    
    
    df = df[df['year'] >= start_forecast]
    
    
    predictions_weo = []
    years = np.arange(start_forecast, end_forecast+1, 1)
    
    for year in years:
       
        df_curr = df[df['year'] == year]
        
        year_WEO = year -1 
        column = 'F' + str(year_WEO) + 'ngdp_rpch'
        y_pred_year = df_curr[column].values[0]
        
        predictions_weo.append(y_pred_year)
    
    predictions_weo = pd.Series(data = predictions_weo, index = years)
    
    return predictions_weo


    


if __name__ == '__main__':
    
    country = 'Germany' 
    year = 1980 
    remove_na=True
    
    
    df_1 = get_imf_woe_data(country, remove_na)
    df_2 = get_imf_data(country, year, remove_na)
    
    
    df = pd.concat([df_1, df_2], axis=1)
    
    df['Y'] = df['Gross domestic product, constant prices'] 
    df['Y'] = df['Y'].str.replace(',', '')
        
      
    del df['Gross domestic product, constant prices']
    del df['Gross domestic product, current prices']  
    del df['Gross domestic product per capita, constant prices']
    del df['External_Sector_CurrentAccount.xls']
    
    
    df = df.astype('float')
    
    df_new = df.iloc[1:, :].copy()
    
    
    for variable in df_new.columns:
        if variable == 'Y':
            continue
        df_new[variable] = df[variable].iloc[:-1].values / df[variable].iloc[1:].values
        
        
    
    # model 
    
    X = df.iloc[:,:-1].values
    y = df['Y'].values
    
    
    from sklearn.ensemble import GradientBoostingRegressor   
    model = GradientBoostingRegressor(n_estimators = 100, max_depth = 2, min_samples_split=2, learning_rate = 0.05)
    model.fit(X,y)
        
    from sklearn.linear_model import LinearRegression
    #model = LinearRegression().fit(X, y)
    #r_squared = reg.score(X, y)
    
    
    from sklearn.metrics import mean_squared_error
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    
    print(mse)
    
    
    
    
    
    
