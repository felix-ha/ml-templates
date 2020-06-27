# source of data https://www.imf.org/external/pubs/ft/weo/2020/01/weodata/download.aspx


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def save_plot(df, country, variable):

    df_country = df[df['Country'] == country]
    x = df_country[df_country['Subject Descriptor'] == variable]
    

    x = x.replace('--', np.nan)
    x = x.replace(',', '')
    x = x.iloc[:, 9:49].squeeze().astype('float')
    
    
    title = country + ' - ' + variable.replace("/", " ")
    
    plt.plot(x)
    plt.xticks(np.arange(0, 40, 2), rotation=90)
    plt.title(title)
    #plt.show()
    
    
    file_name = title + '.png'
    
    plt.savefig(os.path.join('plots', file_name), dpi = 100)
    plt.close()



file = r"C:\Users\hauer\Dropbox\CFDS\Project\data\WEO\WEOApr2020all.csv"

df = pd.read_csv(file)

columns = df.columns

available_countries = df['Country'].unique()
available_variables = df['Subject Descriptor'].unique()

df_available_countries = df.iloc[:,[1,3]].drop_duplicates()



country =  'Germany'
variable = 'Unemployment rate'


available_countries = ['Germany']
for country in available_countries:
    
    df_country = df[df['Country'] == country]
    available_variables = df_country['Subject Descriptor'].unique()
    
    for variable in available_variables:
    
        try:
           save_plot(df, country, variable)
        except Exception as e:
            pass



