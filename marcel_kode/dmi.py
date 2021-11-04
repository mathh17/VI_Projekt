# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:44:19 2020

@author: MGA
"""


####################### SELECTED STATIONS START ########################
#%%
# Import necessary libraries
from requests import get# library for making HTTP requests
from pandas import to_datetime, DataFrame# library for data analysis

#%%
def __datetime_to_unixtime(dt):
    '''Function converting a datetime objects to a Unix microsecond string'''
    return str(int(to_datetime(dt).value*10**-3))


    
#%%
""" 
def __extract_parameter_timeseries(stationId, parameterId, date_start, date_end, api_key):    
    url = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items' # url for the current api version    
    params = {'api-key' : api_key,
              'datetime' : __datetime_to_unixtime(date_start),
              'datetime' : __datetime_to_unixtime(date_end),
              'stationId' : stationId, #'06068', '06174'
              'parameterId' : parameterId,          
              'limit' : '10000000',
              } """
def __extract_parameter_timeseries(stationId, parameterId, date_start, date_end, api_key):    
    url = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items' # url for the current api version    
    params = {'api-key' : api_key,
              'datetime' : '2019-01-01T00:00:00Z/2020-01-01T00:00:00Z',
              'stationId' : stationId, #'06068', '06174'
              'parameterId' : parameterId,          
              'limit' : '300000',
              } 
    r = get(url, params=params) # Issues a HTTP GET request
    print(r)
    data = r.json()
    print(data)
    return DataFrame(data) # Convert JSON object to a Pandas DataFrame
    
#%%
#mean DK1: 06068 (05135 does not have all data)   
#mean DK2: 06174 
def __extract_dmi_station_data(stationId, date_start, date_end, api_key):
    df = __extract_parameter_timeseries(stationId, 'temp_mean_past1h', date_start, date_end, api_key).rename(columns={'value' : 'temp'})
    print(df.head)
    df['precip'] = __extract_parameter_timeseries(stationId, 'precip_past1h', date_start, date_end, api_key)['value']
    df['wind'] = __extract_parameter_timeseries(stationId, 'wind_speed_past1h', date_start, date_end, api_key)['value']
    df['time'] = to_datetime(df['timeObserved'], unit='us') # The unit 'us' corresponds to microseconds

    df_reduced = df.drop(df[df.time.dt.minute != 0].index)    
    df_reduced = df_reduced.drop(columns = ["_id", "parameterId", "stationId", "timeCreated", "timeObserved"])

    return df_reduced

#%%
def extract_dmi_data(date_start, date_end, api_key):
    stationId_DK1 = '06072'
    stationId_DK2 = '06174'    
    df_DK1 = __extract_dmi_station_data(stationId_DK1, date_start, date_end, api_key)
    df_DK2 = __extract_dmi_station_data(stationId_DK2, date_start, date_end, api_key)
    return df_DK1, df_DK2 

######################## SELECTED STATIONS END #########################


# %%
