# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:58:16 2020

@author: MGA
"""
#%%
from requests import get
from pandas import DataFrame, to_datetime, merge
from datetime import datetime, timedelta
from math import floor
from numpy import timedelta64
from dmi import extract_dmi_data
import geojson as gj
#%%
def __calculateDanishEaster(year):
    #https://da.wikipedia.org/wiki/P%C3%A5ske
    a = year % 19
    b = floor(year / 100)
    c = year % 100
    d = floor( b / 4)
    e = b % 4
    f = floor((b + 8) / 25)
    g = floor((b - f + 1) / 3)
    h = (19 * a + b - d - g + 15) % 30
    i = floor(c / 4)
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = floor((a + 11 * h + 22 * l) / 451)
    n = floor((h + l - 7 * m + 114) / 31)
    p = (h + l - 7 * m + 114) % 31 + 1	
    return datetime(year, n, p)

#%%
def __danishHolidays(year):
    easterSunday = __calculateDanishEaster(year)
    easterThursday = easterSunday - timedelta(days = 3)
    easterFriday = easterSunday - timedelta(days = 2)
    easterMonday = easterSunday + timedelta(days = 1)
    #https://da.wikipedia.org/wiki/Store_bededag
    greatPrayerDay = easterSunday + timedelta(days = 26) 
    #https://da.wikipedia.org/wiki/Kristi_himmelfartsdag
    ascensionDay = easterSunday + timedelta(39)
    #https://da.wikipedia.org/wiki/Pinse
    penteCostSunday = easterSunday + timedelta(days = 49)
    penteCostMonday = easterSunday + timedelta(days = 50)
    first_jan = datetime(year, 1, 1)
    christmas = datetime(year, 12, 24)
    christmas1 = datetime(year, 12, 25)
    christmas2 = datetime(year, 12, 26)
    christmas3 = datetime(year, 12, 27)
    christmas4 = datetime(year, 12, 28)
    christmas5 = datetime(year, 12, 29)
    christmas6 = datetime(year, 12, 30)
    newyears = datetime(year, 12, 31)
    return [first_jan, easterThursday, easterFriday, easterSunday, easterMonday, greatPrayerDay, ascensionDay, penteCostSunday, penteCostMonday, christmas, christmas1, christmas2, christmas3, christmas4, christmas5, christmas6, newyears]
#    return [easterThursday, easterFriday, easterSunday, easterMonday, greatPrayerDay, ascensionDay, penteCostSunday, penteCostMonday]

#%%
def __read_market_data(date_start, date_end):    
    #number of needed rows 
    url = """https://api.energidataservice.dk/datastore_search_sql?sql=SELECT * FROM "electricitybalance" WHERE "HourUTC" >= '""" + date_start.strftime("%Y-%m-%dT%H:%M:%S") + """' AND "HourUTC" < '""" + date_end.strftime("%Y-%m-%dT%H:%M:%S") + """' """
    api_call = get(url)
    df = DataFrame(api_call.json()["result"]["records"]).drop(columns = ["_id", "_full_text"])
    df['HourUTC'] =  to_datetime(df['HourUTC'], format='%Y-%m-%dT%H:%M:%S+00:00')
    
    #extract relevant columns
    data_df = df[['HourUTC','PriceArea']].copy()
    data_df['Con'] = df['GrossCon'] - df['ElectricBoilerCon']  #subtract consumption from electric boilers
    data_df['Weekday'] = data_df['HourUTC'].dt.dayofweek #add weekday info to dataset
    hol = __danishHolidays(date_start.year)
    data_df['Holiday'] = data_df['HourUTC'].isin(hol).astype(int) #add holiday info to datast
    data_df = data_df[(data_df.HourUTC >= date_start) & (data_df.HourUTC < date_end)] #filter on time
    df_DK1 = data_df[data_df.PriceArea == 'DK1'] #split into DK1 and DK2
    df_DK2 = data_df[data_df.PriceArea == 'DK2']
    
    return  df_DK1, df_DK2

#%%
def __missing_time_stamps(df, time_col_name):    
    time_stamps = df[time_col_name].unique()
    time_stamps.sort()
    for i in range(len(time_stamps)):
        if i == 0:
            continue
        if (time_stamps[i] - time_stamps[i-1]) != timedelta64(3600000000000,'ns'):
            print("GAP : "  + str(time_stamps[i-1]) + " and " + str(time_stamps[i]))

#%%
def read_data(date_start, date_end, api_key):
    
    df_market_DK1, df_market_DK2 = __read_market_data(date_start, date_end)
    df_dmi_DK1, df_dmi_DK2 = extract_dmi_data(date_start, date_end, api_key)
    #check missing data    
    print("Check market data, DK1, for time gaps")
    __missing_time_stamps(df_market_DK1, 'HourUTC')
    print("Check market data, DK2, for time gaps")
    __missing_time_stamps(df_market_DK2, 'HourUTC')
    print("Check dmi data, DK1, for time gaps")
    __missing_time_stamps(df_dmi_DK1, 'time')
    print("Check dmi data, DK2, for time gaps")
    __missing_time_stamps(df_dmi_DK2, 'time')

    #merge data    
    df_DK1 = merge(df_market_DK1, df_dmi_DK1, how='inner', left_on = 'HourUTC', right_on='time').drop(columns = ['HourUTC', 'PriceArea'])
    df_DK2 = merge(df_market_DK2, df_dmi_DK2, how='inner', left_on = 'HourUTC', right_on='time').drop(columns = ['HourUTC', 'PriceArea'])

    return df_DK1, df_DK2


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
    return data # Convert JSON object to a Pandas DataFrame
    
#%%
#mean DK1: 06068 (05135 does not have all data)   
#mean DK2: 06174 
def __extract_dmi_station_data(stationId, date_start, date_end, api_key):
    data = __extract_parameter_timeseries(stationId, 'temp_mean_past1h', date_start, date_end, api_key)
    df = DataFrame(data)
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



#%%
date_start = datetime(2019,12,1)
date_end = datetime(2019,12,5)
data = __extract_parameter_timeseries('06072', 'temp_mean_past1h', date_start, date_end, "46481bd5-99c3-45df-b1b1-f2743b5277cf")
val = data['features'][7]['properties']['value']
#read_data(date_start,date_end,"46481bd5-99c3-45df-b1b1-f2743b5277cf")

# %%
