#%%
import pandas as pd
from io import BytesIO
import osiris
from osiris.apis.egress import Egress
from osiris.core.azure_client_authorization import ClientAuthorization
from configparser import ConfigParser
#%%
config = ConfigParser()
config.read('conf.ini')

access_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Imwzc1EtNTBjQ0g0eEJWWkxIVEd3blNSNzY4MCIsImtpZCI6Imwzc1EtNTBjQ0g0eEJWWkxIVEd3blNSNzY4MCJ9.eyJhdWQiOiJodHRwczovL3N0b3JhZ2UuYXp1cmUuY29tIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvZjc2MTkzNTUtNmM2Ny00MTAwLTlhNzgtMTg0N2YzMDc0MmUyLyIsImlhdCI6MTYzNzIzMzU0MiwibmJmIjoxNjM3MjMzNTQyLCJleHAiOjE2MzcyMzkxNzYsImFjciI6IjEiLCJhaW8iOiJBU1FBMi84VEFBQUFaMlZzYmZvVUluSEc2K2Y2blpVU3BmY1BGZWNoYVZHdTVPRzYyZDlIM200PSIsImFtciI6WyJwd2QiLCJyc2EiXSwiYXBwaWQiOiJkOWNkNTIwZS0yMzE3LTRkYjYtYTVhZS03N2YwOTQ5MDg1YWYiLCJhcHBpZGFjciI6IjAiLCJkZXZpY2VpZCI6IjMxMTI3YmIxLWY5OGMtNGNiOS1iMzdhLWJlZDY3NGMxZTI1MSIsImZhbWlseV9uYW1lIjoiw5hzdGVyZ2FhcmQgSGFuc2VuIiwiZ2l2ZW5fbmFtZSI6Ik1hdGhpYXMiLCJncm91cHMiOlsiZjg3N2JhMDMtNTk2Zi00NmM5LTkxNWEtNzhmYjQyOWZiNDZjIiwiZjAzYzJhZDktMTFhNi00MDIxLTgxM2EtYjEyNDc0N2QwMDBhIiwiMTllMjIwNTgtOTk3ZS00Y2FjLTg0MjUtYWM1MTA0YzEyNjIyIiwiYjQ5OTcwMTYtYWE5My00YzMxLTkzZGItMGVmMDgyZjJiZjcxIiwiNjk5YWQ0YWEtZTdmYS00M2NlLTgzM2UtNzg1ZjI4NzBmZjk5IiwiZGM3MzY5ZTYtOGFmNC00YTYyLTk3NWUtODAzNTFjNDUwMWMzIiwiNjFjNzRkMTQtOTU1My00MzUxLWFmNGEtZGUwM2I5MmMyZmRiIiwiMTQ5NTlkNTEtYjAzMy00ZTI1LWFkODUtMjFhM2QzZDZkMzIyIiwiNTU5Y2QyNWUtZGQxMC00YjY1LWJkNWItNDBhMGM1NmU2YzlkIiwiNjQ5ZWRjZGQtOGU3Yy00YjBkLWJiYzgtMTIxYTMzZTlhZDM0IiwiODRmZjllMGUtZGUxOC00ZWQ5LTg5MDUtNWU2NWYyZmQ5YTJhIiwiNmQ5ZTlhZWYtZjAzNy00ZDU5LTlmZWYtZmU0NWYyNzAwM2NjIiwiZGE2OWMyMmItMWFiMC00MGRhLTk1NzktMDBiN2MyYWU1MjVjIiwiMDJhZDhiY2ItNjBjNi00OTY4LWJjNTEtMjA2N2YyYWU4ODE2IiwiNDU3YjAyMzgtOWE2Ni00OGE5LWI5OGYtM2FjMDMzYTY0YmY2IiwiZGYxMzRjMmQtZjIzYS00Y2ZkLWExYzgtZWI1MjgzMjUyYmQwIiwiZDdmMjZhNzMtOWE5OC00NjQ4LTk1YmEtNDAyNWY1NDNmNWRkIiwiODg0MmM3NjYtNDc4ZS00ZmNkLTg5MDItNGFhZDBjMmY3YmE3IiwiMDhlZmYwMmUtNDEwMi00NWY3LWIwYjktYTUyNzU4YWMzYTZhIiwiMGMwYzNjZGYtMGJhOC00NjA4LWFiMTYtMjk2YWI2YWVjZjY5IiwiNzBiOGNiN2MtY2Y4Yy00ODA4LTgyMGItYjE0OGJmOGJhNTJmIiwiYWE0MmZhMDItYjIwMi00ZGI1LTk2MzktMjM1OWIyMzNhODIyIiwiYmQ5MTEyYTctMTIwNy00YWU0LWJkZjYtYzhiMTdlZjQ2YmI3IiwiMmE1M2FmYmQtMDBkOC00NDk0LTk3NjItYTg3Y2VlODkxYTZhIiwiYTQ5NTE5ODctNTJhNS00ZWVmLThjZTMtNThjYzc4NjgwNjQyIiwiYjk1MDU4MzEtYTFmYy00YjI2LWI2YmMtMWMxMTNiOWExY2U3IiwiYjI5NWFlODItYjg1My00YTU5LThlMDYtNWZlNWNjMGQzMmFmIiwiNGI0OGZkNjgtNmQ0My00N2RjLTg0MTgtMWYzMGZlNDhlYTQxIiwiYmI0YmNmMTItZTIxZC00MDBhLWEwMWUtODA5YTBhZjlmYzZiIiwiZjdmZTIyYmItOWUxZS00ZGRkLThhNzAtMWMxNDRjMWU1ZmRlIiwiNzU1YzY3ZjMtMDU0Yy00NTkwLWI2ZWEtZTlmY2I2ODVhMjJmIiwiYmJjZWRlYjYtZGExNy00ZjhkLWFkN2EtODRlYjY4NzFmZTFjIiwiODdiZmI5MjQtZWFmYS00ZjRjLWIzNzMtYmI2MThjODM5M2Q3IiwiZjhiODI0OTgtMmQxOC00ZGRjLTk4YWItMTNmNWI1MGM4MGE4IiwiOTJlNDlhMjktOTRkZS00MGJiLWExNzktNDBkY2JkNmI5YmM5IiwiNzU5ZGYzNTktN2YwNi00YjM5LWJkOTMtNmY3ZTE4MDJhMGJjIiwiOGEyN2M4YmEtZmMwMi00MDg3LTg1NzMtNGU0MDVlNDFiNmJjIiwiZDdhOTgzNDMtNzZkMy00MTkzLWE4MDItNmY3ZDEyYTYyZjBmIiwiZWVkZDI5NDYtMDBiNi00MWU2LWE2NTAtNjQ1YjVlNTc0NWMzIiwiOWMxNjFkYzItNDc1MS00Zjc2LTkwMjYtMTM5MDBjNzBjMGMyIiwiNzY0ODRiZTktM2FlNi00NDc3LWJkNmEtYWFjN2QyNDZmODRjIiwiNzY4YjQxMDEtMTg0Yy00Y2M0LWE0ZTItODMzYmE0ZTFlNzU4IiwiMTljNjhhMTItMmE3Yy00MjI1LThmNWItZjAxZjE5NmUxZjk4IiwiYzY4OTJkZjQtNWM0NS00YzdhLTgxNDgtZTFiOTExNmI2MjAyIiwiODI3NzJmYjUtZWZiOS00NjZlLTg1NGItMDdmNmQxMmM1NGVlIiwiMjM5YmFkZmEtOThiMS00ZTIxLWJiNDAtZTgyNDFjNDMzMmNhIiwiZjYxZGM3MWYtMTU5Zi00MDZjLThjNTItNDdjM2I2Y2I2OGZlIiwiZjFlNWViMmItNzA1Zi00ZjdkLTk4NzItZjYyOGZlNjk2MTU1IiwiOWJmODVhODQtMzIwMy00OGFjLTkwMzMtYjNlOWU4YmUxYjU5IiwiYjVlOTdmZDAtMThjMy00YzljLWE0OTQtNTVhYWEzNjk0ZWM5IiwiNTA2Njk5MjItYjIyMi00NThhLWI1YzItMjY0MDYyN2MzM2FlIiwiMTc0YWFmZjMtM2I5Ni00NjUyLTlmNTQtMmU3NDVjOWJjNDAzIiwiODBjYzc0ZjMtMDJhMy00NmQxLTk0ZTYtNGE1YjgxYTg5NjkwIiwiMDhhMjZkNGQtYjQ2YS00YTY3LWI1NjQtNjM4YmNiZTZiZjRmIiwiZTBmM2Q4Y2MtMTkwZi00MjJjLTkxNDktMDg5ZjVlZTBiNjM5IiwiMWRjZTJlYTMtYmZlMi00NjJmLThkZjMtZmI0YjcxYjk0MWYxIiwiMjNiOTdmOWItZTk1ZC00MGNhLTg0MGYtNGYyYTRlZjg3ZGY3IiwiZWRmOTgzZTctODhhMy00MTM1LWEzZjItNWVkODU0OTQwZDU3IiwiMjNjOWY1OWUtYWVlMC00YTIwLTk3MDYtM2E3ODdjOTUyYjcxIiwiY2U5N2EwMmQtMmJiOC00YTljLTg2Y2ItYjYxY2EwMjczOWQ0IiwiM2VmM2YxZjctYjlhMy00ODFlLWFkNzYtMmRlZGJlYzQzMzVhIiwiNmU4ZjEzYmQtZjAwYy00MTIxLWE2ZTMtZmVhOGYyZTZmMzhmIiwiNTc5ODM2NDYtYzFiZi00MWJhLWJlNzctYjgwYjkwYjM0NWI3IiwiOTE1YTIyNWEtZTZlZC00MmMyLTk4ODQtMWZmZjIwYmQ4ZDlhIiwiMTQ4NDJhODgtOGNkNS00ZWI2LWI1MjktMmQwMzA3YjhiN2VmIiwiZThiNWYxNjQtNzA0ZC00YTQ0LTk3OWQtM2U2MDMxNGE2NTI5IiwiMjFhZGQyZjctNDQ4MC00NThmLWFkNmItMGNhMTg5NTZjY2ZlIiwiZjZmYmQ0ZDAtZmU5Mi00OTEyLTk5OWUtN2NiZTdhMjcxNjUzIiwiNWViMTRiY2EtNjg2Mi00Y2NiLWI4YWUtOGI1OGViZTZkMGY4IiwiOGRlMWQ1YzMtOGNmMi00N2FkLWE1NzAtODkzN2ZkNTMzZGE0IiwiOThiZDQ4MmItNzM5NC00YThmLTgwMjItZjk4YWMwNWMyNTdmIiwiNDkxZDUxMjItYWQ2My00MDU3LTkxZDQtNmRjZWJkZWJlMzBkIiwiNzMyNGM1ZDAtZTY3Ni00MWU4LThiODUtYWNkMjM1OWM5YWE1IiwiOWRiYjA3OWItYmM4Yy00Y2M5LWJmOTUtYzE3MjkzYjllNWNmIiwiYzA1MWNjOWEtNmYzYS00ODYwLWE3NGYtOWU5ZmJiN2E3YTE4IiwiZWZjZGY1MWEtZDhhZi00YTcwLThiZTgtZGIwNmNlMjA0MTRjIiwiNGQ5YzliMWQtY2VjNi00ZmE3LTllYTUtYzFhODBkZjI1YWYxIiwiMjU5OTdjY2ItYTZlZi00ZDVmLWJiMTctNGJjMGYzOWJiYTZmIiwiYTI1ZTA3ZjYtMDkwMi00NTEwLTkyNGUtMDQyM2MzNTQ1NWVlIiwiNTc0NTFlZGEtOGE4Zi00NDVhLThiY2QtOGYzOTVlNWUwNzQ5IiwiODhiNDlkODQtNzliOS00ZWU5LTlkNDctOTRmYjlhNGJmZTMzIiwiN2YzZjU0ZjEtY2Q1Ni00NmQyLWIxYjYtMWM2NmIzM2YzZWYxIiwiM2M0YmQ5ODgtNTc2Zi00Y2U5LWI3OGItZWFlZTc3ZGI0Y2RhIl0sImlwYWRkciI6IjE5NC4yMzkuMi4xMDYiLCJuYW1lIjoiTWF0aGlhcyDDmHN0ZXJnYWFyZCBIYW5zZW4iLCJvaWQiOiI5ZTkwOGQ3Zi00YjZmLTRkMWMtYTA4Mi1lOTEwYzMzMWFlM2QiLCJvbnByZW1fc2lkIjoiUy0xLTUtMjEtMjkwMTQ4NjU3NC0yMTk0NzU0NDg2LTEwMjU1NDI0NTAtOTMyMDIiLCJwdWlkIjoiMTAwMzIwMDE0RkMxNEFGRSIsInJoIjoiMC5BUXNBVlpOaDkyZHNBRUdhZUJoSDh3ZEM0ZzVTemRrWEk3Wk5wYTUzOEpTUWhhOExBRmMuIiwic2NwIjoidXNlcl9pbXBlcnNvbmF0aW9uIiwic3ViIjoiT195Mk9LVF96cEQxVk9lT2dSWHVzWjBBQkRxUDFIZDFSY1lEYTZuU1lzdyIsInRpZCI6ImY3NjE5MzU1LTZjNjctNDEwMC05YTc4LTE4NDdmMzA3NDJlMiIsInVuaXF1ZV9uYW1lIjoiTVRHQGVuZXJnaW5ldC5kayIsInVwbiI6Ik1UR0BlbmVyZ2luZXQuZGsiLCJ1dGkiOiJ6VnVGQ2VYSEUwQ1N3VFJnWHBoYUFBIiwidmVyIjoiMS4wIn0.aGB_YKufSSWT1xA-HPfwYSZZ4H1w_zK4krp9QYZRUG-fqgliZhqq98JAKmbZl4T6-Qx4RNwScpZ1BAiux6w9GwgSwpEn2td2RfKoalSt1vmKvHnzY3PX7zjbjNxCLg2kSxRzV0ni4NWWuirUGpBy8MqCPCnDUNNOmcA7N63dbbBDt2Yv_ZpJb9EopkXKN6m5W_kArfHP4mwC_zS-4CuPdl89zvRAgdos2K__PRvwYXIeiKkvTDeDi6K10t9CQpqO6g8-mH_nAJiEFNjBtcS8iQtnKiVanyJ2DmQI3f7_PZDunKYIzNirWd3yUiWEAwWs5TN_zVrQdpoqF_TnsMcRyw'

client_auth = ClientAuthorization(access_token=access_token)

egress = Egress(client_auth=client_auth,
                egress_url=config['Egress']['url'])

#%%
coords = egress.download_dmi_list(from_date='2021-01')
#%%
"""
min and max values for latitude and longitude for DK1 and DK2 stored in variables
"""
#DK1
lat_min_dk1 = 54.8
lat_max_dk1 = 57.6
lon_max_dk1 = 11
lon_min_dk1 = 8.1
 
# DK2
lon_min_dk2 = 10.9
lon_max_dk2 = 12.8
lat_max_dk2 = 56.1
lat_min_dk2 = 55
# %%
"""
Divides the stations from the api request into dk1 and dk2 zones.
Uses the longitude and latitude to divide them.   

"""
dk1_stations = []
dk2_stations = []

for station in coords:
    if station['lat'] is not None: 
        if station['lat'] < lat_max_dk1 and station['lat'] > lat_min_dk1: 
            if station['lon'] < lon_max_dk1 and station['lon'] > lon_min_dk1:
                dk1_stations.append(station)
        if station['lat'] < lat_max_dk2 and station['lat'] > lat_min_dk2:
            if station['lon'] < lon_max_dk2 and station['lon'] > lon_min_dk2:
                dk2_stations.append(station)

#%%
stations_concat_df = pd.DataFrame()
for station in dk2_stations:
    parquet_content = egress.download_dmi_file(lon=station['lon'], lat=station['lat'],
                                            from_date='2021-01', 
                                            to_date='2021-03')
    data = pd.read_parquet(BytesIO(parquet_content))
    data_temp_val = get_station_temp_val(data)
    data_radi_val = get_station_radi_val(data)
    stations_concat_df = pd.merge(data_temp_val,data_radi_val, on='time')


#%%
parquet_content = egress.download_dmi_file(lon=15.19, lat=55.00,
                                            from_date='2021-01',
                                            to_date='2021-03',)
data = pd.read_parquet(BytesIO(parquet_content))
data.head()
#%%
def get_station_temp_val(station):
    values = []
    time = []
    predicted = []
    for index, row  in station.iterrows():
        if row['weather_type'] == 'temperatur_2m':
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
        
    stations_df = pd.DataFrame(columns=['temp_mean_1hr','predicted_ahead','time'])
    stations_df['temp_mean_1hr'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df
#%%
def get_station_radi_val(station):
    values = []
    time = []
    predicted = []
    for index, row  in station.iterrows():
        if row['weather_type'] == 'radiation_hour':
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead']+1)
    stations_df = pd.DataFrame(columns=['radiation_hour','predicted_ahead','time'])
    stations_df['radiation_hour'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df
# %%
get_station_temp_val(data)
# %%
"""
Henter forecast data fra stationen: Jærgersborg.
Jægersborg tilhører grid companiet Radius Elnet. 
Fører det sammen i et datasæt og omregner temperaturen fra Kelvin Celsius
"""
parquet_content = egress.download_dmi_file(lon=12.55, lat=55.7,
                                            from_date='2020-01-01', 
                                            to_date='2020-05-31')
data = pd.read_parquet(BytesIO(parquet_content))
data_temp_val = get_station_temp_val(data)
data_radi_val = get_station_radi_val(data)
stations_concat_df = pd.merge(data_temp_val,data_radi_val, 
                            how='outer', 
                            left_on=['time','predicted_ahead'], 
                            right_on=['time','predicted_ahead'])
stations_concat_df['temp_mean_1hr'] = stations_concat_df['temp_mean_1hr'].add(-273.15)

#%%
pkl_name = "data/forecast_data_dk2"+station_row+"dmi_data.pkl"
pd.to_pickle
# %%
