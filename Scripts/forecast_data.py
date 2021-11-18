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

access_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Imwzc1EtNTBjQ0g0eEJWWkxIVEd3blNSNzY4MCIsImtpZCI6Imwzc1EtNTBjQ0g0eEJWWkxIVEd3blNSNzY4MCJ9.eyJhdWQiOiJodHRwczovL3N0b3JhZ2UuYXp1cmUuY29tIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvZjc2MTkzNTUtNmM2Ny00MTAwLTlhNzgtMTg0N2YzMDc0MmUyLyIsImlhdCI6MTYzNzIyNzAzMCwibmJmIjoxNjM3MjI3MDMwLCJleHAiOjE2MzcyMzE3MjQsImFjciI6IjEiLCJhaW8iOiJFMlpnWUlqNHZaYWxWK0JpeGd0OWRZMjBEMTNMZGk1ZHZOVDRwb2FZMitVamdzYXQ3VThCIiwiYW1yIjpbInB3ZCIsInJzYSJdLCJhcHBpZCI6ImQ5Y2Q1MjBlLTIzMTctNGRiNi1hNWFlLTc3ZjA5NDkwODVhZiIsImFwcGlkYWNyIjoiMCIsImRldmljZWlkIjoiMzExMjdiYjEtZjk4Yy00Y2I5LWIzN2EtYmVkNjc0YzFlMjUxIiwiZmFtaWx5X25hbWUiOiLDmHN0ZXJnYWFyZCBIYW5zZW4iLCJnaXZlbl9uYW1lIjoiTWF0aGlhcyIsImdyb3VwcyI6WyJmODc3YmEwMy01OTZmLTQ2YzktOTE1YS03OGZiNDI5ZmI0NmMiLCJmMDNjMmFkOS0xMWE2LTQwMjEtODEzYS1iMTI0NzQ3ZDAwMGEiLCIxOWUyMjA1OC05OTdlLTRjYWMtODQyNS1hYzUxMDRjMTI2MjIiLCJiNDk5NzAxNi1hYTkzLTRjMzEtOTNkYi0wZWYwODJmMmJmNzEiLCI2OTlhZDRhYS1lN2ZhLTQzY2UtODMzZS03ODVmMjg3MGZmOTkiLCJkYzczNjllNi04YWY0LTRhNjItOTc1ZS04MDM1MWM0NTAxYzMiLCI2MWM3NGQxNC05NTUzLTQzNTEtYWY0YS1kZTAzYjkyYzJmZGIiLCIxNDk1OWQ1MS1iMDMzLTRlMjUtYWQ4NS0yMWEzZDNkNmQzMjIiLCI1NTljZDI1ZS1kZDEwLTRiNjUtYmQ1Yi00MGEwYzU2ZTZjOWQiLCI2NDllZGNkZC04ZTdjLTRiMGQtYmJjOC0xMjFhMzNlOWFkMzQiLCI4NGZmOWUwZS1kZTE4LTRlZDktODkwNS01ZTY1ZjJmZDlhMmEiLCI2ZDllOWFlZi1mMDM3LTRkNTktOWZlZi1mZTQ1ZjI3MDAzY2MiLCJkYTY5YzIyYi0xYWIwLTQwZGEtOTU3OS0wMGI3YzJhZTUyNWMiLCIwMmFkOGJjYi02MGM2LTQ5NjgtYmM1MS0yMDY3ZjJhZTg4MTYiLCI0NTdiMDIzOC05YTY2LTQ4YTktYjk4Zi0zYWMwMzNhNjRiZjYiLCJkZjEzNGMyZC1mMjNhLTRjZmQtYTFjOC1lYjUyODMyNTJiZDAiLCJkN2YyNmE3My05YTk4LTQ2NDgtOTViYS00MDI1ZjU0M2Y1ZGQiLCI4ODQyYzc2Ni00NzhlLTRmY2QtODkwMi00YWFkMGMyZjdiYTciLCIwOGVmZjAyZS00MTAyLTQ1ZjctYjBiOS1hNTI3NThhYzNhNmEiLCIwYzBjM2NkZi0wYmE4LTQ2MDgtYWIxNi0yOTZhYjZhZWNmNjkiLCI3MGI4Y2I3Yy1jZjhjLTQ4MDgtODIwYi1iMTQ4YmY4YmE1MmYiLCJhYTQyZmEwMi1iMjAyLTRkYjUtOTYzOS0yMzU5YjIzM2E4MjIiLCJiZDkxMTJhNy0xMjA3LTRhZTQtYmRmNi1jOGIxN2VmNDZiYjciLCIyYTUzYWZiZC0wMGQ4LTQ0OTQtOTc2Mi1hODdjZWU4OTFhNmEiLCJhNDk1MTk4Ny01MmE1LTRlZWYtOGNlMy01OGNjNzg2ODA2NDIiLCJiOTUwNTgzMS1hMWZjLTRiMjYtYjZiYy0xYzExM2I5YTFjZTciLCJiMjk1YWU4Mi1iODUzLTRhNTktOGUwNi01ZmU1Y2MwZDMyYWYiLCI0YjQ4ZmQ2OC02ZDQzLTQ3ZGMtODQxOC0xZjMwZmU0OGVhNDEiLCJiYjRiY2YxMi1lMjFkLTQwMGEtYTAxZS04MDlhMGFmOWZjNmIiLCJmN2ZlMjJiYi05ZTFlLTRkZGQtOGE3MC0xYzE0NGMxZTVmZGUiLCI3NTVjNjdmMy0wNTRjLTQ1OTAtYjZlYS1lOWZjYjY4NWEyMmYiLCJiYmNlZGViNi1kYTE3LTRmOGQtYWQ3YS04NGViNjg3MWZlMWMiLCI4N2JmYjkyNC1lYWZhLTRmNGMtYjM3My1iYjYxOGM4MzkzZDciLCJmOGI4MjQ5OC0yZDE4LTRkZGMtOThhYi0xM2Y1YjUwYzgwYTgiLCI5MmU0OWEyOS05NGRlLTQwYmItYTE3OS00MGRjYmQ2YjliYzkiLCI3NTlkZjM1OS03ZjA2LTRiMzktYmQ5My02ZjdlMTgwMmEwYmMiLCI4YTI3YzhiYS1mYzAyLTQwODctODU3My00ZTQwNWU0MWI2YmMiLCJkN2E5ODM0My03NmQzLTQxOTMtYTgwMi02ZjdkMTJhNjJmMGYiLCJlZWRkMjk0Ni0wMGI2LTQxZTYtYTY1MC02NDViNWU1NzQ1YzMiLCI5YzE2MWRjMi00NzUxLTRmNzYtOTAyNi0xMzkwMGM3MGMwYzIiLCI3NjQ4NGJlOS0zYWU2LTQ0NzctYmQ2YS1hYWM3ZDI0NmY4NGMiLCI3NjhiNDEwMS0xODRjLTRjYzQtYTRlMi04MzNiYTRlMWU3NTgiLCIxOWM2OGExMi0yYTdjLTQyMjUtOGY1Yi1mMDFmMTk2ZTFmOTgiLCJjNjg5MmRmNC01YzQ1LTRjN2EtODE0OC1lMWI5MTE2YjYyMDIiLCI4Mjc3MmZiNS1lZmI5LTQ2NmUtODU0Yi0wN2Y2ZDEyYzU0ZWUiLCIyMzliYWRmYS05OGIxLTRlMjEtYmI0MC1lODI0MWM0MzMyY2EiLCJmNjFkYzcxZi0xNTlmLTQwNmMtOGM1Mi00N2MzYjZjYjY4ZmUiLCJmMWU1ZWIyYi03MDVmLTRmN2QtOTg3Mi1mNjI4ZmU2OTYxNTUiLCI5YmY4NWE4NC0zMjAzLTQ4YWMtOTAzMy1iM2U5ZThiZTFiNTkiLCJiNWU5N2ZkMC0xOGMzLTRjOWMtYTQ5NC01NWFhYTM2OTRlYzkiLCI1MDY2OTkyMi1iMjIyLTQ1OGEtYjVjMi0yNjQwNjI3YzMzYWUiLCIxNzRhYWZmMy0zYjk2LTQ2NTItOWY1NC0yZTc0NWM5YmM0MDMiLCI4MGNjNzRmMy0wMmEzLTQ2ZDEtOTRlNi00YTViODFhODk2OTAiLCIwOGEyNmQ0ZC1iNDZhLTRhNjctYjU2NC02MzhiY2JlNmJmNGYiLCJlMGYzZDhjYy0xOTBmLTQyMmMtOTE0OS0wODlmNWVlMGI2MzkiLCIxZGNlMmVhMy1iZmUyLTQ2MmYtOGRmMy1mYjRiNzFiOTQxZjEiLCIyM2I5N2Y5Yi1lOTVkLTQwY2EtODQwZi00ZjJhNGVmODdkZjciLCJlZGY5ODNlNy04OGEzLTQxMzUtYTNmMi01ZWQ4NTQ5NDBkNTciLCIyM2M5ZjU5ZS1hZWUwLTRhMjAtOTcwNi0zYTc4N2M5NTJiNzEiLCJjZTk3YTAyZC0yYmI4LTRhOWMtODZjYi1iNjFjYTAyNzM5ZDQiLCIzZWYzZjFmNy1iOWEzLTQ4MWUtYWQ3Ni0yZGVkYmVjNDMzNWEiLCI2ZThmMTNiZC1mMDBjLTQxMjEtYTZlMy1mZWE4ZjJlNmYzOGYiLCI1Nzk4MzY0Ni1jMWJmLTQxYmEtYmU3Ny1iODBiOTBiMzQ1YjciLCI5MTVhMjI1YS1lNmVkLTQyYzItOTg4NC0xZmZmMjBiZDhkOWEiLCIxNDg0MmE4OC04Y2Q1LTRlYjYtYjUyOS0yZDAzMDdiOGI3ZWYiLCJlOGI1ZjE2NC03MDRkLTRhNDQtOTc5ZC0zZTYwMzE0YTY1MjkiLCIyMWFkZDJmNy00NDgwLTQ1OGYtYWQ2Yi0wY2ExODk1NmNjZmUiLCJmNmZiZDRkMC1mZTkyLTQ5MTItOTk5ZS03Y2JlN2EyNzE2NTMiLCI1ZWIxNGJjYS02ODYyLTRjY2ItYjhhZS04YjU4ZWJlNmQwZjgiLCI4ZGUxZDVjMy04Y2YyLTQ3YWQtYTU3MC04OTM3ZmQ1MzNkYTQiLCI5OGJkNDgyYi03Mzk0LTRhOGYtODAyMi1mOThhYzA1YzI1N2YiLCI0OTFkNTEyMi1hZDYzLTQwNTctOTFkNC02ZGNlYmRlYmUzMGQiLCI3MzI0YzVkMC1lNjc2LTQxZTgtOGI4NS1hY2QyMzU5YzlhYTUiLCI5ZGJiMDc5Yi1iYzhjLTRjYzktYmY5NS1jMTcyOTNiOWU1Y2YiLCJjMDUxY2M5YS02ZjNhLTQ4NjAtYTc0Zi05ZTlmYmI3YTdhMTgiLCJlZmNkZjUxYS1kOGFmLTRhNzAtOGJlOC1kYjA2Y2UyMDQxNGMiLCI0ZDljOWIxZC1jZWM2LTRmYTctOWVhNS1jMWE4MGRmMjVhZjEiLCIyNTk5N2NjYi1hNmVmLTRkNWYtYmIxNy00YmMwZjM5YmJhNmYiLCJhMjVlMDdmNi0wOTAyLTQ1MTAtOTI0ZS0wNDIzYzM1NDU1ZWUiLCI1NzQ1MWVkYS04YThmLTQ0NWEtOGJjZC04ZjM5NWU1ZTA3NDkiLCI4OGI0OWQ4NC03OWI5LTRlZTktOWQ0Ny05NGZiOWE0YmZlMzMiLCI3ZjNmNTRmMS1jZDU2LTQ2ZDItYjFiNi0xYzY2YjMzZjNlZjEiLCIzYzRiZDk4OC01NzZmLTRjZTktYjc4Yi1lYWVlNzdkYjRjZGEiXSwiaXBhZGRyIjoiMTk0LjIzOS4yLjEwNiIsIm5hbWUiOiJNYXRoaWFzIMOYc3RlcmdhYXJkIEhhbnNlbiIsIm9pZCI6IjllOTA4ZDdmLTRiNmYtNGQxYy1hMDgyLWU5MTBjMzMxYWUzZCIsIm9ucHJlbV9zaWQiOiJTLTEtNS0yMS0yOTAxNDg2NTc0LTIxOTQ3NTQ0ODYtMTAyNTU0MjQ1MC05MzIwMiIsInB1aWQiOiIxMDAzMjAwMTRGQzE0QUZFIiwicmgiOiIwLkFRc0FWWk5oOTJkc0FFR2FlQmhIOHdkQzRnNVN6ZGtYSTdaTnBhNTM4SlNRaGE4TEFGYy4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJPX3kyT0tUX3pwRDFWT2VPZ1JYdXNaMEFCRHFQMUhkMVJjWURhNm5TWXN3IiwidGlkIjoiZjc2MTkzNTUtNmM2Ny00MTAwLTlhNzgtMTg0N2YzMDc0MmUyIiwidW5pcXVlX25hbWUiOiJNVEdAZW5lcmdpbmV0LmRrIiwidXBuIjoiTVRHQGVuZXJnaW5ldC5kayIsInV0aSI6Ik5KZ1FYWkdRZzBlRzVXNkJHZFpZQUEiLCJ2ZXIiOiIxLjAifQ.fJJ_fLLatk8bugFN9-XQU14b-bKFCibkoOup0wfjKXycjH-bbmvOYjjuRNKAxep5UCERNMyv_lA5nSXOXwalUSW8PErPoaTWzJYIJXw9V9DCXIb5EfvfkMB7oWAD_dtbntA1QhXL3gDgU7H5Ak6Mdg8WXyHfOVqtEzS25AqrRFZzzY97aH2GR99RzNTU-khdM5BowlyIImVzIslZCSEhY6p-d8DWXs2varwzkJ1GSIeaaTfYGYQ9dWom8y7BNPfA-0vrB5fB5XNCaCidYwk9crdHVwtW7ezKm7cDbjRtHy9P9iiMihID10P3fyNVjs36oxnoFkH2nMUaA58BTUv2dQ'

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
                                            to_date='2020-01-02')
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
