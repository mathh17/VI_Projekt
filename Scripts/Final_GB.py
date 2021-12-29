#%%
#Load all the packages
import sklearn
import os
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import Holidays_calc as hc
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from osiris.egress import Egress
from osiris.azure_client_authorization import ClientAuthorization
from configparser import ConfigParser
#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
#path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts\data\stations_data_dk2'
path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts\data\stations_data_dk2'

os.chdir(path)

temp_conc_data = pd.DataFrame(columns=['time'])
radi_conc_data = pd.DataFrame(columns=['time'])

# goes through all the files one by one adding them all together to create a Dataframe with one column for each station
for file in os.listdir():
    df = pd.read_pickle(file)
    file_name = os.path.basename(file)
    if 'temp_mean_past1h' in df.columns:
        temp_conc_data = pd.merge(temp_conc_data,df[['time','temp_mean_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))
    if 'radia_glob_past1h' in df.columns:
        radi_conc_data = pd.merge(radi_conc_data,df[['time','radia_glob_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))

# takes all the columns and calculates the mean for each row. which gives us a mean value for all stations at the given time.
temp_conc_data['mean'] = temp_conc_data.iloc[:,1:12].sum(axis=1) / 11
radi_conc_data['mean'] = radi_conc_data.iloc[:,1:7].sum(axis=1) / 6
dk2_mean = pd.DataFrame()
dk2_mean['time'] = temp_conc_data['time']
dk2_mean['temp_mean_past1h'] = temp_conc_data['mean']
dk2_mean['radia_glob_past1h'] = radi_conc_data['mean']
dk2_mean.head()

# Read Enernginet Pickle Data
# Change back path
old_path = r'C:\Users\oeste\OneDrive\Uni\DS_3_semester\VI_Projekt\Scripts'
os.chdir(old_path)
df_el_data = pd.read_pickle("data/jaegerspris_el_data.pkl")
el_data_2021 = pd.read_pickle("data/jaegerspris_el_data_2021.pkl")
#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_el_data['HourUTC'] = pd.to_datetime(df_el_data['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_el_data = df_el_data.rename(columns={'HourUTC':'time', 'HourlySettledConsumption':'Con'})
conc_data = pd.merge(dk2_mean, df_el_data, on='time', how='outer')
conc_data.dropna(inplace=True)
conc_data = conc_data.iloc[::-1]
conc_data = conc_data.sort_values(['time'])
#%%
#Calling the holiday function to build a column for if its a holiday or not
def holidays(df):
    holidays = []
    for i, row in df.iterrows():
        is_holiday = hc.get_date_type(row['time'])
        holidays.append(is_holiday)
    return holidays
def data_encoder(df): 
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
    df['is_holiday'] = holidays(df)
    return df
#%%
#Take data from the concatenated dataset and put it into label data and train data
pred_data = pd.DataFrame(conc_data[['temp_mean_past1h','radia_glob_past1h']])
conc_data = data_encoder(conc_data)
pred_data['is_holiday'] = conc_data['is_holiday']
conc_data['time'] = conc_data['time'].dt.hour
cat_time = pd.get_dummies(conc_data['time'])
pred_data = pred_data.join(cat_time)
values = conc_data['Con']
#%%
X_train, X_test, y_train, y_test = train_test_split(pred_data, values, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)

#%%
# Declaring hyperparameters, 
# initializing the model, 
# fitting it to the data,
# predicting on the validation data.
losses = 'huber'
lrs = 0.01
estimators = 500
crits = 'mse'
depth = 10
s_samples = 1

results = []
# The for loops goes through the different hyperparameters
gbt = ensemble.GradientBoostingRegressor(loss=losses,
                                        learning_rate=lrs,
                                        n_estimators=estimators,
                                        criterion=crits,
                                        max_depth=depth,
                                        subsample=s_samples)
gbt.fit(X_train,y_train)
y_val_hat = gbt.predict(X_val)
acc = gbt.score(X_val,y_val)
# the results from the different hyperparameters are then stored so we can get the best one
results.append([acc,losses,lrs,estimators,crits])
results_df = pd.DataFrame(results)
results_df.columns=['accuracy',"loss functions","learning_rate","n_estimators","criterions"]
#prints the results from best to worst in regards to accuracy, listting the hyperparameters for the result
results_df = results_df.sort_values('accuracy', ascending=False)
results_df
# %%
#Calculates the MSE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(y_val.to_numpy(),24)
mse_naive_y_val = np.sum((y_val-naive_y_val)**2)/len(y_val)
mse_naive_y_val
#%%
#Calculates the MSE for the predicted values from the validation set.
np.sum((y_val-y_val_hat)**2)/len(y_val)

#%%

y_test_hat = gbt.predict(X_test)
test_acc = gbt.score(X_test,y_test)
test_acc
# %%
# Plots the deviance in the prediction and on the training data. To visualize how the model learns and behaves.
best_est = results_df.iloc[0,3]
test_score = np.zeros((best_est,), dtype=np.float64)
for i, y_pred in enumerate(gbt.staged_predict(X_test)):
    test_score[i] = gbt.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(best_est) + 1, gbt.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(best_est) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
# %%
# Plots the predicted values with the exact values to compare how the model predicts.
y_plot = pd.DataFrame()
y_plot['exact_values'] = y_val
y_plot['predicted_values'] = y_val_hat
y_plot = y_plot.sort_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(range(len(y_val)), y_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(range(len(y_val)), y_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()

#%%

def get_station_temp_val(station):
    values = []
    time = []
    predicted = []
    counter = 0
    for index, row  in station.iterrows():
        if row['weather_type'] == 'temperatur_2m' and row['predicted_ahead']  == counter%49:
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
            counter+=1
        
    stations_df = pd.DataFrame(columns=['temp_mean_1hr','predicted_ahead','time'])
    stations_df['temp_mean_1hr'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df

def get_station_radi_val(station):
    values = []
    time = []
    predicted = []
    counter = 0
    for index, row  in station.iterrows():
        if row['weather_type'] == 'radiation_hour' and row['predicted_ahead'] == counter%49:
            values.append(row['value'])
            time.append(row['Date'])
            predicted.append(row['predicted_ahead'])
            counter+=1
    stations_df = pd.DataFrame(columns=['radiation_hour','predicted_ahead','time'])
    stations_df['radiation_hour'] = values
    stations_df['time'] = time
    stations_df['predicted_ahead'] = predicted
    return stations_df
# %%
"""
Henter forecast data fra stationen: Jægersborg.
Jægersborg tilhører grid companiet Radius Elnet. 
Fører det sammen i et datasæt og omregner temperaturen fra Kelvin Celsius
"""
forecast_data = pd.read_parquet("data/forecast_data_jan_maj")
data_temp_val = get_station_temp_val(forecast_data)
data_radi_val = get_station_radi_val(forecast_data)

data_temp_val = data_encoder(data_temp_val)
data_radi_val = data_encoder(data_radi_val)

pred_con = pd.DataFrame()
df_DK2_maj_con = pd.DataFrame()
df_DK2_maj_con['time'] = el_data_2021['HourUTC']
df_DK2_maj_con['Con'] = el_data_2021['HourlySettledConsumption']
df_DK2_maj_con['time'] = pd.to_datetime(df_DK2_maj_con['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
data_temp_val = pd.merge(df_DK2_maj_con,data_temp_val, on='time', how='outer')
data_temp_val

data_temp_val['time'] = data_temp_val['time'].dt.hour
data_radi_val['time'] = data_radi_val['time'].dt.hour

radi_val = data_radi_val['radiation_hour']
stations_concat_df = data_temp_val.join(radi_val)
stations_concat_df['temp_mean_1hr'] = stations_concat_df['temp_mean_1hr'].add(-273.15)

cat_time = pd.get_dummies(stations_concat_df['time'])
stations_concat_df = stations_concat_df.join(cat_time)
stations_concat_df = stations_concat_df.drop(columns=['predicted_ahead','time'])
stations_concat_df.dropna(inplace=True)
stations_concat_df = stations_concat_df.reindex(columns=['temp_mean_1hr',	'radiation_hour',	'is_holiday',	0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	'Con'])
pred_con = stations_concat_df['Con']
stations_concat_df = stations_concat_df.drop(columns=['Con'])
# %%
preds = gbt.predict(stations_concat_df)
# %%
# Plots the predicted values with the exact values to compare how the model predicts.
y_plot = pd.DataFrame()
y_plot['exact_values'] = pred_con[list(pred_con).index() % 24 == 0]
y_plot['predicted_values'] = preds[list(preds).index() % 24 == 0]
y_plot = y_plot.sort_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(range(len(y_plot)), y_plot['exact_values'], 'r-',
         lw=True,label='Exact values')
plt.plot(range(len(y_plot)), y_plot['predicted_values'], 'b-',
         aa=True,label='Precited Values')
plt.legend(loc='upper right')
plt.ylabel('Consumption')
plt.show()
# %%
#Calculates the MSE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(y_plot['exact_values'].to_numpy(),24)
mse_forecast_naive = np.sum((y_plot['exact_values']-naive_y_val)**2)/len(y_plot['exact_values'])
mse_forecast_naive
#%%
#Calculates the MSE for the predicted values from the validation set.
mse_forecast = np.sum((y_plot['exact_values']-preds)**2)/len(y_plot['exact_values'])
mse_forecast
# %%
mse_forecast = mean_absolute_error(pred_con,preds)
mse_forecast
# %%
mse_forecast = mean_absolute_error(pred_con,naive_y_val)
mse_forecast
# %%
mean_absolute_percentage_error(y_plot['exact_values'],naive_y_val)
# %%
