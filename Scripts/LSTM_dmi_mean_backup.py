#%%
#Load all the packages
import os
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import Holidays_calc as hc
import tensorflow as tf
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing

#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts\data\stations_data_dk2'
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

#%%
# takes all the columns and calculates the mean for each row. which gives us a mean value for all stations at the given time.
temp_conc_data['mean'] = temp_conc_data.iloc[:,1:12].sum(axis=1) / 11
radi_conc_data['mean'] = radi_conc_data.iloc[:,1:7].sum(axis=1) / 6
dk2_mean = pd.DataFrame()
dk2_mean['time'] = temp_conc_data['time']
dk2_mean['temp_mean_past1h'] = temp_conc_data['mean']
dk2_mean['radia_glob_past1h'] = radi_conc_data['mean']
dk2_mean.head()

#%%
# Read Enernginet Pickle Data
# Change back path
old_path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts'
os.chdir(old_path)
df_DK1_2010_2015 = pd.read_pickle("data/dk1_data_2010_2015.pkl")
df_DK2_2010_2015 = pd.read_pickle("data/dk2_data_2010_2015.pkl")
df_DK1_2015_2020 = pd.read_pickle("data/dk1_data_2015_2020.pkl")
df_DK2_2015_2020 = pd.read_pickle("data/dk2_data_2015_2020.pkl")
df_DK1 = pd.concat([df_DK1_2010_2015,df_DK1_2015_2020], ignore_index=True)
df_DK2 = pd.concat([df_DK2_2010_2015,df_DK2_2015_2020], ignore_index=True)

#%%
#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2['HourUTC'] = pd.to_datetime(df_DK2['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2 = df_DK2.rename(columns={'HourUTC':'time'})
conc_data = pd.merge(dk2_mean, df_DK2, on='time', how='outer')
conc_data.dropna(inplace=True)

#%%
#Calling the holiday function to build a column for if its a holiday or not
holidays = []
for i in range(len(conc_data)):
    is_holiday = hc.get_date_type(conc_data.loc[i,'time'])
    holidays.append(is_holiday)
conc_data['is_holiday'] = holidays
#%%
#Take data from the concatenated dataset and put it into label data and train data
pred_data = conc_data[['temp_mean_past1h','radia_glob_past1h']]
cat_holiday = pd.get_dummies(conc_data['is_holiday'])
pred_data['is_not_holiday'] = cat_holiday[0]
pred_data['is_holiday'] = cat_holiday[1]
conc_data['time'] = conc_data['time'].dt.hour
cat_time = pd.get_dummies(conc_data['time'])
pred_data.join(cat_time)
#%%
#Normalize the data so its between 0-1, in this instance i just divided it by the max value of the columns
scaler = preprocessing.MinMaxScaler()
values = conc_data['Con']
pred_data['temp_mean_past1h'] = conc_data['temp_mean_past1h']
pred_data['radia_glob_past1h'] = conc_data['radia_glob_past1h']
scaler.fit(pred_data)
pred_data_scaled = scaler.transform(pred_data)

#pred_data_scaled.columns=(pred_data.columns)
#%%
# Dividing the complete set into train and test
X_train, X_test, y_train, y_test = train_test_split(pred_data_scaled, values, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#%%
# Setting up very very LoFi LSTM NN
model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
model.add(layers.Dense(1))
model.compile(loss='mse', optimizer='adam')

#%%
# Fit the model to our data
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
# %%
#scores to evaluate how the model performs on the test data
score = model.evaluate(X_test,y_test,verbose=0)
print(str(score[0]))
print(score[1])

#%%
# Plot the test loss and validation loss to check for overfitting
history_dict = history.history
loss_vals = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(loss_vals)+1)

plt.plot(epochs, loss_vals, 'bo')
plt.plot(epochs, val_loss, 'b')
plt.show
# %%
