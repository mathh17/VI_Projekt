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
#path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts\data\stations_data_dk2'
path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\VI_Projekt\Scripts\data\stations_data_dk2'

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
old_path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\VI_Projekt\Scripts'
os.chdir(old_path)
df_el_data = pd.read_pickle("data/jaegerspris_el_data.pkl")

#%%
#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_el_data['HourUTC'] = pd.to_datetime(df_el_data['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_el_data = df_el_data.rename(columns={'HourUTC':'time', 'HourlySettledConsumption':'Con'})
conc_data = pd.merge(dk2_mean, df_el_data, on='time', how='outer')
conc_data.dropna(inplace=True)

#%%
#Calling the holiday function to build a column for if its a holiday or not
def holidays(df):
    holidays = []
    for i, row in df.iterrows():
        is_holiday = hc.get_date_type(row['time'])
        holidays.append(is_holiday)
    return holidays

#%%
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
#%%
#Normalize the data so its between 0-1, in this instance i just divided it by the max value of the columns
scaler = preprocessing.MinMaxScaler()
con_scaler = preprocessing.MinMaxScaler()
con_data = np.array(conc_data['Con'])
con_data = con_data.reshape(-1,1)
con_scaled = pd.DataFrame(con_scaler.fit_transform(con_data), columns=['Con'], index=conc_data.index)
pred_data['temp_mean_past1h'] = conc_data['temp_mean_past1h']
pred_data['radia_glob_past1h'] = conc_data['radia_glob_past1h']
scaler.fit(pred_data)
pred_data_scaled = pd.DataFrame(scaler.transform(pred_data),columns=pred_data.columns, index=pred_data.index)
pred_data_scaled['Con'] = con_scaled['Con']
#%%
def create_dataset(df, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size):
    # Feel free to play with shuffle buffer size
    shuffle_buffer_size = len(df)
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    data = data.shuffle(shuffle_buffer_size, seed=42)

    # Extracting past features + deterministic future + labels
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, 0:n_deterministic_features]),
                               k[-forecast_size:, -1]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#%%
# Dividing the complete set into train and test
train_size = 60000
val_size = train_size + 10000
test_size = val_size + 17590

X_train = pred_data_scaled[:train_size]
X_test = pred_data_scaled[val_size:test_size]
X_val = pred_data_scaled[train_size:val_size]

#%%
X_train_windowed = create_dataset(X_train,27,48,24,32)
X_val_windowed = create_dataset(X_val,27,48,24,32)
X_test_windowed = create_dataset(X_test,27,48,24,1)


#%%
# Setting up more layed LSTM which uses the encoding
Latent_dims = 16
past_inputs = tf.keras.Input(shape=(48,28), name='past_inputs')
encoder = layers.LSTM(Latent_dims, return_state=True, dropout=0.2)
encoder_outputs, state_h, state_c = encoder(past_inputs)

future_inputs = tf.keras.Input(shape=(24,27), name='future_inputs')
decoder_lstm = layers.LSTM(Latent_dims, return_sequences=True, dropout=0.2)
non_com_model = decoder_lstm(future_inputs, initial_state=[state_h,state_c])

non_com_model = layers.Dense(Latent_dims,activation='elu')(non_com_model)
non_com_model = layers.Dropout(0.2)(non_com_model)
non_com_model = layers.Dense(Latent_dims,activation='elu')(non_com_model)
non_com_model = layers.Dropout(0.2)(non_com_model)
output = layers.Dense(1,activation='elu')(non_com_model)

model = tf.keras.models.Model(inputs=[past_inputs,future_inputs], outputs=output)
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.001)
loss = tf.keras.losses.Huber()
model.compile(loss=loss,optimizer=optimizer,metrics=['mae'])
model.summary()
#%%
# Fit the model to our data
history = model.fit(X_train_windowed ,epochs=10, validation_data=(X_val_windowed))
# %%
#scores to evaluate how the model performs on the test data
score = model.evaluate(X_test_windowed,verbose=0)
print('loss value: '+str(score[0]))
print('MSE: '+ str(score[1]))

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
# Saving the model to be used later
#model.save('LSTM_Encoder_Decoder_Model_20Epochs')
# %%
loaded_model = keras.models.load_model('LSTM_Encoder_Decoder_Model_250Epochs.h5')

#%%
data_temp_val = data_encoder(data_temp_val)
data_radi_val = data_encoder(data_radi_val)

#%%
df_DK2_maj_con = pd.DataFrame()
df_DK2_maj_con['time'] = df_DK2_maj['HourUTC']
df_DK2_maj_con['Con'] = df_DK2_maj['Con']
df_DK2_maj_con['time'] = pd.to_datetime(df_DK2_maj_con['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
data_temp_val = pd.merge(df_DK2_maj_con,data_temp_val, on='time', how='outer')
data_temp_val
#%%
data_temp_val['time'] = data_temp_val['time'].dt.hour
data_radi_val['time'] = data_radi_val['time'].dt.hour
data_temp_val['time'] = data_temp_val['time'].add(1)
#%%
radi_val = data_radi_val['radiation_hour']
stations_concat_df = data_temp_val.join(radi_val)
stations_concat_df['temp_mean_1hr'] = stations_concat_df['temp_mean_1hr'].add(-273.15)

#%%
cat_time = pd.get_dummies(stations_concat_df['time'])
stations_concat_df = stations_concat_df.join(cat_time)
stations_concat_df = stations_concat_df.drop(columns=['predicted_ahead','time'])
stations_concat_df.dropna(inplace=True)

# %%
scaler_forecast = preprocessing.MinMaxScaler()
scaler_forecast.fit(stations_concat_df)
stations_scaled = pd.DataFrame(scaler_forecast.transform(stations_concat_df),columns=stations_concat_df.columns, index=stations_concat_df.index)
#%%
forecast_windowed_in = create_dataset(stations_scaled,27,48,24,1)
forecast_windowed_in
#%%
forecast_in = stations_scaled[1:49]
forecast_in.reshape(48,48,28)
forecast_out = stations_scaled.drop('Con', axis=1)
forecast_out = forecast_out[48:73]
#%%
predicitions = []
for i, data in enumerate(forecast_windowed_in.take(1)):
    (past, future),truth = data
    predicitions.append(loaded_model.predict((past,future)))



# %%
predictions_loaded = loaded_model.predict(forecast_in,forecast_out)

#%%
preds_rescaled = con_scaler.inverse_transform(predicitions[0][0])
preds_rescaled
# %%
predicitions = []
for i in range(len(preds_rescaled)):
  predicitions.append(preds_rescaled[i][0])
  
  
  # %%
y_plot = pd.DataFrame()
y_plot['exact_values'] = stations_concat_df['Con'][1:25]
y_plot['predicted_values'] = preds_rescaled
y_plot = y_plot.reset_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(np.arange(0,24), y_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(np.arange(0,24), y_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()
#%%
y_plot